from __future__ import annotations

import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from signalforge.config import RuntimeConfig
from signalforge.index_health import IndexHealth, check_index_health
from signalforge.source_ingestion import SourceIngestionResult, ingest_approved_sources
from signalforge.storage import (
    connect_database,
    initialize_database,
    reset_document_index_metadata,
    reset_sec_index_metadata,
)
from signalforge.vector_store import create_qdrant_client
from signalforge.vectorization import VectorizationResult, vectorize_pending_chunks


DEFAULT_WORKER_INTERVAL_SECONDS = 300
DEFAULT_INGEST_LIMIT_PER_SOURCE = None
DEFAULT_ENABLE_SCHEDULED_INGESTION = True
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_CHUNK_SIZE = 4_000
DEFAULT_OVERLAP = 500
DEFAULT_VECTORIZE_BATCH_SIZE = 16

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerConfig:
    database_target: str
    qdrant_target: str
    collection: str
    embedding_model: str
    interval_seconds: int = DEFAULT_WORKER_INTERVAL_SECONDS
    ingest_limit_per_source: int | None = DEFAULT_INGEST_LIMIT_PER_SOURCE
    enable_scheduled_ingestion: bool = DEFAULT_ENABLE_SCHEDULED_INGESTION
    processed_dir: str = DEFAULT_PROCESSED_DIR
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_OVERLAP
    vectorize_batch_size: int = DEFAULT_VECTORIZE_BATCH_SIZE

    @classmethod
    def from_environment(cls) -> "WorkerConfig":
        runtime = RuntimeConfig.from_environment()
        return cls(
            database_target=runtime.database_target,
            qdrant_target=runtime.qdrant_target,
            collection=runtime.collection,
            embedding_model=runtime.embedding_model,
            interval_seconds=_env_int(
                "SIGNALFORGE_WORKER_INTERVAL_SECONDS",
                DEFAULT_WORKER_INTERVAL_SECONDS,
                minimum=1,
            ),
            ingest_limit_per_source=_env_optional_int(
                "SIGNALFORGE_INGEST_LIMIT_PER_SOURCE",
                minimum=1,
            ),
            enable_scheduled_ingestion=_env_bool(
                "SIGNALFORGE_ENABLE_SCHEDULED_INGESTION",
                DEFAULT_ENABLE_SCHEDULED_INGESTION,
            ),
            processed_dir=os.getenv("SIGNALFORGE_PROCESSED_DIR", DEFAULT_PROCESSED_DIR),
            chunk_size=_env_int("SIGNALFORGE_CHUNK_SIZE", DEFAULT_CHUNK_SIZE, minimum=1),
            overlap=_env_int("SIGNALFORGE_CHUNK_OVERLAP", DEFAULT_OVERLAP, minimum=0),
            vectorize_batch_size=_env_int(
                "SIGNALFORGE_VECTORIZE_BATCH_SIZE",
                DEFAULT_VECTORIZE_BATCH_SIZE,
                minimum=1,
            ),
        )


@dataclass(frozen=True)
class WorkerCycleResult:
    ingestion_results: list[SourceIngestionResult]
    vectorization_result: VectorizationResult


def run_worker_cycle(config: WorkerConfig) -> WorkerCycleResult:
    logger.info("worker cycle started")
    ingestion_results: list[SourceIngestionResult] = []

    with connect_database(config.database_target) as connection:
        initialize_database(connection)

        if config.enable_scheduled_ingestion:
            ingestion_results = ingest_approved_sources(
                connection,
                processed_dir=Path(config.processed_dir),
                chunk_size=config.chunk_size,
                overlap=config.overlap,
                limit_per_source=config.ingest_limit_per_source,
            )
            _log_ingestion_results(ingestion_results)
        else:
            logger.info("scheduled ingestion disabled")

        repair_degraded_index(connection, config)

        vectorization_result = vectorize_pending_chunks(
            connection,
            qdrant_target=config.qdrant_target,
            collection=config.collection,
            embedding_model=config.embedding_model,
            batch_size=config.vectorize_batch_size,
        )
        logger.info(
            "vectorization completed: indexed=%s sec_chunks=%s document_chunks=%s",
            vectorization_result.indexed_count,
            vectorization_result.sec_chunk_count,
            vectorization_result.document_chunk_count,
        )

    logger.info("worker cycle completed")
    return WorkerCycleResult(
        ingestion_results=ingestion_results,
        vectorization_result=vectorization_result,
    )


def repair_degraded_index(connection, config: WorkerConfig) -> IndexHealth:
    health = load_worker_index_health(connection, config)
    logger.info(
        "index health: status=%s sec_pg=%s sec_qdrant=%s documents_pg=%s documents_qdrant=%s",
        health.status,
        health.sec.postgres_embedding_records,
        health.sec.qdrant_points,
        health.documents.postgres_embedding_records,
        health.documents.qdrant_points,
    )

    if health.status != "degraded":
        return health

    repaired = []
    if not health.sec.is_consistent_with_qdrant:
        reset_sec_index_metadata(
            connection,
            embedding_model=config.embedding_model,
            vector_collection=config.collection,
        )
        repaired.append("sec")

    if not health.documents.is_consistent_with_qdrant:
        reset_document_index_metadata(
            connection,
            embedding_model=config.embedding_model,
            vector_collection=config.collection,
        )
        repaired.append("documents")

    logger.warning(
        "degraded index detected; reset metadata for %s before vectorization",
        ", ".join(repaired),
    )
    return health


def load_worker_index_health(connection, config: WorkerConfig) -> IndexHealth:
    client = create_qdrant_client(config.qdrant_target)
    try:
        return check_index_health(
            connection,
            client,
            collection=config.collection,
            embedding_model=config.embedding_model,
        )
    finally:
        client.close()


def run_worker_loop(config: WorkerConfig, stop_event: _StopEvent | None = None) -> None:
    stop_event = stop_event or _StopEvent()
    logger.info("worker started with interval=%ss", config.interval_seconds)

    while not stop_event.is_set:
        try:
            run_worker_cycle(config)
        except Exception:
            logger.exception("worker cycle failed")

        if stop_event.wait(config.interval_seconds):
            break

    logger.info("worker stopped")


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=os.getenv("SIGNALFORGE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    stop_event = _StopEvent()

    def request_stop(signum, frame) -> None:
        logger.info("received signal %s; stopping worker", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    run_worker_loop(WorkerConfig.from_environment(), stop_event=stop_event)


class _StopEvent:
    def __init__(self) -> None:
        self.is_set = False

    def set(self) -> None:
        self.is_set = True

    def wait(self, seconds: int) -> bool:
        slept = 0.0
        step = 0.2
        while slept < seconds and not self.is_set:
            interval = min(step, seconds - slept)
            time.sleep(interval)
            slept += interval
        return self.is_set


def _log_ingestion_results(results: list[SourceIngestionResult]) -> None:
    if not results:
        logger.info("no approved enabled sources found")
        return

    for result in results:
        log = logger.warning if result.status in {"failed", "partial"} else logger.info
        log(
            "source ingestion: source_id=%s name=%r status=%s discovered=%s inserted=%s "
            "skipped=%s error=%r",
            result.source_id,
            result.source_name,
            result.status,
            result.discovered_count,
            result.inserted_count,
            result.skipped_count,
            result.error_message,
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false, yes/no, 1/0, on/off")


def _env_int(name: str, default: int, *, minimum: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def _env_optional_int(name: str, *, minimum: int) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


if __name__ == "__main__":
    main()
