from signalforge.index_health import CorpusIndexHealth, IndexHealth
from signalforge.source_ingestion import SourceIngestionResult
from signalforge.vectorization import VectorizationResult
from signalforge.worker import WorkerConfig, run_worker_cycle, run_worker_loop


def test_worker_config_reads_environment(monkeypatch):
    monkeypatch.setenv("SIGNALFORGE_DATABASE_URL", "postgresql://localhost/signalforge")
    monkeypatch.setenv("SIGNALFORGE_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("SIGNALFORGE_COLLECTION", "worker_chunks")
    monkeypatch.setenv("SIGNALFORGE_EMBEDDING_MODEL", "worker-model")
    monkeypatch.setenv("SIGNALFORGE_WORKER_INTERVAL_SECONDS", "15")
    monkeypatch.setenv("SIGNALFORGE_INGEST_LIMIT_PER_SOURCE", "3")
    monkeypatch.setenv("SIGNALFORGE_ENABLE_SCHEDULED_INGESTION", "false")
    monkeypatch.setenv("SIGNALFORGE_PROCESSED_DIR", "data/worker-processed")
    monkeypatch.setenv("SIGNALFORGE_CHUNK_SIZE", "1200")
    monkeypatch.setenv("SIGNALFORGE_CHUNK_OVERLAP", "100")
    monkeypatch.setenv("SIGNALFORGE_VECTORIZE_BATCH_SIZE", "8")

    config = WorkerConfig.from_environment()

    assert config.database_target == "postgresql://localhost/signalforge"
    assert config.qdrant_target == "http://localhost:6333"
    assert config.collection == "worker_chunks"
    assert config.embedding_model == "worker-model"
    assert config.interval_seconds == 15
    assert config.ingest_limit_per_source == 3
    assert config.enable_scheduled_ingestion is False
    assert config.processed_dir == "data/worker-processed"
    assert config.chunk_size == 1200
    assert config.overlap == 100
    assert config.vectorize_batch_size == 8


def test_run_worker_cycle_ingests_and_vectorizes(monkeypatch, tmp_path):
    calls = {}

    def fake_ingest_approved_sources(connection, **kwargs):
        calls["ingest"] = kwargs
        return [
            SourceIngestionResult(
                source_id=1,
                source_name="Example Feed",
                status="completed",
                discovered_count=2,
                inserted_count=1,
                skipped_count=1,
            )
        ]

    def fake_vectorize_pending_chunks(connection, **kwargs):
        calls["vectorize"] = kwargs
        return VectorizationResult(
            indexed_count=4,
            sec_chunk_count=2,
            document_chunk_count=2,
        )

    monkeypatch.setattr("signalforge.worker.ingest_approved_sources", fake_ingest_approved_sources)
    monkeypatch.setattr("signalforge.worker.vectorize_pending_chunks", fake_vectorize_pending_chunks)

    config = WorkerConfig(
        database_target=str(tmp_path / "signalforge.sqlite3"),
        qdrant_target=str(tmp_path / "qdrant"),
        collection="worker_chunks",
        embedding_model="worker-model",
        ingest_limit_per_source=5,
        processed_dir=str(tmp_path / "processed"),
        chunk_size=500,
        overlap=50,
        vectorize_batch_size=7,
    )

    result = run_worker_cycle(config)

    assert len(result.ingestion_results) == 1
    assert result.vectorization_result.indexed_count == 4
    assert calls["ingest"]["limit_per_source"] == 5
    assert str(calls["ingest"]["processed_dir"]) == str(tmp_path / "processed")
    assert calls["ingest"]["chunk_size"] == 500
    assert calls["ingest"]["overlap"] == 50
    assert calls["vectorize"] == {
        "qdrant_target": str(tmp_path / "qdrant"),
        "collection": "worker_chunks",
        "embedding_model": "worker-model",
        "batch_size": 7,
    }


def test_run_worker_cycle_can_skip_scheduled_ingestion(monkeypatch, tmp_path):
    def fail_ingestion(connection, **kwargs):
        raise AssertionError("scheduled ingestion should not run")

    monkeypatch.setattr("signalforge.worker.ingest_approved_sources", fail_ingestion)
    monkeypatch.setattr(
        "signalforge.worker.vectorize_pending_chunks",
        lambda connection, **kwargs: VectorizationResult(0, 0, 0),
    )

    result = run_worker_cycle(
        WorkerConfig(
            database_target=str(tmp_path / "signalforge.sqlite3"),
            qdrant_target=str(tmp_path / "qdrant"),
            collection="worker_chunks",
            embedding_model="worker-model",
            enable_scheduled_ingestion=False,
        )
    )

    assert result.ingestion_results == []
    assert result.vectorization_result.indexed_count == 0


def test_run_worker_cycle_repairs_degraded_index_before_vectorizing(monkeypatch, tmp_path):
    calls = []

    def degraded_health(connection, config):
        return IndexHealth(
            status="degraded",
            collection=config.collection,
            collection_exists=True,
            embedding_model=config.embedding_model,
            sec=CorpusIndexHealth(
                name="sec",
                postgres_expected_points=2,
                postgres_ready_points=2,
                postgres_embedding_records=2,
                qdrant_points=0,
            ),
            documents=CorpusIndexHealth(
                name="documents",
                postgres_expected_points=3,
                postgres_ready_points=3,
                postgres_embedding_records=3,
                qdrant_points=3,
            ),
        )

    def reset_sec(connection, **kwargs):
        calls.append(("reset_sec", kwargs))

    def reset_documents(connection, **kwargs):
        calls.append(("reset_documents", kwargs))

    def vectorize(connection, **kwargs):
        calls.append(("vectorize", kwargs))
        return VectorizationResult(indexed_count=2, sec_chunk_count=2, document_chunk_count=0)

    monkeypatch.setattr("signalforge.worker.load_worker_index_health", degraded_health)
    monkeypatch.setattr("signalforge.worker.reset_sec_index_metadata", reset_sec)
    monkeypatch.setattr("signalforge.worker.reset_document_index_metadata", reset_documents)
    monkeypatch.setattr("signalforge.worker.vectorize_pending_chunks", vectorize)

    result = run_worker_cycle(
        WorkerConfig(
            database_target=str(tmp_path / "signalforge.sqlite3"),
            qdrant_target=str(tmp_path / "qdrant"),
            collection="worker_chunks",
            embedding_model="worker-model",
            enable_scheduled_ingestion=False,
            vectorize_batch_size=7,
        )
    )

    assert result.vectorization_result.indexed_count == 2
    assert calls == [
        (
            "reset_sec",
            {
                "embedding_model": "worker-model",
                "vector_collection": "worker_chunks",
            },
        ),
        (
            "vectorize",
            {
                "qdrant_target": str(tmp_path / "qdrant"),
                "collection": "worker_chunks",
                "embedding_model": "worker-model",
                "batch_size": 7,
            },
        ),
    ]


def test_worker_loop_continues_after_cycle_failure(monkeypatch):
    calls = {"count": 0}

    def fake_cycle(config):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("first cycle failed")

    class StopAfterTwoWaits:
        is_set = False

        def __init__(self):
            self.wait_count = 0

        def wait(self, seconds):
            self.wait_count += 1
            return self.wait_count >= 2

    monkeypatch.setattr("signalforge.worker.run_worker_cycle", fake_cycle)

    run_worker_loop(
        WorkerConfig(
            database_target="unused.sqlite3",
            qdrant_target="unused-qdrant",
            collection="worker_chunks",
            embedding_model="worker-model",
            interval_seconds=1,
        ),
        stop_event=StopAfterTwoWaits(),
    )

    assert calls["count"] == 2
