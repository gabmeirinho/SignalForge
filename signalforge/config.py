import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from signalforge.answer_generator import DEFAULT_ANSWER_MODEL
from signalforge.query_planner import DEFAULT_PLANNER_MODEL
from signalforge.vector_store import DEFAULT_COLLECTION, DEFAULT_EMBEDDING_MODEL


DEFAULT_DB_PATH = "data/signalforge.sqlite3"
DEFAULT_QDRANT_PATH = "data/qdrant"


@dataclass(frozen=True)
class RuntimeConfig:
    database_url: str | None = None
    db_path: str = DEFAULT_DB_PATH
    qdrant_url: str | None = None
    qdrant_path: str = DEFAULT_QDRANT_PATH
    collection: str = DEFAULT_COLLECTION
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    planner_model: str = DEFAULT_PLANNER_MODEL
    answer_model: str = DEFAULT_ANSWER_MODEL

    @classmethod
    def from_environment(cls) -> "RuntimeConfig":
        return cls(
            database_url=_empty_to_none(os.getenv("SIGNALFORGE_DATABASE_URL")),
            db_path=os.getenv("SIGNALFORGE_DB_PATH", cls.db_path),
            qdrant_url=_empty_to_none(os.getenv("SIGNALFORGE_QDRANT_URL")),
            qdrant_path=os.getenv("SIGNALFORGE_QDRANT_PATH", cls.qdrant_path),
            collection=os.getenv("SIGNALFORGE_COLLECTION", cls.collection),
            embedding_model=os.getenv("SIGNALFORGE_EMBEDDING_MODEL", cls.embedding_model),
            planner_model=os.getenv("SIGNALFORGE_PLANNER_MODEL", cls.planner_model),
            answer_model=os.getenv("SIGNALFORGE_ANSWER_MODEL", cls.answer_model),
        )

    @property
    def database_target(self) -> str:
        return self.database_url or self.db_path

    @property
    def qdrant_target(self) -> str:
        return self.qdrant_url or self.qdrant_path

    @property
    def uses_database_url(self) -> bool:
        return self.database_url is not None

    @property
    def uses_qdrant_url(self) -> bool:
        return self.qdrant_url is not None


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def target_exists(target: str) -> bool:
    parsed = urlparse(target)
    if parsed.scheme in {"http", "https"}:
        return True
    if parsed.scheme == "sqlite":
        return Path(unquote(parsed.path)).exists()
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).exists()
    if parsed.scheme:
        return True
    return Path(target).exists()


def sqlalchemy_database_url(target: str) -> str:
    parsed = urlparse(target)

    if parsed.scheme == "":
        db_path = Path(target)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{quote(str(db_path), safe='/')}"

    if parsed.scheme == "file":
        db_path = Path(unquote(parsed.path))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{quote(str(db_path), safe='/')}"

    if parsed.scheme == "sqlite":
        db_path = Path(unquote(parsed.path))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return target

    if parsed.scheme in {"postgres", "postgresql"}:
        return target.replace(f"{parsed.scheme}://", "postgresql+psycopg://", 1)

    return target
