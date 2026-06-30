import sqlite3

import pytest

from signalforge.config import RuntimeConfig
from signalforge.storage import connect_database


def test_runtime_config_prefers_urls_over_local_paths(monkeypatch):
    monkeypatch.setenv("SIGNALFORGE_DATABASE_URL", "postgresql://localhost/signalforge")
    monkeypatch.setenv("SIGNALFORGE_DB_PATH", "data/custom.sqlite3")
    monkeypatch.setenv("SIGNALFORGE_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("SIGNALFORGE_QDRANT_PATH", "data/custom-qdrant")
    monkeypatch.setenv("SIGNALFORGE_COLLECTION", "custom_chunks")
    monkeypatch.setenv("SIGNALFORGE_EMBEDDING_MODEL", "custom-embedding")
    monkeypatch.setenv("SIGNALFORGE_PLANNER_MODEL", "custom-planner")
    monkeypatch.setenv("SIGNALFORGE_ANSWER_MODEL", "custom-answer")

    config = RuntimeConfig.from_environment()

    assert config.database_target == "postgresql://localhost/signalforge"
    assert config.qdrant_target == "http://localhost:6333"
    assert config.collection == "custom_chunks"
    assert config.embedding_model == "custom-embedding"
    assert config.planner_model == "custom-planner"
    assert config.answer_model == "custom-answer"


def test_runtime_config_falls_back_to_local_paths(monkeypatch):
    monkeypatch.delenv("SIGNALFORGE_DATABASE_URL", raising=False)
    monkeypatch.delenv("SIGNALFORGE_QDRANT_URL", raising=False)
    monkeypatch.setenv("SIGNALFORGE_DB_PATH", "data/local.sqlite3")
    monkeypatch.setenv("SIGNALFORGE_QDRANT_PATH", "data/local-qdrant")

    config = RuntimeConfig.from_environment()

    assert config.database_target == "data/local.sqlite3"
    assert config.qdrant_target == "data/local-qdrant"


def test_connect_database_accepts_sqlite_url(tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"

    with connect_database(f"sqlite:///{db_path}") as connection:
        assert isinstance(connection, sqlite3.Connection)

    assert db_path.exists()


def test_connect_database_rejects_postgres_url_until_storage_migration():
    with pytest.raises(NotImplementedError, match="Postgres access"):
        connect_database("postgresql://localhost/signalforge")
