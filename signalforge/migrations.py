from pathlib import Path

from alembic import command
from alembic.config import Config

from signalforge.config import sqlalchemy_database_url


def upgrade_database(target: str | Path, revision: str = "head") -> None:
    command.upgrade(alembic_config(target), revision)


def alembic_config(target: str | Path) -> Config:
    root = Path(__file__).resolve().parents[1]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    cfg.set_main_option("sqlalchemy.url", sqlalchemy_database_url(str(target)))
    return cfg
