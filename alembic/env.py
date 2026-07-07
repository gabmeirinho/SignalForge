from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from signalforge.config import RuntimeConfig, sqlalchemy_database_url
from signalforge.models import Base


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def resolved_database_url() -> str:
    configured_url = config.get_main_option("sqlalchemy.url")
    if configured_url and configured_url != "sqlite:///data/signalforge.sqlite3":
        return sqlalchemy_database_url(configured_url)
    return sqlalchemy_database_url(RuntimeConfig.from_environment().database_target)


def run_migrations_offline() -> None:
    context.configure(
        url=resolved_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = resolved_database_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
