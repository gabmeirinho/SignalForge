PYTHON ?= uv run python
ALEMBIC ?= uv run alembic
DOCKER_COMPOSE ?= docker compose

DB_PATH ?= data/signalforge.sqlite3
POSTGRES_URL ?= postgresql+psycopg://signalforge:signalforge@localhost:15432/signalforge
POSTGRES_USER ?= signalforge
POSTGRES_DB ?= signalforge
QDRANT_URL ?= http://localhost:6333
API_PORT ?= 8000
FRONTEND_PORT ?= 5173

.PHONY: help sync test lint migrate api frontend worker worker-once ingest-sources vectorize index-state compose-up compose-down compose-logs backup-postgres restore-postgres

help:
	@printf '%s\n' \
		'SignalForge operations:' \
		'  make sync              Install Python dependencies' \
		'  make test              Run Python tests' \
		'  make lint              Run Ruff checks' \
		'  make migrate           Run Alembic migrations for current env' \
		'  make api               Start local API with reload' \
		'  make frontend          Start local Vite frontend' \
		'  make worker            Start local worker loop' \
		'  make worker-once       Run one worker cycle' \
		'  make ingest-sources    Ingest approved sources once' \
		'  make vectorize         Vectorize pending chunks once' \
		'  make index-state       Print index/database state' \
		'  make compose-up        Start full Docker Compose stack' \
		'  make compose-down      Stop Compose stack and keep volumes' \
		'  make compose-logs      Follow API and worker logs' \
		'  make backup-postgres   Write backups/signalforge.dump from Compose Postgres' \
		'  make restore-postgres  Restore backups/signalforge.dump into Compose Postgres'

sync:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check signalforge tests

migrate:
	$(ALEMBIC) -c alembic.ini upgrade head

api:
	uv run uvicorn signalforge.api:app --reload --port $(API_PORT)

frontend:
	cd frontend && npm run dev -- --host 127.0.0.1 --port $(FRONTEND_PORT)

worker:
	$(PYTHON) -m signalforge.worker

worker-once:
	$(PYTHON) -m signalforge.cli.run_worker_once

ingest-sources:
	$(PYTHON) -m signalforge.cli.ingest_sources

vectorize:
	$(PYTHON) -m signalforge.cli.vectorize

index-state:
	$(PYTHON) -m signalforge.cli.index_state

compose-up:
	$(DOCKER_COMPOSE) up --build

compose-down:
	$(DOCKER_COMPOSE) down

compose-logs:
	$(DOCKER_COMPOSE) logs -f api worker

backup-postgres:
	mkdir -p backups
	$(DOCKER_COMPOSE) exec -T postgres pg_dump -U $(POSTGRES_USER) -d $(POSTGRES_DB) --format=custom > backups/signalforge.dump

restore-postgres:
	$(DOCKER_COMPOSE) exec -T postgres pg_restore -U $(POSTGRES_USER) -d $(POSTGRES_DB) --clean --if-exists --no-owner < backups/signalforge.dump
