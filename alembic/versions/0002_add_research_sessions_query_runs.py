"""add research sessions and query runs

Revision ID: 0002_add_research_sessions_query_runs
Revises: 0001_initial_schema
Create Date: 2026-07-06 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0002_add_research_sessions_query_runs"
down_revision: Union[str, Sequence[str], None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


QUERY_RUN_STATUSES = ("running", "completed", "failed")


def upgrade() -> None:
    json_type = sa.JSON()
    json_default = sa.text("'{}'")
    if op.get_bind().dialect.name == "postgresql":
        json_type = postgresql.JSONB()
        json_default = sa.text("'{}'::jsonb")

    op.create_table(
        "research_sessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_key", sa.Text(), nullable=False, unique=True),
        sa.Column("title", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("metadata_json", json_type, nullable=False, server_default=json_default),
    )

    op.create_table(
        "query_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("research_session_id", sa.Integer()),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("planner_model", sa.Text()),
        sa.Column("answer_model", sa.Text()),
        sa.Column("embedding_model", sa.Text()),
        sa.Column("vector_collection", sa.Text()),
        sa.Column("planned_query_json", json_type, nullable=False, server_default=json_default),
        sa.Column("retrieval_metadata_json", json_type, nullable=False, server_default=json_default),
        sa.Column("answer_text", sa.Text()),
        sa.Column("error_message", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(_in_constraint("status", QUERY_RUN_STATUSES)),
        sa.ForeignKeyConstraint(
            ["research_session_id"],
            ["research_sessions.id"],
            ondelete="SET NULL",
        ),
    )
    op.create_index(
        "idx_query_runs_research_session_id",
        "query_runs",
        ["research_session_id"],
    )
    op.create_index("idx_query_runs_status", "query_runs", ["status"])
    op.create_index("idx_query_runs_started_at", "query_runs", ["started_at"])


def downgrade() -> None:
    op.drop_index("idx_query_runs_started_at", table_name="query_runs")
    op.drop_index("idx_query_runs_status", table_name="query_runs")
    op.drop_index("idx_query_runs_research_session_id", table_name="query_runs")
    op.drop_table("query_runs")
    op.drop_table("research_sessions")


def _in_constraint(column_name: str, values: tuple[str, ...]) -> str:
    choices = ", ".join(f"'{value}'" for value in values)
    return f"{column_name} IN ({choices})"
