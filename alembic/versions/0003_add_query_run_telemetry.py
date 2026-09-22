"""add nullable query-run telemetry

Revision ID: 0003_add_query_run_telemetry
Revises: 0002_add_research_sessions_query_runs
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0003_add_query_run_telemetry"
down_revision: Union[str, Sequence[str], None] = "0002_add_research_sessions_query_runs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


COLUMNS = (
    ("planning_ms", sa.Float()),
    ("retrieval_ms", sa.Float()),
    ("answer_ms", sa.Float()),
    ("total_ms", sa.Float()),
    ("planner_provider", sa.Text()),
    ("answer_provider", sa.Text()),
    ("planner_prompt_version", sa.Text()),
    ("answer_prompt_version", sa.Text()),
    ("token_usage_status", sa.Text()),
    ("prompt_tokens", sa.Integer()),
    ("completion_tokens", sa.Integer()),
    ("price_catalog_version", sa.Text()),
    ("price_tier", sa.Text()),
    ("estimated_cost_usd", sa.Numeric(14, 8)),
    ("retry_count", sa.Integer()),
    ("fallback_count", sa.Integer()),
    ("error_category", sa.Text()),
)


def upgrade() -> None:
    for name, column_type in COLUMNS:
        op.add_column("query_runs", sa.Column(name, column_type, nullable=True))
    json_type = postgresql.JSONB() if op.get_bind().dialect.name == "postgresql" else sa.JSON()
    op.add_column("query_runs", sa.Column("token_usage_json", json_type, nullable=True))


def downgrade() -> None:
    op.drop_column("query_runs", "token_usage_json")
    for name, _ in reversed(COLUMNS):
        op.drop_column("query_runs", name)
