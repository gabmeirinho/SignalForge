"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-06-30 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0001_initial_schema"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


EMBEDDING_STATUSES = ("pending", "indexing", "ready", "failed")
SOURCE_TYPES = (
    "company_blog",
    "newsroom",
    "investor_relations",
    "industry_blog",
    "news_feed",
    "sec",
    "webpage",
)
SOURCE_OWNERSHIPS = ("official", "third_party", "unknown")
SOURCE_TRUST_LEVELS = ("high", "medium", "low")
SOURCE_DISCOVERY_STATUSES = ("candidate", "approved", "rejected", "manual")
DOCUMENT_TYPES = (
    "article",
    "blog_post",
    "press_release",
    "investor_update",
    "filing",
    "webpage",
)
SOURCE_INGESTION_STATUSES = ("running", "completed", "failed", "partial")


def upgrade() -> None:
    metadata_type = sa.Text()
    metadata_default = sa.text("'{}'")
    if op.get_bind().dialect.name == "postgresql":
        metadata_type = postgresql.JSONB()
        metadata_default = sa.text("'{}'::jsonb")

    op.create_table(
        "filings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("accession_number", sa.Text(), nullable=False, unique=True),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("cik", sa.Text()),
        sa.Column("company_name", sa.Text()),
        sa.Column("form_type", sa.Text(), nullable=False),
        sa.Column("filing_date", sa.Text()),
        sa.Column("period_of_report", sa.Text()),
        sa.Column("raw_path", sa.Text(), nullable=False),
        sa.Column("raw_sha256", sa.Text(), nullable=False),
        sa.Column("clean_text_path", sa.Text(), nullable=False),
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
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("filing_id", sa.Integer(), nullable=False),
        sa.Column("section_id", sa.Text(), nullable=False),
        sa.Column("section_title", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
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
        sa.ForeignKeyConstraint(["filing_id"], ["filings.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("filing_id", "section_id", "chunk_index"),
    )

    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("vector_collection", sa.Text(), nullable=False),
        sa.Column("vector_id", sa.Text(), nullable=False),
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id", "embedding_model", "vector_collection"),
    )

    op.create_table(
        "embedding_runs",
        sa.Column("filing_id", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("vector_collection", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "expected_point_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "indexed_point_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("error_message", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(_in_constraint("status", EMBEDDING_STATUSES)),
        sa.ForeignKeyConstraint(["filing_id"], ["filings.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("filing_id", "embedding_model", "vector_collection"),
    )

    op.create_table(
        "companies",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.Text(), nullable=False, unique=True),
        sa.Column("name", sa.Text()),
        sa.Column("cik", sa.Text()),
        sa.Column("website_domain", sa.Text()),
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
    )

    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("company_id", sa.Integer()),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False, unique=True),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("ownership", sa.Text(), nullable=False),
        sa.Column("trust_level", sa.Text(), nullable=False),
        sa.Column("discovery_status", sa.Text(), nullable=False),
        sa.Column(
            "enabled",
            sa.Boolean(create_constraint=True, name="ck_sources_enabled_boolean"),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column("confidence_score", sa.Float()),
        sa.Column("discovery_reason", sa.Text()),
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
        sa.CheckConstraint(_in_constraint("source_type", SOURCE_TYPES)),
        sa.CheckConstraint(_in_constraint("ownership", SOURCE_OWNERSHIPS)),
        sa.CheckConstraint(_in_constraint("trust_level", SOURCE_TRUST_LEVELS)),
        sa.CheckConstraint(_in_constraint("discovery_status", SOURCE_DISCOVERY_STATUSES)),
        sa.CheckConstraint("confidence_score IS NULL OR confidence_score BETWEEN 0.0 AND 1.0"),
        sa.ForeignKeyConstraint(["company_id"], ["companies.id"], ondelete="SET NULL"),
    )
    op.create_index("idx_sources_company_id", "sources", ["company_id"])
    op.create_index("idx_sources_discovery_status", "sources", ["discovery_status"])

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text()),
        sa.Column("author", sa.Text()),
        sa.Column("published_at", sa.Text()),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("clean_text_path", sa.Text()),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("document_type", sa.Text(), nullable=False),
        sa.Column("metadata_json", metadata_type, nullable=False, server_default=metadata_default),
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
        sa.CheckConstraint(_in_constraint("document_type", DOCUMENT_TYPES)),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("source_id", "url"),
    )
    op.create_index("idx_documents_source_id", "documents", ["source_id"])
    op.create_index("idx_documents_content_hash", "documents", ["content_hash"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
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
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("document_id", "chunk_index"),
    )

    op.create_table(
        "document_chunk_embeddings",
        sa.Column("document_chunk_id", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("vector_collection", sa.Text(), nullable=False),
        sa.Column("vector_id", sa.Text(), nullable=False),
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_chunk_id"], ["document_chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("document_chunk_id", "embedding_model", "vector_collection"),
    )

    op.create_table(
        "source_ingestion_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column(
            "discovered_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "inserted_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "skipped_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("error_message", sa.Text()),
        sa.CheckConstraint(_in_constraint("status", SOURCE_INGESTION_STATUSES)),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "idx_source_ingestion_runs_source_id",
        "source_ingestion_runs",
        ["source_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_source_ingestion_runs_source_id", table_name="source_ingestion_runs")
    op.drop_table("source_ingestion_runs")
    op.drop_table("document_chunk_embeddings")
    op.drop_table("document_chunks")
    op.drop_index("idx_documents_content_hash", table_name="documents")
    op.drop_index("idx_documents_source_id", table_name="documents")
    op.drop_table("documents")
    op.drop_index("idx_sources_discovery_status", table_name="sources")
    op.drop_index("idx_sources_company_id", table_name="sources")
    op.drop_table("sources")
    op.drop_table("companies")
    op.drop_table("embedding_runs")
    op.drop_table("chunk_embeddings")
    op.drop_table("chunks")
    op.drop_table("filings")


def _in_constraint(column_name: str, values: tuple[str, ...]) -> str:
    choices = ", ".join(f"'{value}'" for value in values)
    return f"{column_name} IN ({choices})"
