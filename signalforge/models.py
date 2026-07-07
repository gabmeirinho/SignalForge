from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy import ForeignKey
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


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
QUERY_RUN_STATUSES = ("running", "completed", "failed")


class JSONVariant(TypeDecorator):
    impl = sa.JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(postgresql.JSONB())
        return dialect.type_descriptor(sa.JSON())


class Base(DeclarativeBase):
    pass


def in_constraint(column_name: str, values: tuple[str, ...]) -> str:
    choices = ", ".join(f"'{value}'" for value in values)
    return f"{column_name} IN ({choices})"


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(sa.Text, nullable=False, unique=True)
    name: Mapped[str | None] = mapped_column(sa.Text)
    cik: Mapped[str | None] = mapped_column(sa.Text)
    website_domain: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )

    sources: Mapped[list[Source]] = relationship(back_populates="company")


class Filing(Base):
    __tablename__ = "filings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    accession_number: Mapped[str] = mapped_column(sa.Text, nullable=False, unique=True)
    ticker: Mapped[str] = mapped_column(sa.Text, nullable=False)
    cik: Mapped[str | None] = mapped_column(sa.Text)
    company_name: Mapped[str | None] = mapped_column(sa.Text)
    form_type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    filing_date: Mapped[str | None] = mapped_column(sa.Text)
    period_of_report: Mapped[str | None] = mapped_column(sa.Text)
    raw_path: Mapped[str] = mapped_column(sa.Text, nullable=False)
    raw_sha256: Mapped[str] = mapped_column(sa.Text, nullable=False)
    clean_text_path: Mapped[str] = mapped_column(sa.Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )

    chunks: Mapped[list[Chunk]] = relationship(back_populates="filing", cascade="all, delete-orphan")
    embedding_runs: Mapped[list[EmbeddingRun]] = relationship(
        back_populates="filing", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (sa.UniqueConstraint("filing_id", "section_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filing_id: Mapped[int] = mapped_column(ForeignKey("filings.id", ondelete="CASCADE"), nullable=False)
    section_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    section_title: Mapped[str] = mapped_column(sa.Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    char_count: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )

    filing: Mapped[Filing] = relationship(back_populates="chunks")
    embeddings: Mapped[list[ChunkEmbedding]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    embedding_model: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    vector_collection: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    vector_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    embedded_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)

    chunk: Mapped[Chunk] = relationship(back_populates="embeddings")


class EmbeddingRun(Base):
    __tablename__ = "embedding_runs"
    __table_args__ = (sa.CheckConstraint(in_constraint("status", EMBEDDING_STATUSES)),)

    filing_id: Mapped[int] = mapped_column(
        ForeignKey("filings.id", ondelete="CASCADE"), primary_key=True
    )
    embedding_model: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    vector_collection: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False)
    expected_point_count: Mapped[int] = mapped_column(nullable=False, server_default=sa.text("0"))
    indexed_point_count: Mapped[int] = mapped_column(nullable=False, server_default=sa.text("0"))
    error_message: Mapped[str | None] = mapped_column(sa.Text)
    started_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)

    filing: Mapped[Filing] = relationship(back_populates="embedding_runs")


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (
        sa.CheckConstraint(in_constraint("source_type", SOURCE_TYPES)),
        sa.CheckConstraint(in_constraint("ownership", SOURCE_OWNERSHIPS)),
        sa.CheckConstraint(in_constraint("trust_level", SOURCE_TRUST_LEVELS)),
        sa.CheckConstraint(in_constraint("discovery_status", SOURCE_DISCOVERY_STATUSES)),
        sa.CheckConstraint("confidence_score IS NULL OR confidence_score BETWEEN 0.0 AND 1.0"),
        sa.Index("idx_sources_company_id", "company_id"),
        sa.Index("idx_sources_discovery_status", "discovery_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id", ondelete="SET NULL"))
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    url: Mapped[str] = mapped_column(sa.Text, nullable=False, unique=True)
    source_type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    ownership: Mapped[str] = mapped_column(sa.Text, nullable=False)
    trust_level: Mapped[str] = mapped_column(sa.Text, nullable=False)
    discovery_status: Mapped[str] = mapped_column(sa.Text, nullable=False)
    enabled: Mapped[bool] = mapped_column(
        sa.Boolean(create_constraint=True, name="ck_sources_enabled_boolean"),
        nullable=False,
        server_default=sa.true(),
    )
    confidence_score: Mapped[float | None] = mapped_column(sa.Float)
    discovery_reason: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )

    company: Mapped[Company | None] = relationship(back_populates="sources")
    documents: Mapped[list[Document]] = relationship(back_populates="source", cascade="all, delete-orphan")
    ingestion_runs: Mapped[list[SourceIngestionRun]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        sa.CheckConstraint(in_constraint("document_type", DOCUMENT_TYPES)),
        sa.UniqueConstraint("source_id", "url"),
        sa.Index("idx_documents_source_id", "source_id"),
        sa.Index("idx_documents_content_hash", "content_hash"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), nullable=False)
    url: Mapped[str] = mapped_column(sa.Text, nullable=False)
    title: Mapped[str | None] = mapped_column(sa.Text)
    author: Mapped[str | None] = mapped_column(sa.Text)
    published_at: Mapped[str | None] = mapped_column(sa.Text)
    fetched_at: Mapped[datetime | str] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    clean_text_path: Mapped[str | None] = mapped_column(sa.Text)
    content_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)
    document_type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONVariant, nullable=False, server_default=sa.text("'{}'")
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )

    source: Mapped[Source] = relationship(back_populates="documents")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (sa.UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    char_count: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
    embeddings: Mapped[list[DocumentChunkEmbedding]] = relationship(
        back_populates="document_chunk", cascade="all, delete-orphan"
    )


class DocumentChunkEmbedding(Base):
    __tablename__ = "document_chunk_embeddings"

    document_chunk_id: Mapped[int] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE"), primary_key=True
    )
    embedding_model: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    vector_collection: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    vector_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    embedded_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)

    document_chunk: Mapped[DocumentChunk] = relationship(back_populates="embeddings")


class SourceIngestionRun(Base):
    __tablename__ = "source_ingestion_runs"
    __table_args__ = (
        sa.CheckConstraint(in_constraint("status", SOURCE_INGESTION_STATUSES)),
        sa.Index("idx_source_ingestion_runs_source_id", "source_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False)
    started_at: Mapped[datetime | str] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | str | None] = mapped_column(sa.DateTime(timezone=True))
    discovered_count: Mapped[int] = mapped_column(nullable=False, server_default=sa.text("0"))
    inserted_count: Mapped[int] = mapped_column(nullable=False, server_default=sa.text("0"))
    skipped_count: Mapped[int] = mapped_column(nullable=False, server_default=sa.text("0"))
    error_message: Mapped[str | None] = mapped_column(sa.Text)

    source: Mapped[Source] = relationship(back_populates="ingestion_runs")


class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_key: Mapped[str] = mapped_column(sa.Text, nullable=False, unique=True)
    title: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONVariant, nullable=False, server_default=sa.text("'{}'")
    )

    query_runs: Mapped[list[QueryRun]] = relationship(back_populates="research_session")


class QueryRun(Base):
    __tablename__ = "query_runs"
    __table_args__ = (
        sa.CheckConstraint(in_constraint("status", QUERY_RUN_STATUSES)),
        sa.Index("idx_query_runs_research_session_id", "research_session_id"),
        sa.Index("idx_query_runs_status", "status"),
        sa.Index("idx_query_runs_started_at", "started_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    research_session_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_sessions.id", ondelete="SET NULL")
    )
    question: Mapped[str] = mapped_column(sa.Text, nullable=False)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False)
    planner_model: Mapped[str | None] = mapped_column(sa.Text)
    answer_model: Mapped[str | None] = mapped_column(sa.Text)
    embedding_model: Mapped[str | None] = mapped_column(sa.Text)
    vector_collection: Mapped[str | None] = mapped_column(sa.Text)
    planned_query_json: Mapped[dict[str, Any]] = mapped_column(
        JSONVariant, nullable=False, server_default=sa.text("'{}'")
    )
    retrieval_metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONVariant, nullable=False, server_default=sa.text("'{}'")
    )
    answer_text: Mapped[str | None] = mapped_column(sa.Text)
    error_message: Mapped[str | None] = mapped_column(sa.Text)
    started_at: Mapped[datetime | str] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | str | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )

    research_session: Mapped[ResearchSession | None] = relationship(back_populates="query_runs")
