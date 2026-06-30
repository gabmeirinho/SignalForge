import subprocess
import sys
from pathlib import Path

from signalforge.storage import (
    CompanyRecord,
    SourceRecord,
    approve_source,
    connect_database,
    initialize_database,
    list_sources,
    reject_source,
    upsert_company,
    upsert_source,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_list_sources_filters_by_ticker_status_and_enabled(tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"
    with connect_database(db_path) as connection:
        initialize_database(connection)
        company_id = upsert_company(connection, CompanyRecord(ticker="NVDA", name="NVIDIA"))
        upsert_source(
            connection,
            SourceRecord(
                company_id=company_id,
                name="NVIDIA Blog",
                url="https://blogs.nvidia.com/feed/",
                source_type="news_feed",
                ownership="official",
                trust_level="high",
                confidence_score=0.95,
            ),
        )
        upsert_source(
            connection,
            SourceRecord(
                name="Third Party Feed",
                url="https://example.com/feed.xml",
                source_type="news_feed",
                ownership="third_party",
                discovery_status="rejected",
                enabled=False,
            ),
        )

        sources = list_sources(
            connection,
            ticker="nvda",
            discovery_status="candidate",
            enabled=True,
        )

    assert len(sources) == 1
    assert sources[0]["name"] == "NVIDIA Blog"
    assert sources[0]["ticker"] == "NVDA"
    assert sources[0]["document_count"] == 0


def test_approve_and_reject_source_are_idempotent(tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"
    with connect_database(db_path) as connection:
        initialize_database(connection)
        source_id = upsert_source(
            connection,
            SourceRecord(
                name="NVIDIA Blog",
                url="https://blogs.nvidia.com/feed/",
                source_type="news_feed",
            ),
        )

        approved_once = approve_source(connection, source_id)
        approved_twice = approve_source(connection, source_id)
        rejected_once = reject_source(connection, source_id)
        rejected_twice = reject_source(connection, source_id)

    assert approved_once["discovery_status"] == "approved"
    assert approved_once["enabled"] == 1
    assert approved_twice["discovery_status"] == "approved"
    assert rejected_once["discovery_status"] == "rejected"
    assert rejected_once["enabled"] == 0
    assert rejected_twice["discovery_status"] == "rejected"


def test_add_source_cli_registers_manual_source(tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "signalforge.cli.add_source",
            "--db-path",
            str(db_path),
            "--name",
            "NVIDIA Blog",
            "--url",
            "https://blogs.nvidia.com/feed/",
            "--source-type",
            "news_feed",
            "--ticker",
            "NVDA",
            "--ownership",
            "official",
            "--trust-level",
            "high",
        ],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    with connect_database(db_path) as connection:
        source = list_sources(connection, ticker="NVDA")[0]

    assert "Registered manual source" in result.stdout
    assert source["name"] == "NVIDIA Blog"
    assert source["ticker"] == "NVDA"
    assert source["discovery_status"] == "manual"
    assert source["enabled"] == 1


def test_list_sources_cli_shows_candidate_reason(tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"
    with connect_database(db_path) as connection:
        initialize_database(connection)
        company_id = upsert_company(connection, CompanyRecord(ticker="NVDA"))
        upsert_source(
            connection,
            SourceRecord(
                company_id=company_id,
                name="NVIDIA Blog",
                url="https://blogs.nvidia.com/feed/",
                source_type="news_feed",
                discovery_reason="RSS/Atom link discovered",
            ),
        )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "signalforge.cli.list_sources",
            "--db-path",
            str(db_path),
            "--ticker",
            "NVDA",
            "--status",
            "candidate",
        ],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert "[1] NVIDIA Blog (NVDA, candidate, enabled=yes)" in result.stdout
    assert "reason: RSS/Atom link discovered" in result.stdout


def test_discover_sources_cli_shows_persisted_status(tmp_path, monkeypatch, capsys):
    from signalforge.cli import discover_sources as discover_sources_cli
    from signalforge.source_discovery import DiscoveredSource

    db_path = tmp_path / "signalforge.sqlite3"

    def fake_discover_sources_for_ticker(**kwargs):
        assert kwargs["persist"] is True
        return [
            DiscoveredSource(
                name="Amazon Press Center",
                url="https://press.aboutamazon.com/",
                final_url="https://press.aboutamazon.com/",
                source_type="newsroom",
                ownership="official",
                trust_level="high",
                confidence_score=0.85,
                discovery_reason="already reviewed",
                status_code=200,
                rss_urls=(),
                persisted_id=37,
                discovery_status="rejected",
                enabled=False,
            )
        ]

    monkeypatch.setattr(
        discover_sources_cli,
        "discover_sources_for_ticker",
        fake_discover_sources_for_ticker,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "discover_sources",
            "--db-path",
            str(db_path),
            "--ticker",
            "AMZN",
        ],
    )

    discover_sources_cli.main()
    captured = capsys.readouterr()

    assert "source_id: 37" in captured.out
    assert "status: rejected" in captured.out
    assert "enabled: no" in captured.out
