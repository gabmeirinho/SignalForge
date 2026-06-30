import pytest

from signalforge.source_discovery import (
    FetchResult,
    classify_fetch_result,
    discover_sources_for_ticker,
    domain_root_is_reachable,
    generate_domain_candidates_from_company_name,
    generate_candidate_urls,
    normalize_domain,
)
from signalforge.storage import (
    CompanyRecord,
    FilingMetadata,
    SourceRecord,
    connect_database,
    initialize_database,
    upsert_filing,
    upsert_company,
    upsert_source,
)


class FakeFetcher:
    def __init__(self, responses: dict[str, FetchResult]) -> None:
        self.responses = responses

    def fetch(self, url: str) -> FetchResult:
        return self.responses.get(
            url,
            FetchResult(
                url=url,
                final_url=url,
                status_code=404,
                content_type="text/html",
                text="not found",
            ),
        )


def test_generate_candidate_urls_includes_official_paths_and_subdomains():
    urls = generate_candidate_urls("https://www.nvidia.com/")

    assert "https://nvidia.com/blog" in urls
    assert "https://www.nvidia.com/investor-relations" in urls
    assert "https://blogs.nvidia.com/" in urls
    assert "https://investor.nvidia.com/" in urls


def test_generate_domain_candidates_from_sec_company_name():
    assert generate_domain_candidates_from_company_name("NVIDIA CORP") == ["nvidia.com"]
    assert generate_domain_candidates_from_company_name("AMAZON COM INC") == ["amazon.com"]
    assert generate_domain_candidates_from_company_name("META PLATFORMS, INC.") == [
        "meta.com",
        "metaplatforms.com",
        "meta-platforms.com",
    ]
    assert generate_domain_candidates_from_company_name("BERKSHIRE HATHAWAY INC") == [
        "berkshire.com",
        "berkshirehathaway.com",
        "berkshire-hathaway.com",
    ]


def test_domain_root_reachability_accepts_2xx_responses():
    result = FetchResult(
        url="https://example.com/",
        final_url="https://example.com/",
        status_code=200,
        content_type="text/html",
        text="<html><title>Example</title><body>This domain may be for sale.</body></html>",
    )

    assert domain_root_is_reachable(result) is True


def test_domain_root_reachability_accepts_protected_roots():
    result = FetchResult(
        url="https://example.com/",
        final_url="https://www.example.com/",
        status_code=403,
        content_type="text/html",
        text="<html><title>Access Denied</title></html>",
    )

    assert domain_root_is_reachable(result) is True


def test_classify_fetch_result_scores_official_blog_with_feed():
    result = FetchResult(
        url="https://blogs.nvidia.com/",
        final_url="https://blogs.nvidia.com/",
        status_code=200,
        content_type="text/html",
        text="""
        <html>
          <head>
            <title>NVIDIA Blog</title>
            <link rel="alternate" type="application/rss+xml" href="/feed/" />
          </head>
          <body><main>{body}</main></body>
        </html>
        """.format(body="AI infrastructure update. " * 20),
    )

    source = classify_fetch_result(result, official_domain="nvidia.com")

    assert source is not None
    assert source.url == "https://blogs.nvidia.com/feed/"
    assert source.source_type == "news_feed"
    assert source.ownership == "official"
    assert source.trust_level == "high"
    assert source.confidence_score == 1.0
    assert "RSS/Atom link discovered" in source.discovery_reason


def test_classify_fetch_result_adds_direct_feed_url_bonus():
    result = FetchResult(
        url="https://www.aboutamazon.com/rss/feed.rss",
        final_url="https://www.aboutamazon.com/rss/feed.rss",
        status_code=200,
        content_type="application/rss+xml",
        text="""
        <rss version="2.0">
          <channel>
            <item>
              <title>Fulfillment update</title>
              <link>https://www.aboutamazon.com/news/example</link>
              <description>{body}</description>
            </item>
          </channel>
        </rss>
        """.format(body="Amazon company update. " * 20),
    )

    source = classify_fetch_result(result, official_domain="aboutamazon.com")

    assert source is not None
    assert source.url == "https://www.aboutamazon.com/rss/feed.rss"
    assert source.source_type == "news_feed"
    assert source.confidence_score == 0.90
    assert "selected source URL is an RSS/Atom feed" in source.discovery_reason


def test_classify_fetch_result_skips_404s():
    result = FetchResult(
        url="https://nvidia.com/blog",
        final_url="https://nvidia.com/blog",
        status_code=404,
        content_type="text/html",
        text="not found",
    )

    assert classify_fetch_result(result, official_domain="nvidia.com") is None


def test_classify_fetch_result_rejects_protected_source_pages():
    result = FetchResult(
        url="https://example.com/blog",
        final_url="https://www.example.com/blog",
        status_code=403,
        content_type="text/html",
        text="<html><title>Access Denied</title></html>",
    )

    assert classify_fetch_result(result, official_domain="example.com") is None


def test_discover_sources_persists_candidates_for_known_company_domain(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        upsert_company(
            connection,
            CompanyRecord(
                ticker="NVDA",
                name="NVIDIA CORP",
                cik="0001045810",
                website_domain="nvidia.com",
            ),
        )
        fetcher = FakeFetcher(
            {
                "https://blogs.nvidia.com/": FetchResult(
                    url="https://blogs.nvidia.com/",
                    final_url="https://blogs.nvidia.com/",
                    status_code=200,
                    content_type="text/html",
                    text="""
                    <html>
                      <head><title>NVIDIA Blog</title></head>
                      <body>{body}</body>
                    </html>
                    """.format(body="Company blog update. " * 20),
                ),
                "https://investor.nvidia.com/": FetchResult(
                    url="https://investor.nvidia.com/",
                    final_url="https://investor.nvidia.com/",
                    status_code=200,
                    content_type="text/html",
                    text="""
                    <html>
                      <head><title>NVIDIA Investor Relations</title></head>
                      <body>{body}</body>
                    </html>
                    """.format(body="Investor relations update. " * 20),
                ),
            }
        )

        sources = discover_sources_for_ticker(
            connection=connection,
            ticker="NVDA",
            fetcher=fetcher,
        )
        rows = connection.execute(
            """
            SELECT source_type, discovery_status, confidence_score
            FROM sources
            ORDER BY source_type
            """
        ).fetchall()

    assert {source.source_type for source in sources} == {"company_blog", "investor_relations"}
    assert {row["source_type"] for row in rows} == {"company_blog", "investor_relations"}
    assert all(row["discovery_status"] == "candidate" for row in rows)
    assert all(row["confidence_score"] >= 0.8 for row in rows)


@pytest.mark.parametrize(
    ("existing_status", "existing_enabled"),
    [
        ("approved", True),
        ("rejected", False),
        ("manual", True),
    ],
)
def test_discover_sources_preserves_human_source_state(
    tmp_path,
    existing_status,
    existing_enabled,
):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        company_id = upsert_company(
            connection,
            CompanyRecord(
                ticker="NVDA",
                name="NVIDIA CORP",
                website_domain="nvidia.com",
            ),
        )
        upsert_source(
            connection,
            SourceRecord(
                company_id=company_id,
                name="Old NVIDIA Blog",
                url="https://blogs.nvidia.com/",
                source_type="company_blog",
                ownership="unknown",
                trust_level="medium",
                discovery_status=existing_status,
                enabled=existing_enabled,
                confidence_score=0.45,
                discovery_reason="old metadata",
            ),
        )
        fetcher = FakeFetcher(
            {
                "https://blogs.nvidia.com/": FetchResult(
                    url="https://blogs.nvidia.com/",
                    final_url="https://blogs.nvidia.com/",
                    status_code=200,
                    content_type="text/html",
                    text="""
                    <html>
                      <head><title>NVIDIA Blog</title></head>
                      <body>{body}</body>
                    </html>
                    """.format(body="Company blog update. " * 20),
                ),
            }
        )

        sources = discover_sources_for_ticker(
            connection=connection,
            ticker="NVDA",
            fetcher=fetcher,
        )
        source = connection.execute(
            """
            SELECT *
            FROM sources
            WHERE url = 'https://blogs.nvidia.com/'
            """
        ).fetchone()

    assert source["discovery_status"] == existing_status
    assert source["enabled"] == int(existing_enabled)
    assert sources[0].discovery_status == existing_status
    assert sources[0].enabled is existing_enabled
    assert source["name"] == "NVIDIA Blog"
    assert source["ownership"] == "official"
    assert source["confidence_score"] > 0.45
    assert source["discovery_reason"] != "old metadata"


def test_discover_sources_accepts_and_stores_website_domain(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        fetcher = FakeFetcher(
            {
                "https://example.com/newsroom": FetchResult(
                    url="https://example.com/newsroom",
                    final_url="https://example.com/newsroom",
                    status_code=200,
                    content_type="text/html",
                    text="""
                    <html>
                      <head><title>Example Newsroom</title></head>
                      <body>{body}</body>
                    </html>
                    """.format(body="Press release and company news. " * 20),
                )
            }
        )

        sources = discover_sources_for_ticker(
            connection=connection,
            ticker="EXM",
            website_domain="https://www.example.com/",
            fetcher=fetcher,
        )
        company = connection.execute("SELECT * FROM companies WHERE ticker = 'EXM'").fetchone()

    assert normalize_domain(company["website_domain"]) == "example.com"
    assert len(sources) == 1
    assert sources[0].source_type == "newsroom"


def test_discover_sources_dry_run_does_not_store_website_domain(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        fetcher = FakeFetcher(
            {
                "https://example.com/newsroom": FetchResult(
                    url="https://example.com/newsroom",
                    final_url="https://example.com/newsroom",
                    status_code=200,
                    content_type="text/html",
                    text="""
                    <html>
                      <head><title>Example Newsroom</title></head>
                      <body>{body}</body>
                    </html>
                    """.format(body="Press release and company news. " * 20),
                )
            }
        )

        sources = discover_sources_for_ticker(
            connection=connection,
            ticker="EXM",
            website_domain="https://www.example.com/",
            fetcher=fetcher,
            persist=False,
        )
        company = connection.execute("SELECT * FROM companies WHERE ticker = 'EXM'").fetchone()
        stored_sources = connection.execute("SELECT * FROM sources").fetchall()

    assert len(sources) == 1
    assert sources[0].persisted_id is None
    assert company is None
    assert stored_sources == []


def test_discover_sources_resolves_domain_from_sec_company_name(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        upsert_filing(
            connection,
            FilingMetadata(
                accession_number="0001045810-26-000001",
                ticker="NVDA",
                cik="0001045810",
                company_name="NVIDIA CORP",
                form_type="10-K",
                filing_date="2026-02-25",
                period_of_report="2026-01-25",
                raw_path="raw.txt",
                raw_sha256="a" * 64,
                clean_text_path="clean.txt",
            ),
        )
        fetcher = FakeFetcher(
            {
                "https://nvidia.com/": FetchResult(
                    url="https://nvidia.com/",
                    final_url="https://www.nvidia.com/en-us/",
                    status_code=200,
                    content_type="text/html",
                    text="""
                    <html>
                      <head><title>NVIDIA Corporation</title></head>
                      <body>{body}</body>
                    </html>
                    """.format(body="NVIDIA accelerated computing company. " * 20),
                ),
                "https://blogs.nvidia.com/": FetchResult(
                    url="https://blogs.nvidia.com/",
                    final_url="https://blogs.nvidia.com/",
                    status_code=200,
                    content_type="text/html",
                    text="""
                    <html>
                      <head><title>NVIDIA Blog</title></head>
                      <body>{body}</body>
                    </html>
                    """.format(body="Company blog update. " * 20),
                ),
            }
        )

        sources = discover_sources_for_ticker(
            connection=connection,
            ticker="NVDA",
            fetcher=fetcher,
        )
        company = connection.execute("SELECT * FROM companies WHERE ticker = 'NVDA'").fetchone()

    assert company["name"] == "NVIDIA CORP"
    assert company["website_domain"] == "nvidia.com"
    assert {source.source_type for source in sources} == {"company_blog"}


def test_discover_sources_explains_manual_domain_override_when_resolution_fails(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        upsert_filing(
            connection,
            FilingMetadata(
                accession_number="0001652044-26-000018",
                ticker="GOOGL",
                cik="0001652044",
                company_name="Alphabet Inc.",
                form_type="10-K",
                filing_date="2026-02-05",
                period_of_report="2025-12-31",
                raw_path="raw.txt",
                raw_sha256="a" * 64,
                clean_text_path="clean.txt",
            ),
        )

        with pytest.raises(ValueError) as error:
            discover_sources_for_ticker(
                connection=connection,
                ticker="GOOGL",
                fetcher=FakeFetcher({}),
            )

    message = str(error.value)
    assert "Tried default domain candidates: alphabet.com" in message
    assert "--website-domain example.com" in message
    assert "--ticker GOOGL" in message


def test_discover_sources_resolves_domain_from_sec_name_using_source_path(tmp_path):
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        upsert_filing(
            connection,
            FilingMetadata(
                accession_number="0001018724-26-000001",
                ticker="AMZN",
                cik="0001018724",
                company_name="AMAZON COM INC",
                form_type="10-K",
                filing_date="2026-02-01",
                period_of_report="2025-12-31",
                raw_path="raw.txt",
                raw_sha256="a" * 64,
                clean_text_path="clean.txt",
            ),
        )
        fetcher = FakeFetcher(
            {
                "https://amazon.com/blog": FetchResult(
                    url="https://amazon.com/blog",
                    final_url="https://www.aboutamazon.com/",
                    status_code=200,
                    content_type="text/html",
                    text="""
                    <html>
                      <head><title>Amazon Blog</title></head>
                      <body>{body}</body>
                    </html>
                    """.format(body="Amazon company news and updates. " * 20),
                ),
                "https://amazon.com/": FetchResult(
                    url="https://amazon.com/",
                    final_url="https://www.amazon.com/",
                    status_code=200,
                    content_type="text/html",
                    text="<html><title>Amazon.com</title><body></body></html>",
                ),
                "https://aboutamazon.com/news": FetchResult(
                    url="https://aboutamazon.com/news",
                    final_url="https://www.aboutamazon.com/news",
                    status_code=200,
                    content_type="text/html",
                    text="""
                    <html>
                      <head><title>Amazon News</title></head>
                      <body>{body}</body>
                    </html>
                    """.format(body="Amazon company news and updates. " * 20),
                ),
            }
        )

        sources = discover_sources_for_ticker(
            connection=connection,
            ticker="AMZN",
            fetcher=fetcher,
        )
        company = connection.execute("SELECT * FROM companies WHERE ticker = 'AMZN'").fetchone()

    assert company["website_domain"] == "amazon.com"
    assert {source.source_type for source in sources} == {"company_blog"}
