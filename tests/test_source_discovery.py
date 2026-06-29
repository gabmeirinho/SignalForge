from signalforge.source_discovery import (
    FetchResult,
    classify_fetch_result,
    discover_sources_for_ticker,
    generate_domain_candidates_from_company_name,
    generate_candidate_urls,
    homepage_confirms_company_domain,
    normalize_domain,
)
from signalforge.storage import (
    CompanyRecord,
    FilingMetadata,
    connect_database,
    initialize_database,
    upsert_filing,
    upsert_company,
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
    assert generate_domain_candidates_from_company_name("META PLATFORMS, INC.") == [
        "metaplatforms.com",
        "meta.com",
        "meta-platforms.com",
    ]
    assert generate_domain_candidates_from_company_name("BERKSHIRE HATHAWAY INC") == [
        "berkshirehathaway.com",
        "berkshire.com",
        "berkshire-hathaway.com",
    ]


def test_homepage_confirmation_rejects_parked_domains():
    result = FetchResult(
        url="https://example.com/",
        final_url="https://example.com/",
        status_code=200,
        content_type="text/html",
        text="<html><title>Example</title><body>This domain may be for sale.</body></html>",
    )

    assert (
        homepage_confirms_company_domain(
            result,
            company_name="EXAMPLE CORP",
            candidate_domain="example.com",
        )
        is False
    )


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


def test_classify_fetch_result_skips_404s():
    result = FetchResult(
        url="https://nvidia.com/blog",
        final_url="https://nvidia.com/blog",
        status_code=404,
        content_type="text/html",
        text="not found",
    )

    assert classify_fetch_result(result, official_domain="nvidia.com") is None


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
