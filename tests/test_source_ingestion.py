import httpx

from signalforge.sections import chunk_text
from signalforge.source_ingestion import (
    DEFAULT_USER_AGENT,
    discover_feed_links,
    fetch_article,
    ingest_approved_sources,
    parse_feed_entries,
)
from signalforge.storage import (
    CompanyRecord,
    SourceRecord,
    connect_database,
    initialize_database,
    upsert_company,
    upsert_source,
)


FEED_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Example Blog</title>
    <item>
      <title>AI Infrastructure Update</title>
      <link>https://example.com/articles/ai-infrastructure</link>
      <author>Example Team</author>
      <pubDate>Mon, 01 Jun 2026 12:00:00 GMT</pubDate>
      <guid>article-1</guid>
    </item>
    <item>
      <title>AI Infrastructure Update Duplicate</title>
      <link>https://example.com/articles/ai-infrastructure</link>
      <guid>article-1-copy</guid>
    </item>
  </channel>
</rss>
"""


ARTICLE_HTML = """\
<html>
  <head><title>AI Infrastructure Update</title></head>
  <body>
    <article>
      <h1>AI Infrastructure Update</h1>
      <p>Example is expanding AI infrastructure capacity for enterprise customers.</p>
      <p>The update discusses accelerators, networking, software, and supply planning.</p>
      <p>These details are intentionally long enough to pass extraction checks in tests.</p>
    </article>
  </body>
</html>
"""


def test_chunk_text_handles_paragraphs_overlap_and_validation():
    chunks = chunk_text("Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph.", chunk_size=35, overlap=5)

    assert len(chunks) >= 2
    assert chunks[0].startswith("Alpha paragraph.")
    assert all(len(chunk) <= 35 for chunk in chunks)


def test_parse_feed_entries_deduplicates_urls_and_normalizes_metadata():
    entries = parse_feed_entries(FEED_XML, base_url="https://example.com/feed.xml")

    assert len(entries) == 1
    assert entries[0].url == "https://example.com/articles/ai-infrastructure"
    assert entries[0].title == "AI Infrastructure Update"
    assert entries[0].author == "Example Team"
    assert entries[0].published_at == "2026-06-01T12:00:00+00:00"


def test_discover_feed_links_finds_rss_and_atom_alternates():
    html = """
    <html><head>
      <link rel="alternate" type="application/rss+xml" href="/feed.xml">
      <link rel="alternate" type="application/atom+xml" href="https://example.com/atom.xml">
    </head></html>
    """

    assert discover_feed_links(html, base_url="https://example.com/blog/") == [
        "https://example.com/feed.xml",
        "https://example.com/atom.xml",
    ]


def test_fetch_article_falls_back_to_bs4_extraction(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/article"
        return httpx.Response(200, text=ARTICLE_HTML)

    monkeypatch.setattr("signalforge.source_ingestion.trafilatura.extract", lambda *args, **kwargs: None)

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        article = fetch_article(client, "https://example.com/article")

    assert article.title == "AI Infrastructure Update"
    assert article.text is not None
    assert "Example is expanding AI infrastructure capacity" in article.text
    assert "script" not in article.text.lower()


def test_ingest_approved_sources_fetches_feed_articles_and_chunks_documents(tmp_path):
    db_path = tmp_path / "signalforge.sqlite3"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"] == DEFAULT_USER_AGENT
        if str(request.url) == "https://example.com/feed.xml":
            return httpx.Response(200, text=FEED_XML)
        if str(request.url) == "https://example.com/articles/ai-infrastructure":
            return httpx.Response(200, text=ARTICLE_HTML)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    with connect_database(db_path) as connection:
        initialize_database(connection)
        company_id = upsert_company(connection, CompanyRecord(ticker="EXM", name="Example"))
        source_id = upsert_source(
            connection,
            SourceRecord(
                company_id=company_id,
                name="Example Blog",
                url="https://example.com/feed.xml",
                source_type="company_blog",
                ownership="official",
                trust_level="high",
                discovery_status="approved",
            ),
        )

        with httpx.Client(
            transport=transport,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            follow_redirects=True,
        ) as client:
            from signalforge.source_ingestion import ingest_source, load_approved_enabled_sources

            source = load_approved_enabled_sources(connection, ticker="EXM")[0]
            result = ingest_source(
                connection,
                source,
                client=client,
                processed_dir=tmp_path / "processed",
                chunk_size=120,
                overlap=20,
            )

        documents = connection.execute("SELECT * FROM documents WHERE source_id = ?", (source_id,)).fetchall()
        chunks = connection.execute("SELECT * FROM document_chunks ORDER BY chunk_index").fetchall()
        runs = connection.execute("SELECT * FROM source_ingestion_runs").fetchall()

    assert result.status == "completed"
    assert result.discovered_count == 1
    assert result.inserted_count == 1
    assert documents[0]["document_type"] == "blog_post"
    assert documents[0]["published_at"] == "2026-06-01T12:00:00+00:00"
    assert len(chunks) >= 1
    assert runs[0]["status"] == "completed"


def test_ingest_approved_sources_skips_duplicate_urls_and_content_hashes(tmp_path):
    feed_xml = """\
    <rss version="2.0"><channel>
      <item><title>One</title><link>https://example.com/a</link></item>
      <item><title>Two</title><link>https://example.com/b</link></item>
    </channel></rss>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://example.com/feed.xml":
            return httpx.Response(200, text=feed_xml)
        return httpx.Response(200, text=ARTICLE_HTML)

    transport = httpx.MockTransport(handler)
    with connect_database(tmp_path / "signalforge.sqlite3") as connection:
        initialize_database(connection)
        source_id = upsert_source(
            connection,
            SourceRecord(
                name="Example Feed",
                url="https://example.com/feed.xml",
                source_type="news_feed",
                discovery_status="approved",
            ),
        )

        with httpx.Client(transport=transport, follow_redirects=True) as client:
            from signalforge.source_ingestion import ingest_source, load_approved_enabled_sources

            source = load_approved_enabled_sources(connection)[0]
            first = ingest_source(connection, source, client=client, processed_dir=tmp_path / "processed")
            second = ingest_source(connection, source, client=client, processed_dir=tmp_path / "processed")

        documents = connection.execute("SELECT * FROM documents WHERE source_id = ?", (source_id,)).fetchall()

    assert first.inserted_count == 1
    assert first.skipped_count == 1
    assert second.inserted_count == 0
    assert second.skipped_count == 2
    assert len(documents) == 1


def test_ingest_approved_sources_records_failed_source_and_continues(tmp_path, monkeypatch):
    db_path = tmp_path / "signalforge.sqlite3"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://example.com/good-feed.xml":
            return httpx.Response(200, text=FEED_XML)
        if str(request.url) == "https://example.com/articles/ai-infrastructure":
            return httpx.Response(200, text=ARTICLE_HTML)
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr("signalforge.source_ingestion.httpx.Client", client_factory)

    with connect_database(db_path) as connection:
        initialize_database(connection)
        upsert_source(
            connection,
            SourceRecord(
                name="Broken Feed",
                url="https://example.com/broken-feed.xml",
                source_type="news_feed",
                discovery_status="approved",
            ),
        )
        upsert_source(
            connection,
            SourceRecord(
                name="Good Feed",
                url="https://example.com/good-feed.xml",
                source_type="news_feed",
                discovery_status="approved",
            ),
        )

        results = ingest_approved_sources(connection, processed_dir=tmp_path / "processed")
        runs = connection.execute(
            "SELECT status FROM source_ingestion_runs ORDER BY id"
        ).fetchall()

    assert [result.status for result in results] == ["failed", "completed"]
    assert [row["status"] for row in runs] == ["failed", "completed"]
