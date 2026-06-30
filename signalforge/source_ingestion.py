import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import feedparser
import httpx
import trafilatura
from bs4 import BeautifulSoup

from signalforge.sections import chunk_text
from signalforge.storage import (
    DocumentRecord,
    GenericDocumentChunk,
    complete_source_ingestion_run,
    create_source_ingestion_run,
    initialize_database,
    replace_document_chunks,
    upsert_document,
)


DEFAULT_USER_AGENT = "SignalForge/0.1 source-ingestion (+https://github.com/local/signalforge)"
DEFAULT_TIMEOUT = 20.0


@dataclass(frozen=True)
class FeedEntry:
    url: str
    title: str | None = None
    author: str | None = None
    published_at: str | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class SourceIngestionResult:
    source_id: int
    source_name: str
    status: str
    discovered_count: int
    inserted_count: int
    skipped_count: int
    error_message: str | None = None


def ingest_approved_sources(
    connection,
    *,
    ticker: str | None = None,
    processed_dir: str | Path = "data/processed",
    chunk_size: int = 4_000,
    overlap: int = 500,
    limit_per_source: int | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[SourceIngestionResult]:
    initialize_database(connection)
    sources = load_approved_enabled_sources(connection, ticker=ticker)

    headers = {"User-Agent": user_agent}
    with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
        return [
            ingest_source(
                connection,
                source,
                client=client,
                processed_dir=processed_dir,
                chunk_size=chunk_size,
                overlap=overlap,
                limit=limit_per_source,
            )
            for source in sources
        ]


def load_approved_enabled_sources(connection, *, ticker: str | None = None) -> list:
    query = """
        SELECT
            s.*,
            c.ticker,
            c.name AS company_name
        FROM sources AS s
        LEFT JOIN companies AS c ON c.id = s.company_id
        WHERE s.discovery_status IN ('approved', 'manual')
          AND s.enabled = 1
    """
    parameters: list[object] = []
    if ticker:
        query += " AND c.ticker = ?"
        parameters.append(ticker.upper())

    query += " ORDER BY c.ticker, s.name"
    return connection.execute(query, parameters).fetchall()


def ingest_source(
    connection,
    source,
    *,
    client: httpx.Client,
    processed_dir: str | Path = "data/processed",
    chunk_size: int = 4_000,
    overlap: int = 500,
    limit: int | None = None,
) -> SourceIngestionResult:
    run_id = create_source_ingestion_run(connection, int(source["id"]))
    discovered_count = 0
    inserted_count = 0
    skipped_count = 0
    errors = []

    try:
        entries = discover_feed_entries(client, source["url"])
        if limit is not None:
            entries = entries[:limit]
        discovered_count = len(entries)

        for entry in entries:
            try:
                if document_url_exists(connection, int(source["id"]), entry.url):
                    skipped_count += 1
                    continue

                article = fetch_article(client, entry.url)
                if article.text is None or len(article.text.strip()) < 100:
                    skipped_count += 1
                    errors.append(f"no extractable text: {entry.url}")
                    continue

                content_hash = hash_text(article.text)
                if content_hash_exists(connection, content_hash):
                    skipped_count += 1
                    continue

                clean_text_path = write_clean_article_text(
                    processed_dir=processed_dir,
                    source_id=int(source["id"]),
                    content_hash=content_hash,
                    text=article.text,
                )
                document_id = upsert_document(
                    connection,
                    DocumentRecord(
                        source_id=int(source["id"]),
                        url=entry.url,
                        title=entry.title or article.title,
                        author=entry.author,
                        published_at=entry.published_at,
                        clean_text_path=str(clean_text_path),
                        content_hash=content_hash,
                        document_type=document_type_for_source(source["source_type"]),
                        metadata={
                            "ticker": source["ticker"],
                            "company_name": source["company_name"],
                            "source_url": source["url"],
                            **(entry.metadata or {}),
                        },
                    ),
                )
                replace_document_chunks(
                    connection,
                    document_id,
                    [
                        GenericDocumentChunk(chunk_index=index, text=text)
                        for index, text in enumerate(
                            chunk_text(article.text, chunk_size=chunk_size, overlap=overlap)
                        )
                    ],
                )
                inserted_count += 1
            except Exception as error:
                skipped_count += 1
                errors.append(f"{entry.url}: {type(error).__name__}: {error}")

    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        complete_source_ingestion_run(
            connection,
            run_id=run_id,
            status="failed",
            discovered_count=discovered_count,
            inserted_count=inserted_count,
            skipped_count=skipped_count,
            error_message=message,
        )
        return SourceIngestionResult(
            source_id=int(source["id"]),
            source_name=source["name"],
            status="failed",
            discovered_count=discovered_count,
            inserted_count=inserted_count,
            skipped_count=skipped_count,
            error_message=message,
        )

    status = "partial" if errors else "completed"
    error_message = "; ".join(errors[:3]) if errors else None
    complete_source_ingestion_run(
        connection,
        run_id=run_id,
        status=status,
        discovered_count=discovered_count,
        inserted_count=inserted_count,
        skipped_count=skipped_count,
        error_message=error_message,
    )
    return SourceIngestionResult(
        source_id=int(source["id"]),
        source_name=source["name"],
        status=status,
        discovered_count=discovered_count,
        inserted_count=inserted_count,
        skipped_count=skipped_count,
        error_message=error_message,
    )


@dataclass(frozen=True)
class ArticleText:
    url: str
    text: str | None
    title: str | None = None


def discover_feed_entries(client: httpx.Client, source_url: str) -> list[FeedEntry]:
    response = fetch_url(client, source_url)
    entries = parse_feed_entries(response.text, base_url=str(response.url))
    if entries:
        return entries

    feed_urls = discover_feed_links(response.text, base_url=str(response.url))
    for feed_url in feed_urls:
        feed_response = fetch_url(client, feed_url)
        entries.extend(parse_feed_entries(feed_response.text, base_url=str(feed_response.url)))

    return dedupe_entries(entries)


def parse_feed_entries(content: str | bytes, *, base_url: str) -> list[FeedEntry]:
    parsed = feedparser.parse(content)
    entries = []

    for entry in parsed.entries:
        url = normalize_url(entry.get("link") or "")
        if not url:
            continue

        published_at = normalize_entry_datetime(
            entry.get("published_parsed") or entry.get("updated_parsed")
        )
        entries.append(
            FeedEntry(
                url=urljoin(base_url, url),
                title=strip_text(entry.get("title")),
                author=strip_text(entry.get("author")),
                published_at=published_at,
                metadata={
                    "feed_id": entry.get("id"),
                    "feed_published": entry.get("published"),
                    "feed_updated": entry.get("updated"),
                },
            )
        )

    return dedupe_entries(entries)


def discover_feed_links(html: str, *, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = []
    for link in soup.find_all("link"):
        rel = " ".join(link.get("rel", [])).lower()
        type_value = (link.get("type") or "").lower()
        href = link.get("href")
        if not href:
            continue
        if "alternate" in rel and (
            "rss" in type_value or "atom" in type_value or "xml" in type_value
        ):
            urls.append(normalize_url(urljoin(base_url, href)))

    return [url for url in dict.fromkeys(urls) if url]


def fetch_article(client: httpx.Client, url: str) -> ArticleText:
    response = fetch_url(client, url)
    html = response.text
    extracted = trafilatura.extract(
        html,
        url=str(response.url),
        include_comments=False,
        include_tables=False,
    )
    soup = BeautifulSoup(html, "lxml")
    title = strip_text(soup.title.get_text(" ")) if soup.title else None
    text = normalize_article_text(extracted) or extract_text_with_bs4(soup)
    return ArticleText(url=str(response.url), text=text, title=title)


def fetch_url(client: httpx.Client, url: str) -> httpx.Response:
    response = client.get(url)
    response.raise_for_status()
    return response


def extract_text_with_bs4(soup: BeautifulSoup) -> str | None:
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    article = soup.find("article") or soup.body
    if article is None:
        return None
    return normalize_article_text(article.get_text("\n"))


def normalize_article_text(text: str | None) -> str | None:
    if not text:
        return None
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    normalized = "\n\n".join(line for line in lines if line)
    return normalized or None


def document_url_exists(connection, source_id: int, url: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM documents WHERE source_id = ? AND url = ? LIMIT 1",
        (source_id, url),
    ).fetchone()
    return row is not None


def content_hash_exists(connection, content_hash: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM documents WHERE content_hash = ? LIMIT 1",
        (content_hash,),
    ).fetchone()
    return row is not None


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_clean_article_text(
    *,
    processed_dir: str | Path,
    source_id: int,
    content_hash: str,
    text: str,
) -> Path:
    output_path = (
        Path(processed_dir)
        / "web"
        / f"source-{source_id}"
        / f"{content_hash[:16]}.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def document_type_for_source(source_type: str) -> str:
    if source_type == "company_blog":
        return "blog_post"
    if source_type == "newsroom":
        return "press_release"
    if source_type == "investor_relations":
        return "investor_update"
    if source_type == "webpage":
        return "webpage"
    return "article"


def normalize_entry_datetime(value) -> str | None:
    if value is None:
        return None
    return datetime(*value[:6], tzinfo=UTC).isoformat()


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    parts = urlsplit(url)
    if not parts.scheme and not parts.netloc:
        return url
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def strip_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = re.sub(r"\s+", " ", value).strip()
    return stripped or None


def dedupe_entries(entries: list[FeedEntry]) -> list[FeedEntry]:
    deduped = []
    seen = set()
    for entry in entries:
        if entry.url in seen:
            continue
        seen.add(entry.url)
        deduped.append(entry)
    return deduped
