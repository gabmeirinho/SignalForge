from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from signalforge.storage import (
    CompanyRecord,
    SourceRecord,
    load_company_by_ticker,
    load_latest_filing_company_metadata,
    upsert_company,
    upsert_source,
)

DEFAULT_TIMEOUT_SECONDS = 8.0
DEFAULT_USER_AGENT = "SignalForge source discovery/0.1"
MIN_CANDIDATE_SCORE = 0.45

LEGAL_NAME_SUFFIXES = {
    "class",
    "com",
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holding",
    "holdings",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "ltd",
    "plc",
    "sa",
}

SOURCE_PATHS = (
    "/blog",
    "/blogs",
    "/news",
    "/newsroom",
    "/press",
    "/press-releases",
    "/media",
    "/media-center",
    "/company/news",
    "/company/newsroom",
    "/about/news",
    "/about/newsroom",
    "/investors",
    "/investor",
    "/investor-relations",
    "/ir",
    "/financials",
    "/news-releases",
    "/rss",
    "/feed",
    "/atom.xml",
    "/rss.xml",
    "/feed.xml",
    "/blog/feed",
    "/news/feed",
    "/newsroom/feed",
    "/press-releases/feed",
)

SOURCE_SUBDOMAINS = (
    "blog",
    "blogs",
    "news",
    "newsroom",
    "media",
    "press",
    "investor",
    "investors",
    "ir",
)

RSS_CONTENT_TYPES = (
    "application/rss+xml",
    "application/atom+xml",
    "application/xml",
    "text/xml",
)


@dataclass(frozen=True)
class CompanyResolution:
    company_id: int | None
    ticker: str
    name: str | None
    cik: str | None
    website_domain: str


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str | None
    status_code: int | None
    content_type: str
    text: str
    error: str | None = None


@dataclass(frozen=True)
class DiscoveredSource:
    name: str
    url: str
    final_url: str | None
    source_type: str
    ownership: str
    trust_level: str
    confidence_score: float
    discovery_reason: str
    status_code: int | None
    rss_urls: tuple[str, ...]
    persisted_id: int | None = None


class HttpxFetcher:
    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    def fetch(self, url: str) -> FetchResult:
        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=self.timeout_seconds,
                headers={"User-Agent": self.user_agent},
            ) as client:
                response = client.get(url)
        except httpx.HTTPError as error:
            return FetchResult(
                url=url,
                final_url=None,
                status_code=None,
                content_type="",
                text="",
                error=str(error),
            )

        return FetchResult(
            url=url,
            final_url=str(response.url),
            status_code=response.status_code,
            content_type=response.headers.get("content-type", ""),
            text=response.text,
        )


def discover_sources_for_ticker(
    *,
    connection,
    ticker: str,
    website_domain: str | None = None,
    fetcher: HttpxFetcher | None = None,
    persist: bool = True,
) -> list[DiscoveredSource]:
    fetcher = fetcher or HttpxFetcher()
    resolution = resolve_company(
        connection,
        ticker=ticker,
        website_domain=website_domain,
        fetcher=fetcher,
        persist=persist,
    )
    candidates: list[DiscoveredSource] = []

    for url in generate_candidate_urls(resolution.website_domain):
        result = fetcher.fetch(url)
        source = classify_fetch_result(result, official_domain=resolution.website_domain)
        if source is None:
            continue

        persisted_id = None
        if persist:
            persisted_id = upsert_source(
                connection,
                SourceRecord(
                    company_id=resolution.company_id,
                    name=source.name,
                    url=source.url,
                    source_type=source.source_type,
                    ownership=source.ownership,
                    trust_level=source.trust_level,
                    discovery_status="candidate",
                    enabled=True,
                    confidence_score=source.confidence_score,
                    discovery_reason=source.discovery_reason,
                ),
            )
        candidates.append(
            DiscoveredSource(
                name=source.name,
                url=source.url,
                final_url=source.final_url,
                source_type=source.source_type,
                ownership=source.ownership,
                trust_level=source.trust_level,
                confidence_score=source.confidence_score,
                discovery_reason=source.discovery_reason,
                status_code=source.status_code,
                rss_urls=source.rss_urls,
                persisted_id=persisted_id,
            )
        )

    return sorted(
        _deduplicate_sources(candidates),
        key=lambda source: (-source.confidence_score, source.url),
    )


def resolve_company(
    connection,
    *,
    ticker: str,
    website_domain: str | None = None,
    fetcher: HttpxFetcher | None = None,
    persist: bool = True,
) -> CompanyResolution:
    ticker = ticker.upper()
    company = load_company_by_ticker(connection, ticker)
    filing_metadata = load_latest_filing_company_metadata(connection, ticker)

    name = _first_non_empty(
        company["name"] if company else None,
        filing_metadata["company_name"] if filing_metadata else None,
    )
    cik = _first_non_empty(
        company["cik"] if company else None,
        filing_metadata["cik"] if filing_metadata else None,
    )
    domain = normalize_domain(
        _first_non_empty(website_domain, company["website_domain"] if company else None)
    )
    if not domain:
        if name is None:
            raise ValueError(
                f"No company name is known for {ticker}. "
                "Ingest an SEC filing first or pass --website-domain."
            )
        domain_candidates = generate_domain_candidates_from_company_name(name)
        domain = resolve_domain_from_company_name(
            company_name=name,
            fetcher=fetcher or HttpxFetcher(),
        )
        if not domain:
            tried = ", ".join(domain_candidates) if domain_candidates else "none"
            raise ValueError(
                f"Default domain resolution failed for {ticker} from SEC name {name!r}. "
                f"Tried default domain candidates: {tried}. "
                "Some companies use a domain that cannot be inferred from their legal name. "
                f"Add the official domain manually with --website-domain, for example: "
                f"uv run python -m signalforge.cli.discover_sources --ticker {ticker} "
                "--website-domain example.com. Replace example.com with the company's official domain."
            )

    company_id = None
    if persist:
        company_id = upsert_company(
            connection,
            CompanyRecord(
                ticker=ticker,
                name=name,
                cik=cik,
                website_domain=domain,
            ),
        )
    return CompanyResolution(
        company_id=company_id,
        ticker=ticker,
        name=name,
        cik=cik,
        website_domain=domain,
    )


def resolve_domain_from_company_name(
    *,
    company_name: str,
    fetcher: HttpxFetcher,
) -> str | None:
    for domain in generate_domain_candidates_from_company_name(company_name):
        result = fetcher.fetch(f"https://{domain}/")
        if domain_root_is_reachable(result):
            return domain
    return None


def generate_domain_candidates_from_company_name(company_name: str) -> list[str]:
    tokens = normalize_company_name_tokens(company_name)
    if not tokens:
        return []

    candidates = [f"{tokens[0]}.com"]
    if len(tokens) > 1:
        candidates.append(f"{''.join(tokens)}.com")
        candidates.append(f"{'-'.join(tokens)}.com")

    return list(dict.fromkeys(candidates))


def normalize_company_name_tokens(company_name: str) -> list[str]:
    normalized = re.sub(r"[^a-zA-Z0-9& ]+", " ", company_name).lower()
    normalized = normalized.replace("&", " and ")
    tokens = [token for token in normalized.split() if token not in LEGAL_NAME_SUFFIXES]
    while tokens and tokens[-1] in {"a", "b", "c"}:
        tokens.pop()
    return tokens


def domain_root_is_reachable(result: FetchResult) -> bool:
    return result.status_code is not None and (
        200 <= result.status_code < 400 or result.status_code in {401, 403}
    )


def generate_candidate_urls(website_domain: str) -> list[str]:
    domain = normalize_domain(website_domain)
    urls: list[str] = []
    base_hosts = [domain]
    if not domain.startswith("www."):
        base_hosts.append(f"www.{domain}")

    for host in base_hosts:
        urls.append(f"https://{host}/")
        for path in SOURCE_PATHS:
            urls.append(f"https://{host}{path}")

    for subdomain in SOURCE_SUBDOMAINS:
        urls.append(f"https://{subdomain}.{domain}/")

    return list(dict.fromkeys(urls))


def classify_fetch_result(
    result: FetchResult,
    *,
    official_domain: str,
) -> DiscoveredSource | None:
    if result.status_code in {404, 410}:
        return None

    final_url = result.final_url or result.url
    title, text, rss_urls = inspect_content(result, final_url)
    source_type = infer_source_type(result.url, final_url, title)
    official_match = is_official_domain(result.url, official_domain) or is_official_domain(
        final_url,
        official_domain,
    )
    reachable = result.status_code is not None and 200 <= result.status_code < 300
    source_signal = has_source_signal(result.url, title, rss_urls) or has_source_signal(
        final_url,
        title,
        rss_urls,
    )
    source_url = preferred_source_url(final_url, rss_urls, source_type)
    score, reasons = score_candidate(
        url=result.url if is_official_domain(result.url, official_domain) else final_url,
        selected_url=source_url,
        official_domain=official_domain,
        title=title,
        rss_urls=rss_urls,
        reachable=reachable,
        has_extractable_content=has_extractable_content(result, text),
    )

    if not reachable or source_type is None or not source_signal or score < MIN_CANDIDATE_SCORE:
        return None

    ownership = "official" if official_match else "unknown"
    return DiscoveredSource(
        name=build_source_name(final_url, title, source_type),
        url=source_url,
        final_url=final_url,
        source_type="news_feed" if is_feed_url(source_url) else source_type,
        ownership=ownership,
        trust_level=trust_level_from_score(score),
        confidence_score=round(score, 2),
        discovery_reason=", ".join(reasons),
        status_code=result.status_code,
        rss_urls=tuple(rss_urls),
    )


def inspect_content(result: FetchResult, final_url: str) -> tuple[str | None, str, list[str]]:
    if is_feed_response(result):
        soup = BeautifulSoup(result.text, "xml")
        title = _clean_text(soup.find("title").get_text(" ", strip=True) if soup.find("title") else "")
        entries = soup.find_all(["item", "entry"])
        return title or None, " ".join(entry.get_text(" ", strip=True) for entry in entries[:5]), [final_url]

    soup = BeautifulSoup(result.text, "lxml")
    title = _clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    text = _clean_text(soup.get_text(" ", strip=True))
    rss_urls = []
    for link in soup.find_all("link"):
        rel = " ".join(link.get("rel", [])) if isinstance(link.get("rel"), list) else str(link.get("rel") or "")
        content_type = str(link.get("type") or "").lower()
        href = link.get("href")
        if not href:
            continue
        if "alternate" in rel.lower() and any(feed_type in content_type for feed_type in RSS_CONTENT_TYPES):
            rss_urls.append(urljoin(final_url, href))
    return title or None, text, list(dict.fromkeys(rss_urls))


def score_candidate(
    *,
    url: str,
    selected_url: str,
    official_domain: str,
    title: str | None,
    rss_urls: list[str],
    reachable: bool,
    has_extractable_content: bool,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if is_official_domain(url, official_domain):
        score += 0.35
        reasons.append("official root domain or known official subdomain")
    else:
        score -= 0.30
        reasons.append("domain mismatch")

    if has_strong_path_or_subdomain_signal(url):
        score += 0.20
        reasons.append("path or subdomain strongly matches source type")

    if title_has_source_signal(title):
        score += 0.15
        reasons.append("page title contains source signal")

    if rss_urls:
        score += 0.15
        reasons.append("RSS/Atom link discovered")

    if is_feed_url(selected_url):
        score += 0.05
        reasons.append("selected source URL is an RSS/Atom feed")

    if reachable:
        score += 0.10
        reasons.append("reachable 2xx after redirects")
    else:
        score -= 0.20
        reasons.append("unreachable or timeout")

    if has_extractable_content:
        score += 0.05
        reasons.append("extractable text or feed entries")

    return max(0.0, min(score, 1.0)), reasons


def infer_source_type(url: str, final_url: str | None = None, title: str | None = None) -> str | None:
    parsed = urlparse(url)
    final_parsed = urlparse(final_url or "")
    host = parsed.netloc.lower()
    path = parsed.path.lower().rstrip("/")
    final_host = final_parsed.netloc.lower()
    final_path = final_parsed.path.lower().rstrip("/")
    haystack = f"{host} {path} {final_host} {final_path} {(title or '').lower()}"

    if is_feed_url(url):
        return "news_feed"
    if any(token in haystack for token in ("investor", "investors", "investor-relations", "/ir", " ir ")):
        return "investor_relations"
    if any(token in haystack for token in ("blog", "blogs")):
        return "company_blog"
    if any(token in haystack for token in ("newsroom", "news", "press", "media", "news-releases")):
        return "newsroom"
    return None


def has_source_signal(url: str, title: str | None, rss_urls: list[str]) -> bool:
    return bool(
        has_strong_path_or_subdomain_signal(url)
        or title_has_source_signal(title)
        or rss_urls
        or is_feed_url(url)
    )


def has_strong_path_or_subdomain_signal(url: str) -> bool:
    parsed = urlparse(url)
    labels = parsed.netloc.lower().split(".")
    path = parsed.path.lower()
    subdomain = labels[0] if len(labels) > 2 else ""
    return subdomain in SOURCE_SUBDOMAINS or any(
        token in path
        for token in (
            "blog",
            "news",
            "newsroom",
            "press",
            "media",
            "investor",
            "investors",
            "investor-relations",
            "/ir",
            "feed",
            "rss",
            "atom",
        )
    )


def title_has_source_signal(title: str | None) -> bool:
    if not title:
        return False
    lowered = title.lower()
    return any(
        token in lowered
        for token in ("blog", "newsroom", "press", "investor relations", "news releases", "media")
    )


def is_official_domain(url: str, official_domain: str) -> bool:
    host = urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
    domain = normalize_domain(official_domain)
    return host == domain or host.endswith(f".{domain}") or host == f"www.{domain}"


def normalize_domain(value: str | None) -> str:
    if not value:
        return ""
    parsed = urlparse(value if "://" in value else f"https://{value}")
    domain = (parsed.netloc or parsed.path).lower().strip().strip("/")
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def build_source_name(url: str, title: str | None, source_type: str) -> str:
    if title:
        return title[:120]
    host = urlparse(url).netloc
    label = source_type.replace("_", " ").title()
    return f"{host} {label}"


def preferred_source_url(final_url: str, rss_urls: list[str], source_type: str) -> str:
    if source_type in {"company_blog", "newsroom", "news_feed"} and rss_urls:
        return rss_urls[0]
    return final_url


def trust_level_from_score(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def is_feed_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith((".xml", ".rss", "/feed", "/rss", "/atom")) or "/feed/" in path


def is_feed_response(result: FetchResult) -> bool:
    content_type = result.content_type.lower()
    stripped = result.text.lstrip().lower()
    return any(feed_type in content_type for feed_type in RSS_CONTENT_TYPES) or stripped.startswith(
        ("<?xml", "<rss", "<feed")
    )


def has_extractable_content(result: FetchResult, text: str) -> bool:
    if is_feed_response(result):
        return bool(text)
    return len(text) >= 200


def _deduplicate_sources(sources: list[DiscoveredSource]) -> list[DiscoveredSource]:
    by_url: dict[str, DiscoveredSource] = {}
    for source in sources:
        current = by_url.get(source.url)
        if current is None or source.confidence_score > current.confidence_score:
            by_url[source.url] = source
    return list(by_url.values())


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _clean_text(value: str) -> str:
    return " ".join(unescape(value).split())
