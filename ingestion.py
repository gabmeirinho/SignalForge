import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from sec_edgar_downloader import Downloader

from parser import clean_sec_html, extract_primary_document
from sections import chunk_sections, extract_10k_sections
from storage import FilingMetadata, initialize_database, replace_filing_chunks, upsert_filing


HEADER_RE = re.compile(r"<SEC-HEADER>(?P<header>.*?)</SEC-HEADER>", re.IGNORECASE | re.DOTALL)
HEADER_FIELD_RE = re.compile(r"^[ \t]*(?P<name>[A-Z][A-Z \t]+):[ \t]*(?P<value>.+?)\s*$")


@dataclass(frozen=True)
class IngestionResult:
    accession_number: str
    ticker: str
    raw_path: str
    clean_text_path: str
    section_count: int
    chunk_count: int


def ingest_sec_10k_filings(
    *,
    ticker: str,
    connection,
    company_name: str | None = None,
    email_address: str | None = None,
    limit: int = 1,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    download: bool = True,
    chunk_size: int = 4_000,
    overlap: int = 500,
) -> list[IngestionResult]:
    ticker = ticker.upper()
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    if download:
        if not company_name or not email_address:
            raise ValueError("company_name and email_address are required when download=True")
        downloader = Downloader(company_name, email_address, download_folder=raw_dir)
        downloader.get("10-K", ticker, limit=limit)

    filing_paths = find_downloaded_filings(raw_dir, ticker, "10-K")[:limit]
    if not filing_paths:
        raise FileNotFoundError(f"No downloaded 10-K filings found for {ticker} under {raw_dir}")

    initialize_database(connection)

    results = []
    for filing_path in filing_paths:
        results.append(
            ingest_downloaded_filing(
                filing_path=filing_path,
                ticker=ticker,
                connection=connection,
                processed_dir=processed_dir,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return results


def ingest_downloaded_filing(
    *,
    filing_path: str | Path,
    ticker: str,
    connection,
    processed_dir: str | Path = "data/processed",
    chunk_size: int = 4_000,
    overlap: int = 500,
) -> IngestionResult:
    filing_path = Path(filing_path)
    ticker = ticker.upper()

    raw_bytes = filing_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw_submission = raw_bytes.decode("utf-8")
    metadata_values = parse_submission_metadata(raw_submission)
    accession_number = metadata_values.get("accession_number") or filing_path.parent.name
    form_type = metadata_values.get("form_type") or "10-K"

    primary_document = extract_primary_document(raw_submission, form_type=form_type)
    clean_text = clean_sec_html(primary_document.text)
    clean_text_path = write_clean_text(
        processed_dir=processed_dir,
        ticker=ticker,
        form_type=form_type,
        accession_number=accession_number,
        text=clean_text,
    )

    sections = extract_10k_sections(primary_document.text)
    chunks = chunk_sections(sections, chunk_size=chunk_size, overlap=overlap)

    metadata = FilingMetadata(
        accession_number=accession_number,
        ticker=ticker,
        cik=metadata_values.get("cik"),
        company_name=metadata_values.get("company_name"),
        form_type=form_type,
        filing_date=metadata_values.get("filing_date"),
        period_of_report=metadata_values.get("period_of_report"),
        raw_path=str(filing_path),
        raw_sha256=raw_sha256,
        clean_text_path=str(clean_text_path),
    )

    filing_id = upsert_filing(connection, metadata)
    replace_filing_chunks(connection, filing_id, chunks)

    return IngestionResult(
        accession_number=accession_number,
        ticker=ticker,
        raw_path=str(filing_path),
        clean_text_path=str(clean_text_path),
        section_count=len(sections),
        chunk_count=len(chunks),
    )


def find_downloaded_filings(raw_dir: str | Path, ticker: str, form_type: str) -> list[Path]:
    filings_dir = Path(raw_dir) / "sec-edgar-filings" / ticker.upper() / form_type.upper()
    return sorted(
        filings_dir.glob("*/full-submission.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def parse_submission_metadata(submission: str) -> dict[str, str]:
    header_match = HEADER_RE.search(submission)
    source = header_match.group("header") if header_match else submission[:20_000]

    fields = {}
    for line in source.splitlines():
        match = HEADER_FIELD_RE.match(line)
        if not match:
            continue

        name = re.sub(r"\s+", " ", match.group("name")).strip().upper()
        fields[name] = match.group("value").strip()

    return {
        "accession_number": fields.get("ACCESSION NUMBER"),
        "cik": fields.get("CENTRAL INDEX KEY"),
        "company_name": fields.get("COMPANY CONFORMED NAME"),
        "form_type": fields.get("CONFORMED SUBMISSION TYPE") or fields.get("FORM TYPE"),
        "filing_date": _format_sec_date(fields.get("FILED AS OF DATE")),
        "period_of_report": _format_sec_date(fields.get("CONFORMED PERIOD OF REPORT")),
    }


def write_clean_text(
    *,
    processed_dir: str | Path,
    ticker: str,
    form_type: str,
    accession_number: str,
    text: str,
) -> Path:
    output_path = (
        Path(processed_dir)
        / "clean_text"
        / ticker.upper()
        / form_type.upper()
        / accession_number
        / "clean-text.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _format_sec_date(value: str | None) -> str | None:
    if not value:
        return None

    value = value.strip()
    if not re.fullmatch(r"\d{8}", value):
        return value

    return f"{value[:4]}-{value[4:6]}-{value[6:]}"
