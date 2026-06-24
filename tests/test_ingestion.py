import hashlib
from pathlib import Path

from ingestion import ingest_downloaded_filing, parse_submission_metadata
from storage import connect_database, initialize_database


def test_parse_submission_metadata_from_sec_header():
    submission = """
<SEC-HEADER>
ACCESSION NUMBER:        0000000000-26-000001
CONFORMED SUBMISSION TYPE: 10-K
CONFORMED PERIOD OF REPORT: 20260125
FILED AS OF DATE:        20260225

FILER:
    COMPANY DATA:
        COMPANY CONFORMED NAME:            EXAMPLE CORP
        CENTRAL INDEX KEY:                 0000000000
</SEC-HEADER>
"""

    metadata = parse_submission_metadata(submission)

    assert metadata["accession_number"] == "0000000000-26-000001"
    assert metadata["form_type"] == "10-K"
    assert metadata["period_of_report"] == "2026-01-25"
    assert metadata["filing_date"] == "2026-02-25"
    assert metadata["company_name"] == "EXAMPLE CORP"
    assert metadata["cik"] == "0000000000"


def test_ingest_downloaded_filing_writes_clean_text_and_sqlite_rows(tmp_path: Path):
    filing_path = (
        tmp_path
        / "raw"
        / "sec-edgar-filings"
        / "NVDA"
        / "10-K"
        / "0000000000-26-000001"
        / "full-submission.txt"
    )
    filing_path.parent.mkdir(parents=True)
    filing_path.write_text(_sample_submission(), encoding="utf-8")

    db_path = tmp_path / "signalforge.sqlite3"
    processed_dir = tmp_path / "processed"

    with connect_database(db_path) as connection:
        initialize_database(connection)
        result = ingest_downloaded_filing(
            filing_path=filing_path,
            ticker="NVDA",
            connection=connection,
            processed_dir=processed_dir,
            chunk_size=250,
            overlap=50,
        )

        filing_rows = connection.execute("SELECT * FROM filings").fetchall()
        chunk_rows = connection.execute("SELECT * FROM chunks ORDER BY section_id, chunk_index").fetchall()

    assert result.accession_number == "0000000000-26-000001"
    assert result.section_count == 4
    assert result.chunk_count > 4
    assert Path(result.clean_text_path).exists()
    assert "Visible business text" in Path(result.clean_text_path).read_text(encoding="utf-8")

    assert len(filing_rows) == 1
    assert filing_rows[0]["ticker"] == "NVDA"
    assert filing_rows[0]["company_name"] == "EXAMPLE CORP"
    assert filing_rows[0]["filing_date"] == "2026-02-25"
    assert filing_rows[0]["raw_sha256"] == hashlib.sha256(filing_path.read_bytes()).hexdigest()
    assert {row["section_id"] for row in chunk_rows} == {"1", "1A", "7", "7A"}
    assert all(row["text"] for row in chunk_rows)


def _sample_submission() -> str:
    return f"""
<SEC-DOCUMENT>example.txt
<SEC-HEADER>
ACCESSION NUMBER:        0000000000-26-000001
CONFORMED SUBMISSION TYPE: 10-K
CONFORMED PERIOD OF REPORT: 20260125
FILED AS OF DATE:        20260225

FILER:
    COMPANY DATA:
        COMPANY CONFORMED NAME:            EXAMPLE CORP
        CENTRAL INDEX KEY:                 0000000000
</SEC-HEADER>
<DOCUMENT>
<TYPE>10-K
<FILENAME>example-10k.htm
<DESCRIPTION>Annual report
<TEXT>
<html>
  <body>
    <div style="display:none">Hidden metadata</div>
    <p>Item 1. Business</p>
    <p>{_repeat("Visible business text.", 80)}</p>
    <p>Item 1A. Risk Factors</p>
    <p>{_repeat("Visible risk text.", 80)}</p>
    <p>Item 7. Management's Discussion and Analysis</p>
    <p>{_repeat("Visible discussion text.", 80)}</p>
    <p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p>
    <p>{_repeat("Visible market risk text.", 80)}</p>
    <p>Item 8. Financial Statements</p>
  </body>
</html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-21.1
<FILENAME>exhibit.htm
<TEXT>
<html><body><p>Exhibit should not appear</p></body></html>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""


def _repeat(text: str, count: int) -> str:
    return " ".join([text] * count)
