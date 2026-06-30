import argparse
import os

from dotenv import load_dotenv

from signalforge.config import RuntimeConfig
from signalforge.ingestion import ingest_sec_10k_filings
from signalforge.storage import connect_database


def main() -> None:
    load_dotenv()
    args = parse_args()

    company_name = args.company_name or os.getenv("SEC_COMPANY_NAME")
    email_address = args.email or os.getenv("SEC_EMAIL")

    with connect_database(args.db_path) as connection:
        results = ingest_sec_10k_filings(
            ticker=args.ticker,
            connection=connection,
            company_name=company_name,
            email_address=email_address,
            limit=args.limit,
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            download=not args.no_download,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

    for result in results:
        print(
            f"{result.ticker} {result.accession_number}: "
            f"{result.section_count} sections, {result.chunk_count} chunks"
        )
        print(f"  raw:   {result.raw_path}")
        print(f"  clean: {result.clean_text_path}")


def parse_args() -> argparse.Namespace:
    config = RuntimeConfig.from_environment()
    parser = argparse.ArgumentParser(description="Ingest SEC 10-K filings into local SQLite storage.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol to ingest, for example NVDA.")
    parser.add_argument("--limit", type=int, default=1, help="Number of recent 10-K filings to ingest.")
    parser.add_argument("--company-name", help="Company/app name for the SEC downloader user agent.")
    parser.add_argument("--email", help="Email address for the SEC downloader user agent.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory for raw SEC downloads.")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory for clean text output.")
    parser.add_argument("--db-path", default=config.database_target, help="Database URL or SQLite path.")
    parser.add_argument("--chunk-size", type=int, default=4_000, help="Target chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=500, help="Chunk overlap in characters.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Process existing files under raw-dir without calling SEC EDGAR.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
