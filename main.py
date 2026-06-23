from sec_edgar_downloader import Downloader
from pathlib import Path

from parser import parse_sec_filing_to_text

FILING_PATH = Path("sec-edgar-filings/NVDA/10-K/0001045810-26-000021/full-submission.txt")

if not FILING_PATH.exists():
    dl = Downloader("SignalForge", "gabrielsilva.mei@hotmail.com")
    dl.get("10-K", "NVDA", limit=1)

text = parse_sec_filing_to_text(str(FILING_PATH))

print(text[:4_000])
