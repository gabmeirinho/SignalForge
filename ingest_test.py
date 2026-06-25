from pathlib import Path

from parser import clean_sec_html, extract_primary_document
from sections import (
    extract_10k_sections_from_cross_reference_index,
    extract_10k_sections_with_edgartools,
    extract_10k_sections
)

# path = Path(
#     "data/raw/sec-edgar-filings/INTC/10-K/"
#     "0000050863-26-000011/full-submission.txt"
# )

path = Path(
    "data/raw/sec-edgar-filings/NVDA/10-K/"
    "0001045810-26-000021/full-submission.txt"
)

document = extract_primary_document(path.read_text())
edgar_sections = extract_10k_sections_with_edgartools(document.text)
index_sections = extract_10k_sections_from_cross_reference_index(document.text)
standard_sections = extract_10k_sections(document.text, clean_text=clean_sec_html(document.text))

print("Edgar Tools:", [(section.section_id, section.title, len(section.text)) for section in edgar_sections])
print("Cross-Reference:", [(section.section_id, section.title, len(section.text)) for section in index_sections])
print("Standard", [(section.section_id, section.title, len(section.text)) for section in standard_sections])
