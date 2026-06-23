import re
from dataclasses import dataclass


SECTION_ORDER = ("1", "1A", "1B", "2", "3", "7", "7A", "8")
DEFAULT_TARGET_SECTIONS = ("1", "1A", "7", "7A")

SECTION_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
}

ITEM_HEADING_RE = re.compile(
    r"(?im)^[ \t]*Item[ \t]+(?P<section>1A|1B|7A|1|2|3|7|8)\.?"
    r"(?![A-Z0-9])[ \t]*(?P<title>[^\n]{0,160})$"
)


@dataclass(frozen=True)
class FilingSection:
    section_id: str
    title: str
    text: str


@dataclass(frozen=True)
class TextChunk:
    section_id: str
    section_title: str
    chunk_index: int
    text: str


def split_10k_sections(
    text: str,
    target_sections: tuple[str, ...] = DEFAULT_TARGET_SECTIONS,
    min_section_chars: int = 1_000,
) -> list[FilingSection]:
    headings = list(ITEM_HEADING_RE.finditer(text))
    sections = []

    for index, heading in enumerate(headings):
        section_id = heading.group("section").upper()
        if section_id not in target_sections:
            continue

        next_start = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        section_text = text[heading.end() : next_start].strip()

        if len(section_text) < min_section_chars:
            continue

        title = _normalize_section_title(section_id, heading.group("title"))
        sections.append(FilingSection(section_id=section_id, title=title, text=section_text))

    return _dedupe_sections_keep_longest(sections)


def chunk_sections(
    sections: list[FilingSection],
    chunk_size: int = 4_000,
    overlap: int = 500,
) -> list[TextChunk]:
    chunks = []

    for section in sections:
        for chunk_index, chunk_text in enumerate(chunk_text_by_paragraph(section.text, chunk_size, overlap)):
            chunks.append(
                TextChunk(
                    section_id=section.section_id,
                    section_title=section.title,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )

    return chunks


def chunk_text_by_paragraph(text: str, chunk_size: int = 4_000, overlap: int = 500) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to zero")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue

        candidate = f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.extend(_split_oversized_text(current, chunk_size, overlap))
        current = _overlap_tail(current, overlap)
        current = f"{current}\n\n{paragraph}".strip() if current else paragraph

    if current:
        chunks.extend(_split_oversized_text(current, chunk_size, overlap))

    return chunks


def _normalize_section_title(section_id: str, raw_title: str) -> str:
    title = re.sub(r"\s+", " ", raw_title).strip(" .")
    if not title:
        return SECTION_TITLES[section_id]
    return title


def _dedupe_sections_keep_longest(sections: list[FilingSection]) -> list[FilingSection]:
    longest_by_id = {}

    for section in sections:
        existing = longest_by_id.get(section.section_id)
        if existing is None or len(section.text) > len(existing.text):
            longest_by_id[section.section_id] = section

    return [
        longest_by_id[section_id]
        for section_id in DEFAULT_TARGET_SECTIONS
        if section_id in longest_by_id
    ]


def _split_oversized_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap == 0:
        return ""

    tail = text[-overlap:].strip()
    paragraph_start = tail.find("\n\n")
    if paragraph_start > 0:
        return tail[paragraph_start:].strip()
    return tail
