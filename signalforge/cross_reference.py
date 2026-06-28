import re

from bs4 import BeautifulSoup, Tag

from signalforge.parser import clean_sec_html


CROSS_REFERENCE_HEADING_RE = re.compile(
    r"\b(?:form\s+10-k\s+)?cross[-\s]reference\s+index\b",
    re.IGNORECASE,
)
CROSS_REFERENCE_ITEM_RE = re.compile(
    r"^\s*Item\s+(?P<section>\d{1,2}[A-Z]?)\.",
    re.IGNORECASE,
)
PAGE_REFERENCE_RE = re.compile(r"\bPages?\s+(?P<references>.+)$", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"\d(?:\s*\d)*")
PAGE_RANGE_RE = re.compile(
    r"(?P<start>\d(?:\s*\d)*)\s*-\s*(?P<end>\d(?:\s*\d)*)"
)


def extract_cross_referenced_items(
    html: str,
    *,
    target_items: tuple[str, ...],
    min_item_chars: int = 1_000,
) -> dict[str, str]:
    """Extract SEC items whose content is located by a filing-provided index."""
    soup = BeautifulSoup(html, "lxml")
    index_table = _find_cross_reference_index_table(soup)
    if index_table is None:
        return {}

    item_ranges = _parse_item_ranges(index_table, target_items)
    extracted = {}

    for item_id in target_items:
        fragments = []
        for start_page, end_page, anchor_id in _merge_page_ranges(
            item_ranges.get(item_id, [])
        ):
            fragment = _extract_page_span(
                soup,
                anchor_id=anchor_id,
                page_count=end_page - start_page + 1,
            )
            if fragment:
                fragments.append(fragment)

        item_text = "\n\n".join(fragments).strip()
        if len(item_text) >= min_item_chars:
            extracted[item_id] = item_text

    return extracted


def _find_cross_reference_index_table(soup: BeautifulSoup) -> Tag | None:
    headings = soup.find_all(
        string=lambda value: value and CROSS_REFERENCE_HEADING_RE.search(value)
    )

    for heading in reversed(headings):
        for table in heading.parent.find_all_next("table", limit=5):
            table_text = _normalized_text(table)
            if "Item Number" not in table_text:
                continue

            item_row_count = sum(
                bool(CROSS_REFERENCE_ITEM_RE.match(_normalized_text(row)))
                for row in table.find_all("tr")
            )
            if item_row_count >= 2:
                return table

    return None


def _parse_item_ranges(
    table: Tag,
    target_items: tuple[str, ...],
) -> dict[str, list[tuple[int, int, str]]]:
    ranges_by_item = {item_id: [] for item_id in target_items}
    active_item = None

    for row in table.find_all("tr"):
        row_text = _normalized_text(row)
        item_match = CROSS_REFERENCE_ITEM_RE.match(row_text)
        if item_match:
            active_item = item_match.group("section").upper()

        if active_item not in ranges_by_item:
            continue

        page_match = PAGE_REFERENCE_RE.search(row_text)
        if not page_match:
            continue

        anchors_by_page = {}
        for link in row.find_all("a", href=True):
            href = link["href"]
            page = _parse_page_number(link.get_text(" ", strip=True))
            if href.startswith("#") and page is not None:
                anchors_by_page.setdefault(page, href[1:])

        for start_page, end_page in _parse_page_references(
            page_match.group("references")
        ):
            anchor_id = anchors_by_page.get(start_page)
            if anchor_id:
                ranges_by_item[active_item].append(
                    (start_page, end_page, anchor_id)
                )

    return ranges_by_item


def _parse_page_references(value: str) -> list[tuple[int, int]]:
    references = []

    for part in value.split(","):
        range_match = PAGE_RANGE_RE.search(part)
        if range_match:
            start = _parse_page_number(range_match.group("start"))
            end = _parse_page_number(range_match.group("end"))
        else:
            page_match = PAGE_NUMBER_RE.search(part)
            start = end = _parse_page_number(page_match.group()) if page_match else None

        if start is not None and end is not None and end >= start:
            references.append((start, end))

    return references


def _parse_page_number(value: str) -> int | None:
    digits = re.sub(r"\s+", "", value)
    return int(digits) if digits.isdigit() else None


def _merge_page_ranges(
    ranges: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    merged = []

    for start, end, anchor_id in sorted(ranges, key=lambda value: (value[0], value[1])):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end, anchor_id])
            continue

        merged[-1][1] = max(merged[-1][1], end)

    return [tuple(value) for value in merged]


def _extract_page_span(
    soup: BeautifulSoup,
    *,
    anchor_id: str,
    page_count: int,
) -> str:
    anchor = soup.find(id=anchor_id)
    if anchor is None or page_count < 1:
        return ""

    fragments = []
    page_breaks = 0
    first_tag_seen = False

    for sibling in anchor.next_siblings:
        if not isinstance(sibling, Tag):
            continue

        if _is_page_break(sibling):
            if not first_tag_seen:
                first_tag_seen = True
                continue

            page_breaks += 1
            if page_breaks >= page_count:
                break
            continue

        first_tag_seen = True
        fragments.append(str(sibling))

    return clean_sec_html("".join(fragments))


def _is_page_break(tag: Tag) -> bool:
    if tag.name != "hr":
        return False

    style = re.sub(r"\s+", "", tag.get("style", "").lower())
    return "page-break-after:always" in style or "page-break-before:always" in style


def _normalized_text(tag: Tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
