import pytest

import sections as sections_module
from sections import (
    FilingSection,
    chunk_sections,
    chunk_text_by_paragraph,
    extract_10k_sections,
    extract_10k_sections_from_cross_reference_index,
)


def test_chunk_text_by_paragraph_uses_overlap_and_validates_inputs():
    text = "\n\n".join(
        [
            "alpha " * 20,
            "bravo " * 20,
            "charlie " * 20,
            "delta " * 20,
        ]
    )

    chunks = chunk_text_by_paragraph(text, chunk_size=180, overlap=30)

    assert len(chunks) > 1
    assert all(len(chunk) <= 180 for chunk in chunks)
    assert "alpha" in chunks[0]
    assert any("bravo" in chunk for chunk in chunks[1:])

    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_text_by_paragraph(text, chunk_size=100, overlap=100)


def test_chunk_sections_keeps_section_metadata():
    sections = [
        FilingSection(
            section_id="1",
            title="Business",
            text="Business paragraph one.\n\nBusiness paragraph two.",
        ),
        FilingSection(
            section_id="1A",
            title="Risk Factors",
            text="Risk paragraph one.\n\nRisk paragraph two.",
        ),
    ]

    chunks = chunk_sections(sections, chunk_size=80, overlap=10)

    assert {chunk.section_id for chunk in chunks} == {"1", "1A"}
    assert all(chunk.section_title for chunk in chunks)
    assert all(chunk.chunk_index >= 0 for chunk in chunks)


def test_extract_10k_sections_returns_only_items_found_by_supported_extractors(monkeypatch):
    edgar_sections = [
        sections_module.FilingSection(
            section_id="1A",
            title="Risk Factors",
            text="EdgarTools risk content. " * 100,
        )
    ]
    monkeypatch.setattr(
        sections_module,
        "extract_10k_sections_with_edgartools",
        lambda *args, **kwargs: edgar_sections,
    )
    monkeypatch.setattr(
        sections_module,
        "extract_10k_sections_from_cross_reference_index",
        lambda *args, **kwargs: [],
    )

    extracted = extract_10k_sections(
        "<html></html>",
        min_section_chars=20,
    )

    assert [section.section_id for section in extracted] == ["1A"]
    assert extracted[0].text.startswith("EdgarTools risk content")


def test_extract_10k_sections_rejects_anomalously_short_edgartools_sections(monkeypatch):
    class FakeSection:
        def text(self):
            return "too short"

    class FakeSections:
        def get_item(self, section_id):
            return FakeSection() if section_id == "1" else None

    class FakeDocument:
        sections = FakeSections()

    class FakeParser:
        def __init__(self, config):
            self.config = config

        def parse(self, html):
            return FakeDocument()

    monkeypatch.setattr(sections_module, "HTMLParser", FakeParser)

    extracted = sections_module.extract_10k_sections_with_edgartools(
        "<html></html>",
        target_sections=("1",),
        min_section_chars=20,
    )

    assert extracted == []


def test_extract_10k_sections_with_edgartools_uses_item_lookup(monkeypatch):
    requested_items = []

    class FakeSection:
        def text(self):
            return "Business content. " * 100

    class FakeSections:
        def get_item(self, section_id):
            requested_items.append(section_id)
            return FakeSection() if section_id == "1" else None

    class FakeDocument:
        sections = FakeSections()

    class FakeParser:
        def __init__(self, config):
            self.config = config

        def parse(self, html):
            return FakeDocument()

    monkeypatch.setattr(sections_module, "HTMLParser", FakeParser)

    extracted = sections_module.extract_10k_sections_with_edgartools(
        "<html></html>",
        target_sections=("1", "1A"),
        min_section_chars=20,
    )

    assert requested_items == ["1", "1A"]
    assert [section.section_id for section in extracted] == ["1"]


def test_extracts_non_contiguous_items_from_cross_reference_index():
    html = """
<html><body>
  <div id="page-1"></div>
  <hr style="page-break-after: always">
  <p>Business page one content. Business page one content.</p>
  <hr style="page-break-after: always">
  <p>Business page two content. Business page two content.</p>
  <hr style="page-break-after: always">

  <div id="page-3"></div>
  <hr style="page-break-after: always">
  <p>Risk page content. Risk page content.</p>
  <hr style="page-break-after: always">

  <div id="page-5"></div>
  <hr style="page-break-after: always">
  <p>Accounting estimate content. Accounting estimate content.</p>
  <hr style="page-break-after: always">

  <h2>Form 10-K Cross-Reference Index</h2>
  <table>
    <tr><th>Item Number</th><th>Item</th></tr>
    <tr><td>Item 1.</td><td>Business</td></tr>
    <tr><td>Description of business</td><td>Pages <a href="#page-1">1</a> - 2</td></tr>
    <tr><td>Item 1A.</td><td>Risk Factors</td><td>Page <a href="#page-3">3</a></td></tr>
    <tr><td>Item 7.</td><td>Management's Discussion and Analysis</td></tr>
    <tr><td>Critical accounting estimates</td><td>Page <a href="#page-5">5</a></td></tr>
    <tr><td>Item 7A.</td><td>Market Risk</td><td>None</td></tr>
  </table>
</body></html>
"""

    extracted = extract_10k_sections_from_cross_reference_index(
        html,
        min_section_chars=20,
    )

    assert [section.section_id for section in extracted] == ["1", "1A", "7"]
    assert "Business page one content" in extracted[0].text
    assert "Business page two content" in extracted[0].text
    assert "Risk page content" in extracted[1].text
    assert "Accounting estimate content" in extracted[2].text
    assert "Risk page content" not in extracted[0].text


def test_extract_10k_sections_uses_index_as_authority_when_present(monkeypatch):
    edgar_section = sections_module.FilingSection(
        section_id="1",
        title="Business",
        text="EdgarTools business content. " * 100,
    )
    monkeypatch.setattr(
        sections_module,
        "extract_10k_sections_with_edgartools",
        lambda *args, **kwargs: [edgar_section],
    )

    def extract_from_index(*args, **kwargs):
        return [
            sections_module.FilingSection(
                section_id="1",
                title="Business",
                text="Indexed business content. " * 100,
            ),
            sections_module.FilingSection(
                section_id="1A",
                title="Risk Factors",
                text="Indexed risk content. " * 100,
            )
        ]

    monkeypatch.setattr(
        sections_module,
        "extract_10k_sections_from_cross_reference_index",
        extract_from_index,
    )

    extracted = extract_10k_sections(
        "<html></html>",
        target_sections=("1", "1A"),
        min_section_chars=20,
    )

    assert [section.section_id for section in extracted] == ["1", "1A"]
    assert extracted[0].text.startswith("Indexed business content")
    assert extracted[1].text.startswith("Indexed risk content")
