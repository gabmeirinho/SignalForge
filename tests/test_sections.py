import pytest

from sections import chunk_sections, chunk_text_by_paragraph, split_10k_sections


def test_split_10k_sections_discards_table_of_contents_and_keeps_body_sections():
    text = """
Table of Contents

Item 1. Business
Item 1A. Risk Factors
Item 7. Management's Discussion and Analysis
Item 7A. Quantitative and Qualitative Disclosures About Market Risk

Item 1. Business
Business body paragraph. Business body paragraph. Business body paragraph.

Item 1A. Risk Factors
Risk body paragraph. Risk body paragraph. Risk body paragraph.

Item 7. Management's Discussion and Analysis
MD&A body paragraph. MD&A body paragraph. MD&A body paragraph.

Item 7A. Quantitative and Qualitative Disclosures About Market Risk
Market risk body paragraph. Market risk body paragraph. Market risk body paragraph.

Item 8. Financial Statements
"""

    sections = split_10k_sections(text, min_section_chars=20)

    assert [section.section_id for section in sections] == ["1", "1A", "7", "7A"]
    assert sections[0].title == "Business"
    assert "Business body paragraph" in sections[0].text
    assert "Risk body paragraph" in sections[1].text
    assert "MD&A body paragraph" in sections[2].text
    assert "Market risk body paragraph" in sections[3].text


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
    sections = split_10k_sections(
        """
Item 1. Business
Business paragraph one.

Business paragraph two.

Item 1A. Risk Factors
Risk paragraph one.

Risk paragraph two.

Item 7. MD&A
Discussion paragraph one.

Discussion paragraph two.
""",
        target_sections=("1", "1A"),
        min_section_chars=10,
    )

    chunks = chunk_sections(sections, chunk_size=80, overlap=10)

    assert {chunk.section_id for chunk in chunks} == {"1", "1A"}
    assert all(chunk.section_title for chunk in chunks)
    assert all(chunk.chunk_index >= 0 for chunk in chunks)
