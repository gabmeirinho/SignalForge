import pytest

from parser import clean_sec_html, extract_primary_document, extract_primary_document_text


def test_extracts_only_primary_10k_from_full_submission():
    submission = """
<SEC-DOCUMENT>example.txt
<DOCUMENT>
<TYPE>EX-21.1
<FILENAME>subsidiaries.htm
<DESCRIPTION>Subsidiaries
<TEXT>
<html><body><p>Exhibit subsidiaries content</p></body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>10-K
<FILENAME>company-10k.htm
<DESCRIPTION>Annual report
<TEXT>
<html><body><p>Main annual report content</p></body></html>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""

    document = extract_primary_document(submission)
    text = clean_sec_html(submission)

    assert document.form_type == "10-K"
    assert document.filename == "company-10k.htm"
    assert "Main annual report content" in text
    assert "Exhibit subsidiaries content" not in text


def test_removes_hidden_html_and_inline_xbrl_metadata():
    html = """
<html>
  <body>
    <ix:header>Hidden XBRL header</ix:header>
    <xbrli:context>Hidden XBRL context</xbrli:context>
    <div style="display: none">Hidden display text</div>
    <div style="visibility: hidden">Hidden visibility text</div>
    <p>Visible filing text</p>
  </body>
</html>
"""

    text = clean_sec_html(html)

    assert "Visible filing text" in text
    assert "Hidden XBRL header" not in text
    assert "Hidden XBRL context" not in text
    assert "Hidden display text" not in text
    assert "Hidden visibility text" not in text


def test_preserves_inline_text_without_extra_line_breaks():
    html = """
<html>
  <body>
    <p>NVIDIA CORP<span>ORATION</span></p>
    <p>FORM <span>10-K</span></p>
  </body>
</html>
"""

    text = clean_sec_html(html)

    assert "NVIDIA CORPORATION" in text
    assert "FORM 10-K" in text
    assert "NVIDIA CORP\nORATION" not in text
    assert "FORM\n10-K" not in text


def test_adds_readable_spacing_between_block_elements():
    html = "<html><body><p>First paragraph</p><p>Second paragraph</p></body></html>"

    text = clean_sec_html(html)

    assert text == "First paragraph\n\nSecond paragraph"


def test_normalizes_whitespace_and_common_punctuation():
    html = """
<html>
  <body>
    <p>First&nbsp;&nbsp;line\twith   spaces</p>


    <p>“Quoted” ‘text’ and 2024–2026 growth</p>
  </body>
</html>
"""

    text = clean_sec_html(html)

    assert "First line with spaces" in text
    assert '"Quoted" \'text\' and 2024-2026 growth' in text
    assert "\n\n\n" not in text


def test_raises_when_full_submission_has_no_requested_form_type():
    submission = """
<DOCUMENT>
<TYPE>EX-21.1
<FILENAME>subsidiaries.htm
<TEXT>
<html><body><p>Exhibit content</p></body></html>
</TEXT>
</DOCUMENT>
"""

    with pytest.raises(ValueError, match="No 10-K document found"):
        extract_primary_document_text(submission)
