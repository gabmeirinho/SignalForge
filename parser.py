import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment


@dataclass(frozen=True)
class SecDocument:
    form_type: str
    filename: str | None
    description: str | None
    text: str


DOCUMENT_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.IGNORECASE | re.DOTALL)
TEXT_RE = re.compile(r"<TEXT>\s*(.*?)\s*</TEXT>", re.IGNORECASE | re.DOTALL)
FIELD_RE = re.compile(r"^<(?P<name>[A-Z0-9.-]+)>\s*(?P<value>.*)$", re.IGNORECASE)
BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "caption",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}


def parse_sec_submission_documents(submission: str) -> list[SecDocument]:
    documents = []

    for match in DOCUMENT_RE.finditer(submission):
        raw_document = match.group(1)
        fields = {}

        for line in raw_document.splitlines():
            field_match = FIELD_RE.match(line.strip())
            if field_match:
                fields[field_match.group("name").upper()] = field_match.group("value").strip()

        text_match = TEXT_RE.search(raw_document)
        if not text_match:
            continue

        documents.append(
            SecDocument(
                form_type=fields.get("TYPE", ""),
                filename=fields.get("FILENAME"),
                description=fields.get("DESCRIPTION"),
                text=text_match.group(1).strip(),
            )
        )

    return documents


def extract_primary_document(submission: str, form_type: str = "10-K") -> SecDocument:
    documents = parse_sec_submission_documents(submission)
    target_form = form_type.upper()

    for document in documents:
        if document.form_type.upper() == target_form:
            return document

    if documents:
        available = ", ".join(document.form_type for document in documents[:10])
        raise ValueError(f"No {form_type} document found. Available document types: {available}")

    return SecDocument(form_type=form_type, filename=None, description=None, text=submission.strip())


def extract_primary_document_text(submission: str, form_type: str = "10-K") -> str:
    return extract_primary_document(submission, form_type=form_type).text


def clean_sec_html(html: str) -> str:
    if "<DOCUMENT>" in html[:100_000].upper():
        html = extract_primary_document_text(html)

    soup = BeautifulSoup(html, "lxml")

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    for tag in soup(["script", "style", "noscript", "meta", "link"]):
        tag.decompose()

    for tag in soup.find_all():
        name = tag.name.lower() if tag.name else ""
        attrs = tag.attrs or {}
        style = re.sub(r"\s+", "", attrs.get("style", "").lower())

        if name == "ix:header" or name.startswith(("xbrli:", "xbrldi:", "link:")):
            tag.decompose()
            continue

        if "display:none" in style or "visibility:hidden" in style:
            tag.decompose()
            continue

        if name in BLOCK_TAGS:
            tag.insert_before("\n")
            tag.insert_after("\n")

    text = soup.get_text(separator="")

    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[–—]", "-", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_sec_filing_to_text(path: str, form_type: str = "10-K", encoding: str = "utf-8") -> str:
    with open(path, encoding=encoding) as filing:
        return clean_sec_html(extract_primary_document_text(filing.read(), form_type=form_type))
