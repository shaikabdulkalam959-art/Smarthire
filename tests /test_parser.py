"""Tests for the resume text parser."""
from __future__ import annotations

import io
import struct
import zlib

import pytest

from src.parsing.resume_parser import extract_resume_text


# ── TXT ─────────────────────────────────────────────────────────────────────

def test_extract_txt_returns_content() -> None:
    content = b"Python developer with 3 years experience in Django and PostgreSQL."
    assert extract_resume_text("resume.txt", content) == content.decode("utf-8")


def test_extract_txt_handles_utf8() -> None:
    content = "Développeur Python — expérience avec Django.".encode("utf-8")
    result = extract_resume_text("my_resume.txt", content)
    assert "Développeur" in result


# ── DOCX ─────────────────────────────────────────────────────────────────────

def _make_minimal_docx(paragraph_text: str) -> bytes:
    """Return the bytes of a minimal but valid DOCX containing one paragraph."""
    # A DOCX is a ZIP file. We need the minimum required XML files.
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mo="http://schemas.microsoft.com/office/mac/office/2008/main" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:mv="urn:schemas-microsoft-com:mac:vml" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14">'
        "<w:body>"
        f"<w:p><w:r><w:t>{paragraph_text}</w:t></w:r></w:p>"
        "<w:sectPr/>"
        "</w:body>"
        "</w:document>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    word_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )

    buf = io.BytesIO()
    import zipfile
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/_rels/document.xml.rels", word_rels_xml)
        zf.writestr("word/document.xml", doc_xml)
    return buf.getvalue()


def test_extract_docx_returns_paragraph_text() -> None:
    docx_bytes = _make_minimal_docx("Senior Data Scientist with Python expertise")
    result = extract_resume_text("cv.docx", docx_bytes)
    assert "Senior Data Scientist" in result


# ── Unsupported format ────────────────────────────────────────────────────────

def test_unsupported_format_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        extract_resume_text("resume.odt", b"some content")


def test_no_extension_raises_value_error() -> None:
    with pytest.raises(ValueError):
        extract_resume_text("resume", b"some content")
