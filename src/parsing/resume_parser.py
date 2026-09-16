from __future__ import annotations

from io import BytesIO


def extract_resume_text(filename: str, content: bytes) -> str:
    """Extract text from a TXT, PDF, or DOCX upload without retaining the file."""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix == "txt":
        return content.decode("utf-8", errors="replace")
    if suffix == "pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
    if suffix == "docx":
        from docx import Document
        return "\n".join(paragraph.text for paragraph in Document(BytesIO(content)).paragraphs)
    raise ValueError("Unsupported resume format. Upload a TXT, PDF, or DOCX file.")
