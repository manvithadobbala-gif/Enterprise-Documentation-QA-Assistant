import os
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import pymupdf

from ingestion.text_cleaner import clean_text, extract_text_from_html


@dataclass
class DocumentPage:
    """Represents an extracted unit of a document (e.g., a page or section)."""
    content: str
    metadata: dict = field(default_factory=dict)


def compute_file_hash(file_bytes_or_path) -> str:
    """Computes SHA256 hash of a file path or bytes to detect duplicates."""
    hasher = hashlib.sha256()
    if isinstance(file_bytes_or_path, (str, Path)):
        with open(file_bytes_or_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    elif isinstance(file_bytes_or_path, bytes):
        hasher.update(file_bytes_or_path)
    else:
        # File-like object (e.g., Streamlit UploadedFile)
        file_bytes_or_path.seek(0)
        hasher.update(file_bytes_or_path.read())
        file_bytes_or_path.seek(0)
    return hasher.hexdigest()


class DocumentLoader:
    """Loads documents from PDF, TXT, Markdown, and HTML with metadata preservation."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".html", ".htm"}

    @classmethod
    def load_file(cls, file_path_or_buffer, filename: Optional[str] = None) -> List[DocumentPage]:
        """
        Loads a document and returns a list of DocumentPage objects with metadata.
        Accepts either a string path/Path object or an in-memory byte buffer.
        """
        if isinstance(file_path_or_buffer, (str, Path)):
            path = Path(file_path_or_buffer)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            fname = filename or path.name
            ext = path.suffix.lower()
            with open(path, "rb") as f:
                content_bytes = f.read()
        else:
            if not filename:
                raise ValueError("Filename must be provided when passing a buffer/stream.")
            fname = filename
            ext = Path(filename).suffix.lower()
            if hasattr(file_path_or_buffer, "read"):
                file_path_or_buffer.seek(0)
                content_bytes = file_path_or_buffer.read()
                file_path_or_buffer.seek(0)
            else:
                content_bytes = bytes(file_path_or_buffer)

        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: '{ext}'. Supported: {cls.SUPPORTED_EXTENSIONS}")

        file_hash = compute_file_hash(content_bytes)
        base_meta = {
            "document": fname,
            "filename": fname,
            "file_type": ext,
            "file_hash": file_hash,
        }

        if ext == ".pdf":
            return cls._load_pdf(content_bytes, base_meta)
        elif ext in {".txt"}:
            return cls._load_txt(content_bytes, base_meta)
        elif ext in {".md", ".markdown"}:
            return cls._load_markdown(content_bytes, base_meta)
        elif ext in {".html", ".htm"}:
            return cls._load_html(content_bytes, base_meta)

        return []

    @classmethod
    def _load_pdf(cls, content_bytes: bytes, base_meta: dict) -> List[DocumentPage]:
        pages = []
        doc = pymupdf.open(stream=content_bytes, filetype="pdf")
        total_pages = len(doc)

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_text = page.get_text()
            cleaned = clean_text(page_text)
            if cleaned:
                meta = {
                    **base_meta,
                    "page": page_idx + 1,
                    "total_pages": total_pages,
                    "section": f"Page {page_idx + 1}",
                }
                pages.append(DocumentPage(content=cleaned, metadata=meta))
        doc.close()
        return pages

    @classmethod
    def _load_txt(cls, content_bytes: bytes, base_meta: dict) -> List[DocumentPage]:
        text = content_bytes.decode("utf-8", errors="replace")
        cleaned = clean_text(text)
        if not cleaned:
            return []
        meta = {
            **base_meta,
            "page": 1,
            "total_pages": 1,
            "section": "General",
        }
        return [DocumentPage(content=cleaned, metadata=meta)]

    @classmethod
    def _load_markdown(cls, content_bytes: bytes, base_meta: dict) -> List[DocumentPage]:
        text = content_bytes.decode("utf-8", errors="replace")
        cleaned = clean_text(text)
        if not cleaned:
            return []

        # Split by markdown headers (# Header, ## Subheader) while preserving sections
        lines = cleaned.split("\n")
        pages = []
        current_section = "Introduction"
        current_lines = []
        section_idx = 1

        for line in lines:
            if line.startswith("#"):
                # Header encountered
                if current_lines:
                    sec_text = "\n".join(current_lines).strip()
                    if sec_text:
                        pages.append(DocumentPage(
                            content=sec_text,
                            metadata={
                                **base_meta,
                                "page": section_idx,
                                "section": current_section,
                            }
                        ))
                        section_idx += 1
                        current_lines = []
                current_section = line.lstrip("#").strip()
            current_lines.append(line)

        if current_lines:
            sec_text = "\n".join(current_lines).strip()
            if sec_text:
                pages.append(DocumentPage(
                    content=sec_text,
                    metadata={
                        **base_meta,
                        "page": section_idx,
                        "section": current_section,
                    }
                ))

        # If no markdown headings were detected, treat as single page
        if not pages and cleaned:
            pages.append(DocumentPage(
                content=cleaned,
                metadata={
                    **base_meta,
                    "page": 1,
                    "total_pages": 1,
                    "section": "General",
                }
            ))

        for p in pages:
            p.metadata["total_pages"] = len(pages)

        return pages

    @classmethod
    def _load_html(cls, content_bytes: bytes, base_meta: dict) -> List[DocumentPage]:
        raw_html = content_bytes.decode("utf-8", errors="replace")
        cleaned = extract_text_from_html(raw_html)
        if not cleaned:
            return []
        meta = {
            **base_meta,
            "page": 1,
            "total_pages": 1,
            "section": "Web Document",
        }
        return [DocumentPage(content=cleaned, metadata=meta)]
