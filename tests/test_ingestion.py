import io
import pytest
import pymupdf

from ingestion.text_cleaner import clean_text, extract_text_from_html
from ingestion.document_loader import DocumentLoader, compute_file_hash
from ingestion.text_splitter import RecursiveTextSplitter, DocumentChunk


def test_clean_text():
    raw = "  Hello \t World!\r\n\r\n\r\nThis is a    test with \u201csmart quotes\u201d.\x00\x08  "
    cleaned = clean_text(raw)
    assert "Hello World!" in cleaned
    assert "This is a test with \"smart quotes\"." in cleaned or "This is a test with “smart quotes”." in cleaned
    assert "\x00" not in cleaned
    assert "\r" not in cleaned


def test_extract_text_from_html():
    html = """
    <html>
        <head><title>Company Policy</title><style>body { color: red; }</style></head>
        <body>
            <script>console.log("secret");</script>
            <h1>Security Policy</h1>
            <p>Employees must use 2FA for all internal portals.</p>
        </body>
    </html>
    """
    cleaned = extract_text_from_html(html)
    assert "Security Policy" in cleaned
    assert "Employees must use 2FA for all internal portals." in cleaned
    assert "console.log" not in cleaned
    assert "color: red" not in cleaned


def test_document_loader_txt_and_hash():
    sample_text = b"Enterprise Data Retention Policy:\nStore customer records for 7 years."
    hash1 = compute_file_hash(sample_text)
    hash2 = compute_file_hash(sample_text)
    assert hash1 == hash2

    pages = DocumentLoader.load_file(io.BytesIO(sample_text), filename="retention.txt")
    assert len(pages) == 1
    assert "Enterprise Data Retention Policy:" in pages[0].content
    assert pages[0].metadata["document"] == "retention.txt"
    assert pages[0].metadata["file_hash"] == hash1


def test_document_loader_markdown():
    sample_md = b"# Section 1\nOverview of perks.\n\n# Section 2\nHealthcare insurance details."
    pages = DocumentLoader.load_file(io.BytesIO(sample_md), filename="benefits.md")
    assert len(pages) == 2
    assert pages[0].metadata["section"] == "Section 1"
    assert pages[1].metadata["section"] == "Section 2"


def test_document_loader_pdf():
    # Dynamically generate a 2-page PDF in memory using pymupdf
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Page 1: Annual leave is 20 days.")
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Page 2: Sick leave is 10 days.")
    pdf_bytes = doc.tobytes()
    doc.close()

    pages = DocumentLoader.load_file(io.BytesIO(pdf_bytes), filename="hr_policy.pdf")
    assert len(pages) == 2
    assert pages[0].metadata["page"] == 1
    assert "Annual leave is 20 days." in pages[0].content
    assert pages[1].metadata["page"] == 2
    assert "Sick leave is 10 days." in pages[1].content


def test_text_splitter():
    text = (
        "Enterprise Documentation Q&A Assistant Architecture.\n\n"
        "Paragraph 1 explains the retrieval augmented generation pipeline in detail. "
        "It includes embeddings, vector store indexation, semantic search, and Groq LLM inference.\n\n"
        "Paragraph 2 discusses conflict detection between various policy versions, "
        "document effective dates, and condition validation."
    )
    from ingestion.document_loader import DocumentPage

    page = DocumentPage(
        content=text,
        metadata={"document": "arch.pdf", "page": 1, "file_hash": "abc12345678"}
    )

    splitter = RecursiveTextSplitter(chunk_size=120, chunk_overlap=30)
    chunks = splitter.split_documents([page])

    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.metadata["document"] == "arch.pdf"
        assert chunk.chunk_id.startswith("abc12345_p1_c")
        assert len(chunk.text) <= 150  # chunks respect bound
