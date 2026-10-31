import math
import shutil
import tempfile
import pytest

from ingestion.text_splitter import DocumentChunk
from ingestion.embeddings import EmbeddingManager
from ingestion.vector_store import VectorStoreManager
from rag.retriever import DocumentRetriever


@pytest.fixture
def temp_chroma_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_embedding_manager():
    manager = EmbeddingManager()
    vector = manager.embed_query("Security and Compliance Policy")
    assert len(vector) == 384

    # Check vector normalization: sum(v^2) should be approximately 1.0
    norm = sum(x * x for x in vector)
    assert math.isclose(norm, 1.0, rel_tol=1e-3)


def test_vector_store_and_deduplication(temp_chroma_dir):
    vs = VectorStoreManager(persist_directory=temp_chroma_dir, collection_name="test_collection")

    chunks = [
        DocumentChunk(
            chunk_id="doc1_p1_c1",
            text="Employees must enroll in medical insurance within 30 days of joining.",
            metadata={"document": "benefits.pdf", "page": 1, "section": "Health", "file_hash": "hash_doc1"}
        ),
        DocumentChunk(
            chunk_id="doc1_p1_c2",
            text="Dental insurance covers routine cleanings twice per year.",
            metadata={"document": "benefits.pdf", "page": 1, "section": "Dental", "file_hash": "hash_doc1"}
        ),
    ]

    added = vs.add_chunks(chunks)
    assert added == 2
    assert vs.count() == 2

    # Deduplication check: adding same chunks should be ignored
    added_again = vs.add_chunks(chunks)
    assert added_again == 0
    assert vs.count() == 2

    # Verify document listing
    doc_summary = vs.get_indexed_documents()
    assert len(doc_summary) == 1
    assert doc_summary[0]["document"] == "benefits.pdf"
    assert doc_summary[0]["chunk_count"] == 2


def test_retriever_semantic_search_and_context_formatting(temp_chroma_dir):
    vs = VectorStoreManager(persist_directory=temp_chroma_dir, collection_name="test_collection")

    chunks = [
        DocumentChunk(
            chunk_id="sec_p1_c1",
            text="Passwords must be at least 16 characters long and updated every 90 days.",
            metadata={"document": "security.pdf", "page": 1, "section": "Passwords", "file_hash": "sec_hash"}
        ),
        DocumentChunk(
            chunk_id="sec_p2_c1",
            text="Remote access requires VPN connection with hardware token authentication.",
            metadata={"document": "security.pdf", "page": 2, "section": "VPN", "file_hash": "sec_hash"}
        ),
        DocumentChunk(
            chunk_id="hr_p1_c1",
            text="Standard paid time off is 25 days per calendar year.",
            metadata={"document": "leave.pdf", "page": 1, "section": "PTO", "file_hash": "hr_hash"}
        )
    ]
    vs.add_chunks(chunks)

    retriever = DocumentRetriever(vector_store=vs)
    results = retriever.retrieve("What are the password requirements?", top_k=2)

    assert len(results) > 0
    top_result = results[0]
    assert "Passwords must be at least 16 characters" in top_result["text"]
    assert top_result["metadata"]["document"] == "security.pdf"
    assert top_result["metadata"]["page"] == 1
    assert top_result["similarity"] > 0.4

    # Format context test
    context_str = retriever.format_context_for_prompt(results)
    assert "[Evidence #1 | Document: security.pdf | Page: 1 | Section: Passwords" in context_str
    assert "Passwords must be at least 16 characters" in context_str
