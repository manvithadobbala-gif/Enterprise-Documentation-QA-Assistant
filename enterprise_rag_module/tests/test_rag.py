import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

# Ensure enterprise_rag_module is on sys.path
module_root = Path(__file__).resolve().parent.parent
if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))

from rag.retriever import DocumentRetriever, MockVectorStore
from rag.validator import EvidenceValidator, ValidationResult
from rag.conflict_detector import ConflictDetector, ConflictResult
from rag.pipeline import RAGPipeline


def test_retriever_mock_store():
    retriever = DocumentRetriever()
    results = retriever.retrieve("meal per diem", top_k=2)
    assert len(results) > 0
    assert "$75" in results[0]["text"]
    assert results[0]["metadata"]["document"] == "Travel_Expense_Policy.txt"


def test_validator_empty_evidence():
    validator = EvidenceValidator(generator=None)
    res = validator.validate("What is the dental plan?", [], "")
    assert res.is_sufficient is False
    assert res.is_ambiguous is False


def test_validator_ambiguous_query():
    validator = EvidenceValidator(generator=None)
    mock_chunks = [{"chunk_id": "c1", "similarity": 0.8}]
    res = validator.validate("leave", mock_chunks, "Leave context")
    assert res.is_ambiguous is True
    assert "clarify" in res.clarification_question.lower() or "elaborate" in res.clarification_question.lower()


def test_conflict_detector_version_mismatch():
    detector = ConflictDetector(generator=None)
    chunks = [
        {"metadata": {"document": "HR_Policy_2024.md"}},
        {"metadata": {"document": "HR_Policy_2022_Archived.md"}}
    ]
    res = detector.detect_conflicts(chunks, "context")
    assert res.conflict_detected is True
    assert "multiple versions" in res.conflict_summary.lower()


def test_pipeline_answered():
    pipeline = RAGPipeline()
    res = pipeline.answer_question("What is the maximum allowable per diem for daily meal expenses?")
    assert res["status"] == "answered"
    assert res["evidence_sufficient"] is True
    assert len(res["sources"]) > 0
    assert "75" in res["answer"]


def test_pipeline_unanswerable_fallback():
    pipeline = RAGPipeline()
    res = pipeline.answer_question("Can I receive corporate bonuses in Bitcoin?")
    assert res["status"] == "insufficient_evidence"
    assert res["evidence_sufficient"] is False
    assert res["answer"] == "Insufficient information in the uploaded documents."


def test_pipeline_ambiguous():
    pipeline = RAGPipeline()
    res = pipeline.answer_question("leave")
    assert res["status"] == "clarification_needed"
    assert res["evidence_sufficient"] is False


def test_pipeline_with_custom_vector_store():
    # Demonstrates how Member 1 can inject their live ChromaDB vector store
    custom_vs = MagicMock()
    custom_vs.search.return_value = [
        {
            "chunk_id": "custom_c1",
            "text": "Cybersecurity training is due every October.",
            "metadata": {"document": "Sec_Training.pdf", "page": 3, "section": "Compliance"},
            "similarity": 0.85
        }
    ]
    custom_retriever = DocumentRetriever(vector_store=custom_vs)
    pipeline = RAGPipeline(retriever=custom_retriever)
    res = pipeline.answer_question("When is cybersecurity training due?")

    assert res["status"] == "answered"
    assert "Cybersecurity training is due" in res["answer"]
    assert res["sources"][0]["document"] == "Sec_Training.pdf"
    assert res["sources"][0]["page"] == 3
