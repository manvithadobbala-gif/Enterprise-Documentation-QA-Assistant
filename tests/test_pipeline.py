import pytest
from unittest.mock import MagicMock
from rag.pipeline import RAGPipeline
from rag.retriever import DocumentRetriever
from rag.generator import GroqGenerator
from rag.validator import EvidenceValidator, ValidationResult
from rag.conflict_detector import ConflictDetector, ConflictResult


def test_pipeline_empty_query():
    pipeline = RAGPipeline(
        retriever=MagicMock(),
        generator=MagicMock(),
        validator=MagicMock(),
        conflict_detector=MagicMock()
    )
    res = pipeline.answer_question("")
    assert res["status"] == "error"
    assert "valid question" in res["answer"]
    assert res["evidence_sufficient"] is False


def test_pipeline_no_evidence_fallback():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_retriever.format_context_for_prompt.return_value = "No relevant document evidence found."

    mock_validator = MagicMock()
    mock_validator.validate.return_value = ValidationResult(
        is_sufficient=False,
        is_ambiguous=False,
        reason="No chunks retrieved"
    )

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=MagicMock(),
        validator=mock_validator,
        conflict_detector=MagicMock()
    )

    res = pipeline.answer_question("What is the quantum encryption protocol?")
    assert res["status"] == "insufficient_evidence"
    assert res["answer"] == "Insufficient information in the uploaded documents."
    assert res["evidence_sufficient"] is False


def test_pipeline_ambiguous_question():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [{"chunk_id": "c1", "similarity": 0.5}]
    mock_retriever.format_context_for_prompt.return_value = "Sample context"

    mock_validator = MagicMock()
    mock_validator.validate.return_value = ValidationResult(
        is_sufficient=False,
        is_ambiguous=True,
        clarification_question="Did you mean annual leave or medical leave?"
    )

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=MagicMock(),
        validator=mock_validator,
        conflict_detector=MagicMock()
    )

    res = pipeline.answer_question("leave")
    assert res["status"] == "clarification_needed"
    assert "annual leave or medical leave" in res["answer"]


def test_pipeline_answered_successfully():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "chunk_id": "hr_p1_c1",
            "text": "Annual leave entitlement is 20 days per year.",
            "metadata": {"document": "hr_policy.pdf", "page": 1, "section": "Leave"},
            "similarity": 0.88
        }
    ]
    mock_retriever.format_context_for_prompt.return_value = "[Evidence #1 | Document: hr_policy.pdf | Page: 1] Annual leave is 20 days."

    mock_validator = MagicMock()
    mock_validator.validate.return_value = ValidationResult(is_sufficient=True, is_ambiguous=False)

    mock_conflict = MagicMock()
    mock_conflict.detect_conflicts.return_value = ConflictResult(conflict_detected=False)

    mock_generator = MagicMock()
    mock_generator.generate_answer.return_value = "Employees receive 20 days of annual leave per year. [Doc: hr_policy.pdf, Page: 1, Section: Leave, ChunkID: hr_p1_c1]"

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        validator=mock_validator,
        conflict_detector=mock_conflict
    )

    res = pipeline.answer_question("How many days of annual leave do employees receive?")
    assert res["status"] == "answered"
    assert res["conflict_detected"] is False
    assert res["evidence_sufficient"] is True
    assert len(res["sources"]) == 1
    assert res["sources"][0]["document"] == "hr_policy.pdf"
    assert res["sources"][0]["page"] == 1
    assert res["sources"][0]["chunk_id"] == "hr_p1_c1"


def test_pipeline_conflict_detected():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"chunk_id": "c1", "text": "Expense limit is $50", "metadata": {"document": "policy2023.pdf", "page": 2, "section": "Meals"}, "similarity": 0.8},
        {"chunk_id": "c2", "text": "Expense limit is $75", "metadata": {"document": "policy2024.pdf", "page": 3, "section": "Meals"}, "similarity": 0.82}
    ]
    mock_retriever.format_context_for_prompt.return_value = "Multi-document meal expense rules."

    mock_validator = MagicMock()
    mock_validator.validate.return_value = ValidationResult(is_sufficient=True, is_ambiguous=False)

    mock_conflict = MagicMock()
    mock_conflict.detect_conflicts.return_value = ConflictResult(
        conflict_detected=True,
        conflict_summary="policy2023 states $50 meal limit while policy2024 states $75."
    )

    mock_generator = MagicMock()
    mock_generator.generate_answer.return_value = "Discrepancy: policy2023 specifies $50 whereas policy2024 specifies $75."

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        validator=mock_validator,
        conflict_detector=mock_conflict
    )

    res = pipeline.answer_question("What is the meal expense limit?")
    assert res["status"] == "conflict_detected"
    assert res["conflict_detected"] is True
    assert len(res["warnings"]) > 0
    assert "Contradiction detected" in res["warnings"][0]
