import pytest
from rag.validator import EvidenceValidator, ValidationResult
from rag.conflict_detector import ConflictDetector, ConflictResult


def test_validator_empty_evidence():
    validator = EvidenceValidator(generator=None)
    result = validator.validate("What is the travel reimbursement rate?", [], "")
    assert result.is_sufficient is False
    assert result.is_ambiguous is False
    assert "No document chunks" in result.reason


def test_validator_ambiguous_single_word_query():
    validator = EvidenceValidator(generator=None)
    mock_chunks = [{"chunk_id": "c1", "text": "Policy rules", "similarity": 0.8}]
    result = validator.validate("leave", mock_chunks, "Policy rules")
    assert result.is_ambiguous is True
    assert result.clarification_question is not None


def test_validator_low_similarity():
    validator = EvidenceValidator(generator=None)
    mock_chunks = [{"chunk_id": "c1", "text": "Unrelated topic", "similarity": 0.1}]
    result = validator.validate("What is the sick leave policy?", mock_chunks, "Unrelated topic")
    assert result.is_sufficient is False
    assert "below" in result.reason.lower()


def test_conflict_detector_insufficient_chunks():
    detector = ConflictDetector(generator=None)
    # 0 or 1 chunk cannot form a cross-document contradiction
    result_empty = detector.detect_conflicts([], "")
    assert result_empty.conflict_detected is False

    result_single = detector.detect_conflicts([{"chunk_id": "c1", "text": "Leave is 20 days"}], "Leave is 20 days")
    assert result_single.conflict_detected is False
