import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from rag.generator import GroqGenerator
from rag.prompts import VALIDATION_PROMPT


@dataclass
class ValidationResult:
    """Outcome of evidence sufficiency and ambiguity evaluation."""
    is_sufficient: bool
    is_ambiguous: bool
    clarification_question: Optional[str] = None
    reason: str = ""


class EvidenceValidator:
    """
    Validates whether retrieved document evidence contains factual answers
    for the user's question, and detects ambiguous queries needing clarification.
    """

    def __init__(self, generator: Optional[GroqGenerator] = None):
        self.generator = generator

    def validate(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        context_str: str
    ) -> ValidationResult:
        """
        Evaluates evidence sufficiency and question ambiguity.
        Uses layered heuristics + LLM validation.
        """
        # Heuristic 1: No retrieved chunks
        if not retrieved_chunks or not context_str.strip() or context_str.strip() == "No relevant document evidence found.":
            return ValidationResult(
                is_sufficient=False,
                is_ambiguous=False,
                clarification_question=None,
                reason="No matching document passages were found in the uploaded files."
            )

        # LLM-based verification if generator is configured
        if self.generator and self.generator.is_configured():
            try:
                prompt = VALIDATION_PROMPT.format(context=context_str, question=question)
                raw_json = self.generator.generate_json(prompt=prompt)
                data = json.loads(raw_json)
                if "error" in data:
                    raise RuntimeError(data["error"])

                is_suff = bool(data.get("sufficient", False))
                is_ambi = bool(data.get("ambiguous", False))
                clarif = data.get("clarification_question")
                reason = data.get("reason", "")

                return ValidationResult(
                    is_sufficient=is_suff,
                    is_ambiguous=is_ambi,
                    clarification_question=clarif if is_ambi else None,
                    reason=reason
                )
            except Exception:
                pass

        # Offline heuristic check based on top similarity score
        top_sim = max((c.get("similarity", 0.0) for c in retrieved_chunks), default=0.0)
        if top_sim < 0.40:
            return ValidationResult(
                is_sufficient=False,
                is_ambiguous=False,
                clarification_question=None,
                reason=f"Top evidence relevance ({top_sim:.1%}) is below factual threshold (40%)."
            )

        return ValidationResult(
            is_sufficient=True,
            is_ambiguous=False,
            clarification_question=None,
            reason=f"Heuristic similarity check satisfied ({top_sim:.1%})."
        )
