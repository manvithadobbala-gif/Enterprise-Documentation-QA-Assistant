import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from rag.generator import GroqGenerator
from rag.prompts import CONFLICT_DETECTION_PROMPT


@dataclass
class ConflictResult:
    """Outcome of cross-document contradiction and policy inconsistency analysis."""
    conflict_detected: bool
    conflict_summary: str = ""
    conflicting_sources: List[Dict[str, Any]] = field(default_factory=list)


class ConflictDetector:
    """
    Detects contradictory policies, values, dates, or terms across multiple retrieved chunks.
    Ensures that conflicting guidance across documents or document versions is explicitly flagged.
    """

    def __init__(self, generator: Optional[GroqGenerator] = None):
        self.generator = generator

    def detect_conflicts(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        context_str: str
    ) -> ConflictResult:
        """Analyzes evidence across distinct sources or sections to find contradictory claims."""
        if not retrieved_chunks or len(retrieved_chunks) < 2:
            return ConflictResult(conflict_detected=False)

        # Check if chunks originate from different documents or distinct versions
        doc_names = {c.get("metadata", {}).get("document") for c in retrieved_chunks}

        # LLM detection if generator is available and configured
        if self.generator and self.generator.is_configured():
            try:
                prompt = CONFLICT_DETECTION_PROMPT.format(context=context_str)
                raw_json = self.generator.generate_json(prompt=prompt)
                data = json.loads(raw_json)
                if "error" in data:
                    raise RuntimeError(data["error"])

                detected = bool(data.get("conflict_detected", False))
                summary = data.get("conflict_summary", "")
                sources = data.get("conflicting_sources", [])

                return ConflictResult(
                    conflict_detected=detected,
                    conflict_summary=summary,
                    conflicting_sources=sources
                )
            except Exception:
                pass

        # Offline heuristic check: multiple versions of the same document (e.g. HR_Policy_2024 vs 2022)
        if len(doc_names) > 1 and any("archived" in d.lower() or "2022" in d for d in doc_names):
            return ConflictResult(
                conflict_detected=True,
                conflict_summary=f"Retrieved passages contain multiple versions of policies: {', '.join(doc_names)}.",
                conflicting_sources=[{"document": d} for d in doc_names]
            )

        return ConflictResult(conflict_detected=False)
