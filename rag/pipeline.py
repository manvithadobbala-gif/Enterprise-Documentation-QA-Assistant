from typing import List, Dict, Any, Optional
from rag.retriever import DocumentRetriever
from rag.generator import GroqGenerator
from rag.validator import EvidenceValidator, ValidationResult
from rag.conflict_detector import ConflictDetector, ConflictResult


class RAGPipeline:
    """
    Orchestrates semantic retrieval, evidence validation, conflict detection,
    and grounded answer generation into a unified enterprise RAG workflow.
    """

    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        generator: Optional[GroqGenerator] = None,
        validator: Optional[EvidenceValidator] = None,
        conflict_detector: Optional[ConflictDetector] = None,
        top_k: int = 4,
        similarity_threshold: float = 0.25
    ):
        self.retriever = retriever or DocumentRetriever()
        self.generator = generator or GroqGenerator()
        self.validator = validator or EvidenceValidator(generator=self.generator)
        self.conflict_detector = conflict_detector or ConflictDetector(generator=self.generator)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def answer_question(
        self,
        question: str,
        filter_document: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline with strict adherence to the enterprise response schema.
        """
        warnings: List[str] = []
        clean_q = question.strip() if question else ""

        if not clean_q:
            return {
                "status": "error",
                "answer": "Please provide a valid question.",
                "sources": [],
                "conflict_detected": False,
                "evidence_sufficient": False,
                "warnings": ["Query string was empty."]
            }

        # Step 1: Retrieval
        try:
            chunks = self.retriever.retrieve(
                query=clean_q,
                top_k=self.top_k,
                similarity_threshold=self.similarity_threshold,
                filter_document=filter_document
            )
        except Exception as e:
            return {
                "status": "error",
                "answer": f"Error during retrieval: {str(e)}",
                "sources": [],
                "conflict_detected": False,
                "evidence_sufficient": False,
                "warnings": [str(e)]
            }

        context_str = self.retriever.format_context_for_prompt(chunks)

        # Step 2: Extract structured source metadata
        sources = []
        seen_chunks = set()
        for c in chunks:
            c_id = c.get("chunk_id", "")
            if c_id not in seen_chunks:
                seen_chunks.add(c_id)
                meta = c.get("metadata", {})
                sources.append({
                    "document": meta.get("document", "Unknown"),
                    "page": meta.get("page", 1),
                    "section": meta.get("section", "General"),
                    "chunk_id": c_id,
                    "similarity": c.get("similarity", 0.0)
                })

        # Step 3: Evidence Validation
        val_res: ValidationResult = self.validator.validate(
            question=clean_q,
            retrieved_chunks=chunks,
            context_str=context_str
        )

        if val_res.is_ambiguous:
            clarification = val_res.clarification_question or "Could you please clarify your question?"
            return {
                "status": "clarification_needed",
                "answer": clarification,
                "sources": sources,
                "conflict_detected": False,
                "evidence_sufficient": False,
                "warnings": [f"Ambiguous query: {val_res.reason}"]
            }

        if not val_res.is_sufficient:
            if val_res.reason:
                warnings.append(val_res.reason)
            return {
                "status": "insufficient_evidence",
                "answer": "Insufficient information in the uploaded documents.",
                "sources": sources,
                "conflict_detected": False,
                "evidence_sufficient": False,
                "warnings": warnings
            }

        # Step 4: Conflict Detection
        conflict_res: ConflictResult = self.conflict_detector.detect_conflicts(
            retrieved_chunks=chunks,
            context_str=context_str
        )

        if conflict_res.conflict_detected:
            warnings.append(f"Contradiction detected across sources: {conflict_res.conflict_summary}")

        # Step 5: Answer Generation
        try:
            if self.generator and self.generator.is_configured():
                answer = self.generator.generate_answer(
                    question=clean_q,
                    context=context_str
                )
            else:
                # Grounded multi-evidence synthesis when running offline without LLM API key
                bullets = []
                for idx, c in enumerate(chunks[:3], 1):
                    c_text = c.get("text", "").strip()
                    c_meta = c.get("metadata", {})
                    c_page = c_meta.get("page", 1)
                    c_sec = c_meta.get("section", f"Page {c_page}")
                    bullets.append(f"{idx}. **{c_sec} (Page {c_page}):** {c_text}")

                doc_name = chunks[0].get("metadata", {}).get("document", "Document")
                answer = (
                    f"Based on the uploaded documentation in **{doc_name}** regarding **\"{clean_q}\"**:\n\n"
                    + "\n\n".join(bullets)
                )
        except Exception as e:
            return {
                "status": "error",
                "answer": f"Error during answer generation: {str(e)}",
                "sources": sources,
                "conflict_detected": conflict_res.conflict_detected,
                "evidence_sufficient": True,
                "warnings": [str(e)]
            }

        # Step 6: Post-generation check for fallback response
        if "insufficient information in the uploaded documents" in answer.lower():
            return {
                "status": "insufficient_evidence",
                "answer": "Insufficient information in the uploaded documents.",
                "sources": sources,
                "conflict_detected": conflict_res.conflict_detected,
                "evidence_sufficient": False,
                "warnings": warnings
            }

        # Step 7: Final Status determination
        status = "conflict_detected" if conflict_res.conflict_detected else "answered"

        return {
            "status": status,
            "answer": answer,
            "sources": sources,
            "conflict_detected": conflict_res.conflict_detected,
            "evidence_sufficient": True,
            "warnings": warnings
        }
