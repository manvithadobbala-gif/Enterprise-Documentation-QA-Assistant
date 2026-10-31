from typing import List, Dict, Any, Optional
from ingestion.vector_store import VectorStoreManager


class DocumentRetriever:
    """
    Handles semantic query retrieval from ChromaDB, formatting evidence blocks,
    and structuring metadata for downstream LLM generation and validation.
    """

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.0,
        filter_document: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks matching the query.
        Applies optional similarity filtering and document filtering.
        """
        if not query or not query.strip():
            return []

        where_clause = None
        if filter_document:
            where_clause = {"document": filter_document}

        raw_results = self.vector_store.search(
            query=query,
            top_k=top_k,
            where=where_clause
        )

        filtered_results = []
        for r in raw_results:
            if r.get("similarity", 0.0) >= similarity_threshold:
                filtered_results.append(r)

        return filtered_results

    def format_context_for_prompt(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks into a standardized, numbered evidence context block.
        Includes document name, page number, section, and chunk_id to enable accurate citation.
        """
        if not retrieved_chunks:
            return "No relevant document evidence found."

        context_blocks = []
        for i, item in enumerate(retrieved_chunks, 1):
            meta = item.get("metadata", {})
            doc = meta.get("document", "Unknown Document")
            page = meta.get("page", 1)
            section = meta.get("section", "General")
            chunk_id = item.get("chunk_id", f"chunk_{i}")
            text = item.get("text", "").strip()

            block = (
                f"[Evidence #{i} | Document: {doc} | Page: {page} | Section: {section} | ChunkID: {chunk_id}]\n"
                f"{text}"
            )
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)
