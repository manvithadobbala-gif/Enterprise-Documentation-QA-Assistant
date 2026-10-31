from typing import List, Dict, Any, Optional


class MockVectorStore:
    """
    Built-in sample vector store for Member 2 to test RAG independently,
    even before Member 1 provides the live ChromaDB database.
    """
    def __init__(self):
        self.sample_chunks = [
            {
                "chunk_id": "hr_p1_c1",
                "text": "Full-time employees receive 25 days of paid annual leave per calendar year. Leave requests must be submitted at least 5 business days in advance.",
                "metadata": {"document": "HR_Policy_2024.md", "page": 1, "section": "Annual Leave"},
                "keywords": ["annual leave", "vacation", "pto", "days off", "holiday"]
            },
            {
                "chunk_id": "hr_p1_c2",
                "text": "All eligible primary caregivers receive 12 weeks of fully paid parental leave following birth or legal adoption. Secondary caregivers receive 4 weeks.",
                "metadata": {"document": "HR_Policy_2024.md", "page": 1, "section": "Parental Leave"},
                "keywords": ["parental leave", "maternity", "paternity", "child", "adoption"]
            },
            {
                "chunk_id": "hr_archived_c1",
                "text": "Under the Archived 2022 policy, full-time employees receive 20 days of paid annual leave per year. Unused leave is forfeited.",
                "metadata": {"document": "HR_Policy_2022_Archived.md", "page": 1, "section": "Leave Allowance"},
                "keywords": ["annual leave", "20 days", "archived", "leave"]
            },
            {
                "chunk_id": "travel_p1_c1",
                "text": "The maximum allowable daily meal per diem is $75 per day. Itemized receipts are required for all meals exceeding $25.",
                "metadata": {"document": "Travel_Expense_Policy.txt", "page": 1, "section": "Meals"},
                "keywords": ["meal", "per diem", "$75", "reimbursement", "food", "expense"]
            },
            {
                "chunk_id": "sec_p1_c1",
                "text": "All enterprise passwords must be at least 16 characters in length and include uppercase, lowercase, numbers, and symbols. 2FA is mandatory.",
                "metadata": {"document": "IT_Security_Guidelines.html", "page": 1, "section": "Passwords"},
                "keywords": ["password", "length", "16 characters", "2fa", "authentication", "security"]
            }
        ]

    def search(self, query: str, top_k: int = 4, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        q_lower = query.lower()
        scored = []
        for c in self.sample_chunks:
            # Check filter if provided
            if where and "document" in where and c["metadata"]["document"] != where["document"]:
                continue
            # Calculate keyword similarity score
            score = 0.1
            for kw in c["keywords"]:
                if kw in q_lower:
                    score += 0.35
            for word in q_lower.split():
                if len(word) > 3 and word in c["text"].lower():
                    score += 0.15
            score = min(0.95, score)
            if score > 0.20:
                scored.append({
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "metadata": c["metadata"],
                    "similarity": round(score, 4)
                })
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]


class DocumentRetriever:
    """
    Handles semantic query retrieval and prompt context formatting.
    Accepts Member 1's VectorStoreManager or defaults to the built-in MockVectorStore.
    """

    def __init__(self, vector_store: Optional[Any] = None):
        self.vector_store = vector_store or MockVectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.25,
        filter_document: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves top_k chunks matching the query, filtered by similarity threshold."""
        if not query or not query.strip():
            return []

        where_clause = {"document": filter_document} if filter_document else None

        raw_results = self.vector_store.search(
            query=query,
            top_k=top_k,
            where=where_clause
        )

        filtered = []
        for r in raw_results:
            if r.get("similarity", 0.0) >= similarity_threshold:
                filtered.append(r)

        return filtered

    def format_context_for_prompt(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into a standardized evidence context block with citations."""
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
