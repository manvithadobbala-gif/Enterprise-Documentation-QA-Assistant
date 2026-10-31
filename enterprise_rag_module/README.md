# Member 2: Enterprise RAG Module

This is the standalone **RAG (Retrieval-Augmented Generation) Module** for **Member 2**.
It contains semantic retrieval, evidence validation, conflict detection, grounded answer generation, and prompt injection defenses.

---

## Folder Layout

```
enterprise_rag_module/
│
├── README.md               # Integration guide for Member 1 & Member 3
├── requirements.txt        # Minimal dependencies for Member 2
├── demo.py                 # Standalone interactive CLI runner
│
├── rag/
│   ├── __init__.py         # Package exports
│   ├── pipeline.py         # Master RAG pipeline orchestrator
│   ├── retriever.py        # Retriever with built-in MockVectorStore + Member 1 adapter
│   ├── generator.py        # Groq LLM client + offline grounded fallback
│   ├── prompts.py          # Grounded QA & injection defense prompts
│   ├── validator.py        # Evidence verification & query ambiguity checker
│   └── conflict_detector.py# Contradiction analyzer across document versions
│
└── tests/
    └── test_rag.py         # 8 automated unit tests for Member 2
```

---

## How to Run & Verify Member 2's Work

### 1. Run Unit Tests (8 Tests)
From the project root:
```powershell
.\.venv\Scripts\python.exe -m pytest enterprise_rag_module/tests -v -p no:cacheprovider
```

### 2. Run the Standalone Interactive Demo
```powershell
.\.venv\Scripts\python.exe enterprise_rag_module/demo.py
```
This lets you test answerable questions, unanswerable questions (zero-hallucination fallback), ambiguous queries, and conflict detection directly in your terminal.

---

## Teammate Integration Guide

### 🤝 For Member 1 (Document Ingestion & ChromaDB):
When Member 1 is ready, they can pass their `VectorStoreManager` directly into Member 2's retriever:

```python
from rag.retriever import DocumentRetriever
from rag.pipeline import RAGPipeline

# Member 1 passes their ChromaDB vector store:
retriever = DocumentRetriever(vector_store=member1_vector_store)
pipeline = RAGPipeline(retriever=retriever)
```

Member 1 only needs to provide a `.search(query, top_k, where)` method that returns:
```python
[
    {
        "chunk_id": "chunk_001",
        "text": "Extracted text here...",
        "metadata": {"document": "Policy.pdf", "page": 1, "section": "Leave"},
        "similarity": 0.85
    }
]
```

---

### 🤝 For Member 3 (Streamlit Frontend / UI):
Member 3 only needs **3 lines of code** in their UI to call Member 2's RAG pipeline:

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
response = pipeline.answer_question(user_question)

# response contains:
# response["status"]              -> "answered", "insufficient_evidence", "clarification_needed", "conflict_detected"
# response["answer"]              -> The generated grounded text
# response["sources"]             -> List of {document, page, section, chunk_id}
# response["conflict_detected"]   -> bool
# response["evidence_sufficient"] -> bool
# response["warnings"]            -> List of warnings
```
