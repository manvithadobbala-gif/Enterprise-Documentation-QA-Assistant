# Enterprise Documentation Q&A Assistant

An enterprise-grade, retrieval-augmented generation (RAG) assistant built from scratch in Python. It allows employees to upload heterogeneous corporate documents (PDF, TXT, Markdown, HTML), index them into ChromaDB with Sentence Transformers, and query them using Groq LLM inference with strict factual grounding, citation tracing, contradiction detection, and hallucination safeguards.

---

## Architecture Overview

```
enterprise_doc_assistant/
│
├── app.py                      # Main Streamlit chat UI
├── requirements.txt            # Project dependencies
├── .env.example                # Template for environment variables
├── .gitignore                  # Git ignore rules
│
├── ingestion/
│   ├── document_loader.py      # PDF (PyMuPDF), TXT, Markdown, HTML (BS4) loaders + SHA256 deduplication
│   ├── text_cleaner.py         # Unicode NFKC normalization, whitespace & control character cleanup
│   ├── text_splitter.py        # Recursive semantic chunking with overlap & metadata preservation
│   ├── embeddings.py           # Sentence-Transformers embedding manager with offline caching
│   └── vector_store.py         # Persistent ChromaDB client, cosine similarity, index management
│
├── rag/
│   ├── prompts.py              # System prompts enforcing passive reference data & injection defense
│   ├── retriever.py            # Top-k semantic retriever & prompt context formatter
│   ├── generator.py            # Groq API client with error handling (supports Llama 3.3 70B, etc.)
│   ├── validator.py            # Evidence sufficiency & ambiguity evaluation (prevents guessing)
│   ├── conflict_detector.py    # Cross-document & multi-version contradiction analysis
│   └── pipeline.py             # Master RAG pipeline orchestrator with strict JSON response schema
│
├── frontend/
│   └── components.py           # Streamlit UI widgets (status badges, citation accordions, warnings)
│
├── evaluation/
│   ├── test_questions.json     # Benchmark evaluation dataset (answerable, unanswerable, conflict, ambiguous)
│   └── evaluate.py             # Evaluation benchmark script (retrieval hit rate, fallback, conflict accuracy)
│
├── tests/
│   ├── test_ingestion.py       # 6 unit tests for text cleaning, loaders, and recursive splitter
│   ├── test_retriever.py       # 3 unit tests for embeddings, ChromaDB deduplication, and retriever
│   ├── test_validator.py       # 4 unit tests for evidence validation and conflict detection
│   └── test_pipeline.py        # 5 unit tests for RAG pipeline orchestration and response formatting
│
└── data/
    └── uploads/                # Sample enterprise test policies (HR, Travel, IT Security)
```

---

## Getting Started in VS Code on Windows

### 1. Activate the Virtual Environment
Open PowerShell inside VS Code:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
*(You can get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys). Alternatively, you can enter the API key directly in the Streamlit web interface sidebar!)*

---

## Running the Application

Launch the Streamlit web application:
```powershell
.\.venv\Scripts\streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Key Features in the Interface:
1. **Interactive Document Upload**: Drag and drop PDF, TXT, MD, or HTML files in the sidebar and click **Process & Ingest**.
2. **Real-Time Factual Badges**: Every response displays an audit badge:
   - 🟢 `Answered with Evidence`: Grounded factually with citations.
   - 🟠 `Insufficient Evidence`: Clear abstention when knowledge base lacks evidence.
   - 🔵 `Clarification Needed`: Prompts user when query is ambiguous.
   - 🔴 `Contradiction Detected`: Warns if distinct documents/versions present conflicting policies.
3. **Expandable Source Citations**: Direct inspection of the document name, page number, section, chunk ID, and semantic match score.

---

## Running Tests & Verification

### Run the Unit Test Suite (18 tests)
```powershell
.\.venv\Scripts\python.exe -m pytest tests -v -p no:cacheprovider
```

### Run Benchmark Evaluation Script
Evaluates retrieval precision, fallback behavior, conflict detection, and clarification prompts against the sample policies in `data/uploads/`:
```powershell
.\.venv\Scripts\python.exe evaluation\evaluate.py
```
*(Achieves 100% on retrieval relevance, 100% on fallback abstention, 100% on conflict detection, and 100% on clarification triggering).*
