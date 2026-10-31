import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Ensure modules are on path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

rag_module_path = project_root / "enterprise_rag_module"
if str(rag_module_path) not in sys.path:
    sys.path.insert(0, str(rag_module_path))

from ingestion.document_loader import DocumentLoader
from ingestion.text_splitter import RecursiveTextSplitter
from ingestion.embeddings import EmbeddingManager
from ingestion.vector_store import VectorStoreManager
from rag.retriever import DocumentRetriever
from rag.generator import GroqGenerator
from rag.validator import EvidenceValidator
from rag.conflict_detector import ConflictDetector
from rag.pipeline import RAGPipeline
from frontend.components import (
    render_response_header,
    render_citation_cards,
    render_scenario_alert
)

load_dotenv()

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom card styling
st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
    }
    .user-query-card {
        padding: 4px 8px;
        font-weight: 600;
        color: #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = {}

if "embedding_manager" not in st.session_state:
    st.session_state.embedding_manager = EmbeddingManager()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager(
        embedding_manager=st.session_state.embedding_manager
    )

if "splitter" not in st.session_state:
    st.session_state.splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=100)

if "pipeline" not in st.session_state:
    retriever = DocumentRetriever(vector_store=st.session_state.vector_store)
    generator = GroqGenerator()
    validator = EvidenceValidator(generator=generator)
    conflict_detector = ConflictDetector(generator=generator)
    st.session_state.pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        validator=validator,
        conflict_detector=conflict_detector,
        top_k=4,
        similarity_threshold=0.15
    )


# =========================================================
# SIDEBAR: Document Ingestion & Management
# =========================================================
with st.sidebar:
    st.subheader("📁 Upload Enterprise Documents")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "html", "htm", "md", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload PDF books, reports, policies, Markdown, or text documents."
    )

    # Automatic ingestion on file upload
    if uploaded_files:
        current_names = {f.name for f in uploaded_files}
        # Check which uploaded files need indexing
        for file_obj in uploaded_files:
            if file_obj.name not in st.session_state.indexed_files:
                with st.spinner(f"Reading & indexing {file_obj.name}..."):
                    try:
                        pages = DocumentLoader.load_file(file_obj, filename=file_obj.name)
                        if not pages:
                            st.error(
                                f"⚠️ **{file_obj.name} contains no selectable digital text.**\n\n"
                                f"This file appears to be a scanned photocopy/image PDF. "
                                f"Please upload a digital PDF (where you can highlight text with your cursor), TXT, or Markdown file."
                            )
                        else:
                            chunks = st.session_state.splitter.split_documents(pages)
                            added = st.session_state.vector_store.add_chunks(chunks)
                            st.session_state.indexed_files[file_obj.name] = {
                                "pages": len(pages),
                                "chunks": len(chunks),
                                "size_kb": file_obj.size // 1024
                            }
                    except Exception as e:
                        st.error(f"Error loading {file_obj.name}: {e}")

    doc_count = len(st.session_state.indexed_files)
    st.subheader(f"📑 Active Documents ({doc_count})")

    if not st.session_state.indexed_files:
        st.info("ℹ️ No documents uploaded yet. Upload a PDF, HTML, MD, or TXT file above.")
    else:
        st.success(f"✓ {doc_count} document{'s' if doc_count > 1 else ''} ready for Q&A.")
        for fname, info in st.session_state.indexed_files.items():
            st.markdown(f"📄 **{fname}**")
            st.caption(f"{info['pages']} pages &nbsp;•&nbsp; {info['chunks']} passages &nbsp;•&nbsp; {info['size_kb']} KB")
            st.divider()

    st.divider()

    # Clear options
    if st.button("🗑️ Clear Documents & Knowledge Base", width="stretch"):
        st.session_state.vector_store.reset()
        st.session_state.indexed_files = {}
        st.session_state.messages = []
        st.success("Knowledge base cleared.")
        st.rerun()

    if st.button("💬 Clear Chat History", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# MAIN AREA: Responses & Citations
# =========================================================
st.markdown("## 📋 Responses & Citations")

# If no documents uploaded yet, display friendly onboarding guidance
if not st.session_state.indexed_files:
    st.info(
        "👋 **Welcome! Please upload a document in the sidebar to get started.**\n\n"
        "Once uploaded, the system will automatically extract and index the content. "
        "You can then ask any question, and the assistant will answer directly from your files with exact page citations."
    )

# Render interaction history
for item in st.session_state.messages:
    q = item.get("question", "")
    ans = item.get("answer", "")
    sources = item.get("sources", [])
    status = item.get("status", "answered")
    warnings = item.get("warnings", [])

    # 1. User Query Card
    with st.container(border=True):
        st.markdown(f"👤 &nbsp; **{q}**")

    # 2. Generated Enterprise Response Card
    with st.container(border=True):
        render_response_header(status)
        st.markdown(ans)
        render_scenario_alert(status, warnings)

        # 3. Source Citations & Excerpts
        if sources:
            render_citation_cards(sources)

    st.write("")


# =========================================================
# CHAT INPUT: Live Question Answering
# =========================================================
if user_query := st.chat_input("Ask a question about your uploaded documents..."):
    if not st.session_state.indexed_files:
        st.warning("Please upload at least one document in the sidebar first.")
    else:
        with st.spinner("Searching document passages and preparing verified answer..."):
            # Execute RAG Pipeline on the uploaded files
            res = st.session_state.pipeline.answer_question(user_query)

            new_message = {
                "question": user_query,
                "status": res["status"],
                "answer": res["answer"],
                "sources": res.get("sources", []),
                "warnings": res.get("warnings", [])
            }

            st.session_state.messages.append(new_message)
            st.rerun()
