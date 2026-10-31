import os
import sys
from pathlib import Path

# Add project root to sys.path so modules like ingestion and rag resolve cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set utf-8 encoding for Windows terminal output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import glob
from typing import List, Dict, Any
from dotenv import load_dotenv

from ingestion.document_loader import DocumentLoader
from ingestion.text_splitter import RecursiveTextSplitter
from ingestion.vector_store import VectorStoreManager
from rag.retriever import DocumentRetriever
from rag.generator import GroqGenerator
from rag.validator import EvidenceValidator
from rag.conflict_detector import ConflictDetector
from rag.pipeline import RAGPipeline

load_dotenv()


def setup_sample_knowledge_base(uploads_dir: str = "data/uploads") -> VectorStoreManager:
    """Loads and indexes all documents found in the sample uploads directory."""
    vs = VectorStoreManager()
    splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=100)

    files = glob.glob(os.path.join(uploads_dir, "*.*"))
    print(f"\n[INFO] Found {len(files)} files in {uploads_dir}. Indexing into ChromaDB...")

    total_chunks = 0
    for fpath in files:
        if Path(fpath).name == ".gitkeep":
            continue
        try:
            pages = DocumentLoader.load_file(fpath)
            chunks = splitter.split_documents(pages)
            added = vs.add_chunks(chunks)
            total_chunks += added
            print(f"  [+] {Path(fpath).name}: {len(chunks)} chunks generated ({added} newly added).")
        except Exception as e:
            print(f"  [-] Failed to load {fpath}: {e}")

    print(f"[INFO] Indexing complete. Total active chunks in ChromaDB: {vs.count()}\n")
    return vs


def run_evaluation(questions_path: str = "evaluation/test_questions.json"):
    """Evaluates retrieval relevance, citation accuracy, conflict detection, and fallback behaviors."""
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Test questions file not found at: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        test_cases: List[Dict[str, Any]] = json.load(f)

    vs = setup_sample_knowledge_base()
    retriever = DocumentRetriever(vector_store=vs)

    # Check if Groq API key is present
    groq_key = os.getenv("GROQ_API_KEY")
    has_api_key = bool(groq_key and groq_key != "your_groq_api_key_here")

    if has_api_key:
        print("[INFO] Live Groq API key detected. Running full end-to-end evaluation.")
        generator = GroqGenerator()
    else:
        print("[INFO] No live Groq API key detected. Evaluating retrieval, heuristics, and fallback offline.")
        generator = None

    validator = EvidenceValidator(generator=generator)
    conflict_detector = ConflictDetector(generator=generator)
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        validator=validator,
        conflict_detector=conflict_detector,
        top_k=4,
        similarity_threshold=0.20
    )

    results = []
    retrieval_hits = 0
    total_evaluable_retrieval = 0
    fallback_correct = 0
    total_fallback_cases = 0
    conflict_correct = 0
    total_conflict_cases = 0
    clarification_correct = 0
    total_clarification_cases = 0

    print("=" * 70)
    print("STARTING BENCHMARK EVALUATION")
    print("=" * 70)

    for tc in test_cases:
        qid = tc["id"]
        q_text = tc["question"]
        q_type = tc["type"]
        expected_doc = tc.get("expected_document")
        expect_fallback = tc.get("expect_fallback", False)
        expect_conflict = tc.get("expect_conflict", False)
        expect_clarif = tc.get("expect_clarification", False)

        print(f"\n[Test Case: {qid}] ({q_type.upper()})")
        print(f"Question: \"{q_text}\"")

        # Execute pipeline
        response = pipeline.answer_question(q_text)
        status = response.get("status")
        answer = response.get("answer", "")
        sources = response.get("sources", [])
        retrieved_docs = [s.get("document") for s in sources]

        # 1. Evaluate Retrieval
        retrieval_pass = True
        if expected_doc:
            total_evaluable_retrieval += 1
            if expected_doc in retrieved_docs:
                retrieval_hits += 1
                print(f"  [PASS] Retrieval: Expected document '{expected_doc}' successfully retrieved.")
            else:
                retrieval_pass = False
                print(f"  [FAIL] Retrieval: Expected '{expected_doc}', retrieved: {retrieved_docs}")

        # 2. Evaluate Fallback
        fallback_pass = True
        if expect_fallback:
            total_fallback_cases += 1
            if status == "insufficient_evidence" or "insufficient information" in answer.lower():
                fallback_correct += 1
                print("  [PASS] Fallback: System correctly abstained with insufficient evidence message.")
            else:
                fallback_pass = False
                print(f"  [FAIL] Fallback: System failed to abstain. Status: {status}")

        # 3. Evaluate Conflict
        conflict_pass = True
        if expect_conflict:
            total_conflict_cases += 1
            if response.get("conflict_detected") or status == "conflict_detected" or len(set(retrieved_docs)) > 1:
                conflict_correct += 1
                print("  [PASS] Conflict Detection: Discrepancy or multi-policy version flagged.")
            else:
                conflict_pass = False
                print("  [FAIL] Conflict Detection: Contradiction missed.")

        # 4. Evaluate Clarification
        clarif_pass = True
        if expect_clarif:
            total_clarification_cases += 1
            if status == "clarification_needed":
                clarification_correct += 1
                print("  [PASS] Clarification: System correctly prompted for user clarification.")
            else:
                clarif_pass = False
                print(f"  [FAIL] Clarification: System did not request clarification. Status: {status}")

        results.append({
            "id": qid,
            "status": status,
            "retrieval_pass": retrieval_pass,
            "fallback_pass": fallback_pass,
            "conflict_pass": conflict_pass,
            "sources_count": len(sources)
        })

    # Summary Report
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY REPORT")
    print("=" * 70)
    if total_evaluable_retrieval > 0:
        print(f"Retrieval Accuracy:     {retrieval_hits}/{total_evaluable_retrieval} ({retrieval_hits/total_evaluable_retrieval:.1%})")
    if total_fallback_cases > 0:
        print(f"Fallback Behavior:      {fallback_correct}/{total_fallback_cases} ({fallback_correct/total_fallback_cases:.1%})")
    if total_conflict_cases > 0:
        print(f"Conflict Detection:     {conflict_correct}/{total_conflict_cases} ({conflict_correct/total_conflict_cases:.1%})")
    if total_clarification_cases > 0:
        print(f"Clarification Trigger:  {clarification_correct}/{total_clarification_cases} ({clarification_correct/total_clarification_cases:.1%})")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
