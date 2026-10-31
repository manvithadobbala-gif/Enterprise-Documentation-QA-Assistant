import os
import sys
import json
from pathlib import Path

# Add module root to sys.path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from rag.pipeline import RAGPipeline


def run_demo():
    print("=" * 70)
    print("   MEMBER 2 STANDALONE RAG PIPELINE DEMO")
    print("=" * 70)
    print("This demo runs Member 2's RAG pipeline independently.\n")

    pipeline = RAGPipeline()

    sample_questions = [
        "What is the maximum allowable per diem for daily meal expenses?",
        "What is the paid parental leave duration for primary caregivers?",
        "Can I receive corporate bonuses in Bitcoin?",
        "How many days of annual leave do full-time employees receive?",
        "leave"
    ]

    print("Preset Test Questions:")
    for idx, q in enumerate(sample_questions, 1):
        print(f"  [{idx}] {q}")
    print("  [0] Type a custom question")

    choice = input("\nSelect an option [0-5] (default 1): ").strip()
    if not choice:
        choice = "1"

    if choice.isdigit() and 1 <= int(choice) <= len(sample_questions):
        question = sample_questions[int(choice) - 1]
    else:
        question = input("Enter your custom question: ").strip()

    if not question:
        print("[ERROR] Question cannot be empty.")
        return

    print(f"\n[QUERY]: \"{question}\"")
    print("[RUNNING RAG PIPELINE] Retrieving evidence -> Validating -> Detecting Conflicts -> Generating Answer...")

    result = pipeline.answer_question(question)

    print("\n" + "=" * 70)
    print("MEMBER 2 JSON OUTPUT:")
    print("=" * 70)
    print(json.dumps(result, indent=4))
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
