import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure root is in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()

from rag.pipeline import RAGPipeline


def main():
    print("=" * 65)
    print("   ENTERPRISE RAG PIPELINE (MEMBER 2 MODULE)")
    print("=" * 65)

    pipeline = RAGPipeline(top_k=4, similarity_threshold=0.25)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("\nEnter your question: ").strip()

    if not question:
        print("[ERROR] Question cannot be empty.")
        return

    print(f"\n[QUERY]: \"{question}\"")
    print("[PROCESSING] Retrieving evidence, validating facts, detecting conflicts...")

    # Execute RAG Pipeline
    result = pipeline.answer_question(question)

    print("\n" + "=" * 65)
    print("PIPELINE JSON RESPONSE:")
    print("=" * 65)
    print(json.dumps(result, indent=4))
    print("=" * 65)


if __name__ == "__main__":
    main()
