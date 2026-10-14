"""Manual verification script for graph.run_agent (search_topic) -- indexes
real sample chunks into scratch stores, then runs a fixed set of real
questions end-to-end through the LLM, printing full answers (not just
non-empty checks) so grounding and citation correctness can be read
directly. Also checks the empty-knowledge-base short-circuit and logs
whether native tool-calling fired vs the forced-retrieval fallback.

Usage: .venv/Scripts/python.exe scripts/test_search_topic.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config

config.CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "_scratch_chroma_agent"
config.BM25_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "_scratch_bm25_agent.pkl"

from app.agent.graph import run_agent
from app.rag.chunker import chunk_document
from app.rag.parser import parse_file
from app.rag import hybrid

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

TEST_QUESTIONS = [
    "Kiberxavfsizlik nima?",  # what is cybersecurity -- on-topic
    "AES shifrlash algoritmi haqida gapirib bering",  # tell me about the AES algorithm -- on-topic, exact term
    "Atmosfera qatlamlari qanday tuzilgan?",  # how are atmosphere layers structured -- on-topic
    "Jismoniy tayyorgarlik sinovlarida necha ball to'plash kerak?",  # how many points needed in fitness tests -- on-topic
    "Kvant kompyuterlari qanday ishlaydi?",  # how do quantum computers work -- deliberately OFF-topic
]


def main():
    print("=" * 100)
    print("EMPTY KNOWLEDGE BASE CHECK (before indexing anything)")
    print("=" * 100)
    hybrid.clear_knowledge_base()
    result = run_agent("Kiberxavfsizlik nima?")
    print(f"Answer: {result['answer_text']}")
    print(f"used_native_tool_calling={result['used_native_tool_calling']}  tool_calls_log={result['tool_calls_log']}")
    print()

    files = sorted(SAMPLES_DIR.iterdir())
    total = 0
    for path in files:
        doc = parse_file(path)
        chunks = chunk_document(doc)
        total += hybrid.index_chunks(chunks)
    print(f"Indexed {total} chunks from {len(files)} files.\n")

    native_count = 0
    for question in TEST_QUESTIONS:
        print("=" * 100)
        print(f"QUESTION: {question}")
        print("=" * 100)
        start = time.time()
        result = run_agent(question)
        elapsed = time.time() - start
        native_count += int(result["used_native_tool_calling"])
        print(f"[{elapsed:.1f}s]  used_native_tool_calling={result['used_native_tool_calling']}")
        print(f"tool_calls_log: {result['tool_calls_log']}")
        print(f"\nAnswer:\n{result['answer_text']}\n")

    print("=" * 100)
    print(f"Native tool-calling fired on {native_count}/{len(TEST_QUESTIONS)} questions "
          f"(rest used the forced-retrieval fallback).")


if __name__ == "__main__":
    main()
