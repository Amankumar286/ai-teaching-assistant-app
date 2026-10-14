"""Manual verification script for the remaining 5 tools (generate_quiz,
generate_assignment, compare_topics, summarize, explain_difficult_concept)
plus tool-dispatch correctness across all 6. Runs real questions through the
actual qwen2.5:7b model against the same 3 real sample files, printing full
output (not just non-empty checks) so grounding/structure can be read
directly.

Usage: .venv/Scripts/python.exe scripts/test_all_tools.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config

config.CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "_scratch_chroma_tools"
config.BM25_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "_scratch_bm25_tools.pkl"

from app.agent.graph import run_agent
from app.rag.chunker import chunk_document
from app.rag.parser import parse_file
from app.rag import hybrid

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def ask(label, question):
    print("=" * 100)
    print(f"[{label}] {question}")
    print("=" * 100)
    start = time.time()
    result = run_agent(question)
    elapsed = time.time() - start
    print(f"[{elapsed:.1f}s]  used_native_tool_calling={result['used_native_tool_calling']}")
    print(f"tool_calls_log: {result['tool_calls_log']}")
    print(f"\nOutput:\n{result['answer_text']}\n")
    return result


def main():
    hybrid.clear_knowledge_base()
    files = sorted(SAMPLES_DIR.iterdir())
    total = 0
    for path in files:
        doc = parse_file(path)
        chunks = chunk_document(doc)
        total += hybrid.index_chunks(chunks)
    print(f"Indexed {total} chunks from {len(files)} files.\n")

    # --- generate_quiz ---
    ask("QUIZ on-topic", "Kiberxavfsizlik bo'yicha 3 ta test savoli tuzib ber")
    ask("QUIZ off-topic", "Kvant kompyuterlari bo'yicha test savollari tuzib ber")

    # --- compare_topics ---
    ask("COMPARE both covered", "Troposfera va stratosferani solishtirib ber")
    ask("COMPARE one uncovered", "Troposfera va kvant kompyuterlarini solishtirib ber")

    # --- summarize ---
    ask("SUMMARIZE matched reference", "Aerologiya darsligini qisqacha aytib ber")
    ask("SUMMARIZE unmatched reference", "Mavjud bo'lmagan 'Fizika 101' darsligini xulosalab ber")

    # --- explain_difficult_concept ---
    ask("EXPLAIN on-topic", "Shifrlash tushunchasini sodda qilib, misol bilan tushuntirib ber")
    ask("EXPLAIN off-topic", "Blokcheyn tushunchasini sodda qilib tushuntirib ber")

    # --- generate_assignment ---
    ask("ASSIGNMENT on-topic", "Kiberxavfsizlik bo'yicha yangi amaliy topshiriq tuzib ber")
    ask("ASSIGNMENT off-topic", "Sun'iy intellekt etikasi bo'yicha topshiriq tuzib ber")

    # --- dispatch correctness ---
    print("=" * 100)
    print("DISPATCH CHECK -- expect the tool name shown in [] for each")
    print("=" * 100)
    dispatch_cases = [
        ("search_topic", "Kiberxavfsizlik nima?"),
        ("generate_quiz", "Meni jismoniy tayyorgarlik sinovlari bo'yicha sinab ko'r"),
        ("compare_topics", "AES va shifrlashning boshqa turlarini solishtir"),
        ("summarize", "JTD sinov me'yori hujjatini xulosalab ber"),
        ("explain_difficult_concept", "Ekzosfera nima ekanini sodda qilib tushuntir"),
        ("generate_assignment", "Atmosfera qatlamlari bo'yicha mashq topshirig'i yarat"),
    ]
    for expected_tool, question in dispatch_cases:
        result = run_agent(question)
        actual = next((c["name"] for c in result["tool_calls_log"] if not c.get("forced")), "search_topic (forced)")
        match = "OK" if actual == expected_tool else "MISMATCH"
        print(f"  [{match}] expected={expected_tool}  actual={actual}  question={question!r}")


if __name__ == "__main__":
    main()
