"""Manual verification script for embeddings.py + retriever.py -- indexes
real sample chunks into a scratch Chroma collection, then runs a handful of
hand-written queries and prints retrieved chunk text + metadata + score, so
relevance can be eyeballed directly (per SPEC.md's build order).

Uses a separate scratch Chroma dir (not the app's real database/) so running
this repeatedly never pollutes real indexed data.

Usage: .venv/Scripts/python.exe scripts/test_retriever.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config

config.CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "_scratch_chroma"

from app.rag.chunker import chunk_document
from app.rag.parser import parse_file
from app.rag import retriever

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

# Filled in once real file contents are known from test_parser.py output --
# a mix of on-topic queries (should return relevant hits) and one clearly
# off-topic query (should return weak/irrelevant hits, proving the retriever
# doesn't force a match when nothing relevant exists).
TEST_QUERIES = [
    "shifrlash asoslari",  # encryption basics (cybersecurity deck)
    "kiberxavfsizlik tahdidlari",  # cybersecurity threats
    "atmosfera qatlamlari",  # atmosphere layers (aerology textbook)
    "havo harorati va bosimi",  # air temperature and pressure
    "sinov me'yorlari",  # test norms (JTD docx)
    "kvant kompyuterlari va blokcheyn",  # quantum computers/blockchain -- should NOT be in any of these docs
]


def preview(text: str, n: int = 200) -> str:
    flat = " ".join(text.split())
    return flat[:n] + ("..." if len(flat) > n else "")


def main():
    retriever.clear_knowledge_base()

    files = sorted(SAMPLES_DIR.iterdir())
    if not files:
        print(f"No sample files found in {SAMPLES_DIR}")
        return

    total_chunks = 0
    for path in files:
        doc = parse_file(path)
        chunks = chunk_document(doc)
        n = retriever.index_chunks(chunks)
        total_chunks += n
        print(f"Indexed {n} chunks from {path.name}")

    print(f"\nTotal indexed: {total_chunks} chunks. Collection count: {retriever.document_count()}")
    print(f"Indexed files: {retriever.list_indexed_files()}\n")

    for q in TEST_QUERIES:
        print("=" * 100)
        print(f"QUERY: {q}")
        print("=" * 100)
        hits = retriever.query(q, top_k=3)
        if not hits:
            print("  (no hits)")
        for i, h in enumerate(hits):
            m = h["metadata"]
            print(f"  [{i}] score={h['score']:.3f}  source={m.get('source')}")
            print(f"      text: {preview(h['text'])}")
        print()


if __name__ == "__main__":
    main()
