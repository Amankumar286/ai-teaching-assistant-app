"""Manual verification script for hybrid.py -- indexes real sample chunks
into scratch Chroma + BM25 stores, then for a handful of queries prints
dense-only, BM25-only, and RRF-fused results side by side, so the fusion's
actual effect can be eyeballed directly (per SPEC.md's build order: "verify
fused ranking looks better than either alone").

Uses scratch store paths (not the app's real database/) so running this
repeatedly never pollutes real indexed data.

Usage: .venv/Scripts/python.exe scripts/test_hybrid.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config

config.CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "_scratch_chroma_hybrid"
config.BM25_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "_scratch_bm25.pkl"

from app.rag.chunker import chunk_document
from app.rag.parser import parse_file
from app.rag import hybrid

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

# Deliberately mixed: exact-term queries (BM25's strength -- literal
# vocabulary shared with the source text) and paraphrase/semantic queries
# with little shared vocabulary (dense's strength), plus one filtered query
# to exercise the where= passthrough on both sides.
TEST_QUERIES = [
    ("AES", None),  # exact algorithm name, verbatim in the cybersecurity deck
    ("ERI", None),  # exact acronym (elektron raqamli imzo), verbatim in the deck
    ("ma'lumotlarni ruxsatsiz kirishdan qanday himoya qilish mumkin", None),  # paraphrase, no exact term overlap
    ("bolalar uchun jismoniy sinov ballari", None),  # paraphrase of sports test-norms content
    ("shifrlash", {"file_name": "Lecture_Kiberxavfsizlik.pptx"}),  # where-filtered
]


def preview(text: str, n: int = 150) -> str:
    flat = " ".join(text.split())
    return flat[:n] + ("..." if len(flat) > n else "")


def print_hits(label: str, hits: list):
    print(f"  -- {label} --")
    if not hits:
        print("    (no hits)")
    for i, h in enumerate(hits):
        m = h["metadata"]
        extra = ""
        if "dense_rank" in h:
            extra = f"  (dense_rank={h['dense_rank']}, bm25_rank={h['bm25_rank']})"
        print(f"    [{i}] score={h['score']:.4f}  source={m.get('source')}{extra}")
        print(f"        text: {preview(h['text'])}")


def main():
    hybrid.clear_knowledge_base()

    files = sorted(SAMPLES_DIR.iterdir())
    if not files:
        print(f"No sample files found in {SAMPLES_DIR}")
        return

    total = 0
    for path in files:
        doc = parse_file(path)
        chunks = chunk_document(doc)
        n = hybrid.index_chunks(chunks)
        total += n
        print(f"Indexed {n} chunks from {path.name}")
    print(f"\nTotal indexed: {total} chunks. Document count: {hybrid.document_count()}\n")

    for query_text, where in TEST_QUERIES:
        print("=" * 100)
        print(f"QUERY: {query_text!r}  where={where}")
        print("=" * 100)
        print_hits("DENSE ONLY", hybrid.search_dense_only(query_text, top_k=3, where=where))
        print_hits("BM25 ONLY", hybrid.search_bm25_only(query_text, top_k=3, where=where))
        print_hits("RRF FUSED", hybrid.search(query_text, top_k=3, where=where))
        print()

    # Re-index the same files again and confirm the BM25 corpus doesn't grow
    # (upsert-by-id, not duplication).
    print("=" * 100)
    print("RE-INDEXING SAME FILES (checking for duplication)")
    print("=" * 100)
    before_chroma, before_bm25 = hybrid.document_count(), hybrid.bm25_document_count()
    for path in files:
        doc = parse_file(path)
        chunks = chunk_document(doc)
        hybrid.index_chunks(chunks)
    after_chroma, after_bm25 = hybrid.document_count(), hybrid.bm25_document_count()
    print(f"Chroma count before/after: {before_chroma}/{after_chroma} (should match)")
    print(f"BM25 count before/after: {before_bm25}/{after_bm25} (should match)")


if __name__ == "__main__":
    main()
