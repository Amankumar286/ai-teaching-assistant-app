"""Manual verification script for chunker.py -- runs parser -> chunker on
real sample files and prints each chunk's metadata + a text preview, so
topic coherence can be eyeballed (per SPEC.md's build order), not just
inferred from token counts.

Usage: .venv/Scripts/python.exe scripts/test_chunker.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.rag.chunker import chunk_document
from app.rag.parser import parse_file

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def preview(text: str, n: int = 300) -> str:
    flat = " ".join(text.split())
    return flat[:n] + ("..." if len(flat) > n else "")


def main():
    files = sorted(SAMPLES_DIR.iterdir())
    if not files:
        print(f"No sample files found in {SAMPLES_DIR}")
        return

    for path in files:
        print("=" * 100)
        print(f"FILE: {path.name}")
        print("=" * 100)
        doc = parse_file(path)
        chunks = chunk_document(doc)
        token_counts = [c.token_count for c in chunks]
        print(f"units={len(doc.units)}  chunks={len(chunks)}")
        if token_counts:
            print(
                f"token_count: min={min(token_counts)} max={max(token_counts)} "
                f"avg={sum(token_counts) / len(token_counts):.0f}"
            )
        print()

        for i, c in enumerate(chunks):
            print(f"--- chunk {i} ---")
            print(f"  chapter={c.chapter!r}  topic={c.topic!r}  tokens={c.token_count}")
            print(f"  page={c.page_number}-{c.page_end}  slide={c.slide_number}-{c.slide_end}")
            print(f"  source: {c.source}")
            print(f"  text: {preview(c.text)}")
        print()


if __name__ == "__main__":
    main()
