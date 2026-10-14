"""Manual verification script for parser.py -- run against real sample files
and eyeball the output for correctness (text/heading/page/slide extraction),
not just assert counts.

Usage: .venv/Scripts/python.exe scripts/test_parser.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.rag.parser import parse_file

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def preview(text: str, n: int = 150) -> str:
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
        print(f"file_type={doc.file_type}  lecture_number={doc.lecture_number}  units={len(doc.units)}")
        if doc.warnings:
            print(f"WARNINGS ({len(doc.warnings)}):")
            for w in doc.warnings[:10]:
                print(f"  - {w}")
            if len(doc.warnings) > 10:
                print(f"  ... and {len(doc.warnings) - 10} more")
        print()

        # Print first N units in full, then just a summary line for the rest,
        # so real content is eyeballed without flooding the terminal.
        FULL_DETAIL_COUNT = 8
        for i, u in enumerate(doc.units):
            loc = f"page={u.page_number}" if u.page_number is not None else (
                f"slide={u.slide_number}" if u.slide_number is not None else "doc-section"
            )
            if i < FULL_DETAIL_COUNT:
                print(f"[{i}] type={u.unit_type} {loc} chapter={u.chapter!r} heading={u.heading!r} empty={u.is_empty}")
                print(f"    text: {preview(u.text)}")
            elif i == FULL_DETAIL_COUNT:
                print(f"... ({len(doc.units) - FULL_DETAIL_COUNT} more units, showing headings only)")
            else:
                if u.heading or u.chapter:
                    print(f"[{i}] {loc} chapter={u.chapter!r} heading={u.heading!r}")
        print()


if __name__ == "__main__":
    main()
