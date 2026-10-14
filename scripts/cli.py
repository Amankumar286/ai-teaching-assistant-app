"""CLI entry point -- a quick way to use/test the assistant without the
Streamlit UI (e.g. after making a code change). Indexes real files into the
persistent knowledge base (app/database/), then runs an interactive chat
loop over the full agent (all 6 tools, native tool-calling, hybrid retrieval).

Usage:
    .venv/Scripts/python.exe scripts/cli.py --add path/to/file.pdf path/to/other.pptx ...
    .venv/Scripts/python.exe scripts/cli.py --clear
    .venv/Scripts/python.exe scripts/cli.py                          # interactive chat loop
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.agent.graph import run_agent
from app.memory.session_memory import trim_history
from app.rag import hybrid
from app.rag.chunker import chunk_document
from app.rag.parser import ParseError, parse_file


def add_files(paths):
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"  ! {path}: file not found, skipping")
            continue
        try:
            doc = parse_file(path)
        except ParseError as e:
            print(f"  ! {path.name}: {e}")
            continue
        chunks = chunk_document(doc)
        n = hybrid.index_chunks(chunks)
        print(f"  + {path.name}: indexed {n} chunks" + (f" ({len(doc.warnings)} warnings)" if doc.warnings else ""))
    print(f"\nKnowledge base now has {hybrid.document_count()} chunks from {len(hybrid.list_indexed_files())} files.")


def run_chat():
    print("AI Teaching Assistant -- type your question, or 'exit'/'quit' to leave.\n")
    files = hybrid.list_indexed_files()
    if files:
        print(f"Indexed materials: {', '.join(files)}\n")
    else:
        print("No materials indexed yet -- run with --add <file> first.\n")

    history = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit"):
            break

        result = run_agent(question, history=trim_history(history))
        print(f"\nAssistant: {result['answer_text']}\n")
        if result["tool_calls_log"]:
            print(f"  [tool: {result['tool_calls_log']}]\n")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer_text"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--add", nargs="+", metavar="FILE", help="Index one or more files (.pdf/.pptx/.docx), then exit.")
    parser.add_argument("--clear", action="store_true", help="Clear the knowledge base, then exit.")
    args = parser.parse_args()

    if args.clear:
        hybrid.clear_knowledge_base()
        print("Knowledge base cleared.")
        return

    if args.add:
        add_files(args.add)
        return

    run_chat()


if __name__ == "__main__":
    main()
