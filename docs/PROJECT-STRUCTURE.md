# Project structure

The full annotated tree. Note the split between `tests/` (automated pytest, run in CI)
and `scripts/test_*.py` (manual verification run by hand against real files).

[← back to the README](../README.md)

---

## Project structure

```
ai-teaching-assistant/
  app/
    agent/
      graph.py     # orchestration: tool dispatch, structured generation, grounding checks
      nodes.py     # retrieval + per-tool schemas/renderers (shared reasoning steps)
      tools.py     # tool schemas + retrieval-only implementations
    rag/
      parser.py      # PDF/PPTX/DOCX -> text + page/slide/heading/chapter metadata
      chunker.py      # heading-based grouping, 1200-2000 tokens, 15-20% overlap
      embeddings.py   # BGE-small-en-v1.5 wrapper (asymmetric query/doc encoding)
      retriever.py    # Chroma dense retrieval
      hybrid.py       # BM25 + RRF fusion
    database/         # Chroma + BM25 persistence (gitignored)
      chroma/            # Chroma's own persistent sqlite + vector storage
      bm25_index.pkl      # pickled BM25 sparse index
    memory/           # rolling conversation-history trimming
      session_memory.py
    ui/
      app.py           # Streamlit page
      components.py    # sidebar + panel render functions
  config.py
  main.py              # streamlit entry point
  scripts/
    cli.py              # terminal alternative to the UI
    test_*.py           # MANUAL verification scripts for each pipeline stage (parser, chunker, hybrid, retriever, search_topic, all 6 tools) -- run by hand against real files, not CI
  tests/
    test_chunker.py         # automated pytest: chunk_document's grouping/merge/split/back-matter logic
    test_bm25_retrieval.py  # automated pytest: BM25 keyword search + Uzbek apostrophe normalization
    test_hybrid_fusion.py   # automated pytest: Reciprocal Rank Fusion formula
  data/
    samples/           # real sample files used during development verification
    uploads/            # files uploaded through the running app
  requirements.txt
  requirements-dev.txt
  pytest.ini
  README.md
  SPEC.md
```
