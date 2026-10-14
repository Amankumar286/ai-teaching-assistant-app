# AI Teaching Assistant — Build Instructions for Claude Code

## Project Overview

Build an **AI Teaching Assistant Agent** that helps a university student study from their own uploaded materials (lecture slides, books, assignments). The agent must answer strictly from the uploaded content — if something isn't in the materials, it must say so explicitly rather than guessing.

This runs **fully local** — no cloud APIs. Local LLM via Ollama, local embeddings, local vector store.

## Tech Stack (decided)

- **LLM**: Qwen2.5 7B Instruct via Ollama (`ollama pull qwen2.5:7b`), Q4_K_M quantization. Fallback to `qwen2.5:3b` if generation is too slow on a 6GB VRAM GPU.
- **Dense retrieval**: Chroma (persistent local client, native metadata filtering)
- **Sparse retrieval**: BM25 (`rank_bm25` package)
- **Fusion**: Reciprocal Rank Fusion (RRF) to combine dense + sparse rankings
- **UI**: Streamlit
- **Parsing**: PyMuPDF (`fitz`) for PDF, `python-pptx` for PPTX, `python-docx` for DOCX
- **Embeddings**: a local sentence-transformers model (e.g. `BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2`) run via `sentence-transformers`, no API calls

## Hard Requirements (non-negotiable)

1. **Answers must be grounded in retrieved chunks only.** If retrieval returns nothing relevant, the agent must say the topic wasn't found in the uploaded materials — never fabricate.
2. **Every answer must cite sources** (file name + page/slide/chapter number) at the end.
3. **RAG must actually be used for every content-generating tool** — no tool should generate an answer, quiz, or assignment without first retrieving context.
4. Retrieval must be **hybrid** (dense + sparse fused), not dense-only or sparse-only.
5. Chunking must be **large and topic-coherent** (1200–2000 tokens, 15–20% overlap, OR heading/chapter/section-based) — not naive fixed 300–500 token splitting.
6. Everything must run offline once models are pulled — no network calls to external APIs.

## Project Structure

```
project/
  app/
    agent/
      graph.py       # agent orchestration / control flow
      nodes.py       # individual reasoning steps
      tools.py       # tool definitions for tool-calling
    rag/
      parser.py      # PDF/PPTX/DOCX -> text + metadata
      chunker.py      # heading/section-based or large-chunk splitting
      embeddings.py   # local embedding model wrapper
      retriever.py    # dense (Chroma) retrieval
      hybrid.py       # BM25 + dense fusion (RRF)
    database/         # Chroma persistence, BM25 index storage
    memory/           # conversation history management
    ui/               # Streamlit pages/components
  config.py
  main.py
  README.md
  SPEC.md             # this file, for reference
```

## Metadata Schema (attach to every chunk)

```python
{
    "file_name": str,
    "file_type": str,       # "pdf" | "pptx" | "docx"
    "lecture_number": Optional[int],
    "chapter": Optional[str],
    "page_number": Optional[int],
    "slide_number": Optional[int],
    "topic": Optional[str],  # inferred if possible, else None
    "source": str,           # human-readable source label
}
```

Every answer shown to the user must display this metadata for each chunk it drew from, e.g.:

```
Sources:
- Lecture_05.pdf, Page 18 — Topic: Transformer Architecture
- Deep_Learning_Book.pdf, Chapter 9
```

## Tools the Agent Must Support (tool calling)

1. **search_topic** — retrieve relevant chunks on a topic, answer using them.
2. **generate_quiz** — retrieve context on a topic, generate N multiple-choice questions, each with 4 options, correct answer marked, and a short explanation.
3. **generate_assignment** — retrieve context across lecture/book/assignment materials and synthesize a new assignment (e.g. modeled after an existing one).
4. **compare_topics** — retrieve context for two topics, compare them, and present the result as a table.
5. **summarize** — retrieve a specific lecture/chapter and produce a summary.
6. **explain_difficult_concept** — retrieve context and produce a simplified explanation with an example.

All tools must go through the hybrid retriever — none should answer from the LLM's own knowledge alone.

## Conversation Memory

The agent must retain context across turns within a session (e.g. "Transformer nima?" followed by "Encoder haqida-chi?" should be understood as still about Transformers). Implement as a rolling conversation history injected into the agent's prompt/context, stored in `memory/`.

## Streamlit UI Requirements

**Sidebar:**
- Upload Lecture Slides
- Upload Books
- Upload Assignments
- Indexed Documents (list of what's been parsed/indexed)
- Clear Knowledge Base (button to reset Chroma + BM25 index)
- Settings (model choice, top-k, chunking params)

**Main page:**
- Chat interface
- Generated Questions (quiz display area)
- Retrieved Sources (shown per answer)
- Tool Logs (which tool was called, what it retrieved — useful for debugging and grading transparency)
- Conversation History

## Build Order (follow this sequence — do not skip ahead)

Build and verify each stage before moving to the next. Don't wire up the LLM or UI until parsing + retrieval are proven to work correctly on real files.

1. Project skeleton with stub files matching the structure above.
2. `parser.py` — test on real PDF/PPTX/DOCX files, verify text + metadata extraction is correct.
3. `chunker.py` — verify chunks are topic-coherent by printing samples, not just token-counting.
4. `embeddings.py` + `retriever.py` (Chroma) — verify by manually querying and checking retrieved chunks make sense.
5. BM25 index + `hybrid.py` (RRF fusion) — verify fused ranking looks better than either alone on a few test queries.
6. Ollama integration + tool calling — get ONE tool (`search_topic`) working reliably before adding the rest.
7. Remaining five tools — should reuse the retriever, so should come fast.
8. Memory — multi-turn context handling.
9. Streamlit UI — wire up last.
10. README with architecture diagram, installation, Ollama setup, screenshots.

## Grading Weights (for reference — prioritize accordingly)

| Criterion | Points |
|---|---|
| Hybrid RAG | 20 |
| Tool Calling | 15 |
| Question Generation | 15 |
| Multi-format Parsing | 10 |
| Chunking Strategy | 10 |
| Topic Search | 10 |
| Memory | 10 |
| UI | 5 |
| Source Citation | 5 |

## Bonus Ideas (only after core 100 points are solid)

- Flashcard generator
- Difficulty levels (Easy/Medium/Hard) for generated questions
- Study plan generator
- Exam simulation mode
- Voice input
- OCR support for scanned PDFs
- Multi-agent architecture
- Reranking model added to the retrieval pipeline
- PDF/DOCX export of generated quizzes

---

**Start here**: Set up the project structure above, then build `app/rag/parser.py` first. Do not touch the LLM or Streamlit UI until parsing, chunking, and hybrid retrieval have been manually tested on real lecture/book/assignment files.
