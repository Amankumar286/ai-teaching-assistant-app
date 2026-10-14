import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # optional: only needed for the Gemini settings below

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_DIR = PROJECT_ROOT / "app" / "database"
CHROMA_DIR = DATABASE_DIR / "chroma"
BM25_INDEX_PATH = DATABASE_DIR / "bm25_index.pkl"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_MODEL_FALLBACK = "qwen2.5:3b-instruct-q4_K_M"

# ── Gemini (cloud, optional) ─────────────────────────
# This project is local-first BY DESIGN -- every anti-hallucination fix in
# the commit history was tuned and tested against Ollama/qwen2.5's specific
# response format. Gemini support exists so a public deployment (no Ollama
# server available) is still usable with zero setup: a visitor pastes
# their own key in the sidebar. See README's "Anti-hallucination
# guarantees are Ollama-only" section before treating Gemini mode as
# equivalent to the default path.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

CHUNK_MIN_TOKENS = 1200  # sections smaller than this keep merging forward with the next one
CHUNK_MAX_TOKENS = 2000
CHUNK_OVERLAP_RATIO = 0.175

DEFAULT_TOP_K = 5
QUIZ_TOP_K = 8  # quiz needs distinct material for several non-repetitive items, but not so much that the
# model starts fabricating unrelated content -- found live: "Quiz me on diffusion models" at the old
# QUIZ_TOP_K=12 produced 5 questions entirely about Graph Convolutional Networks, a topic that didn't
# appear in ANY of the 12 retrieved chunks, while falsely citing real diffusion-model page numbers as
# if they supported that content. Verified directly: the same query at top_k=6 produced 5/5 genuinely
# on-topic questions (noise scheduler, accelerator.prepare, pipeline.save_pretrained); 8 is a middle
# ground that keeps more headroom for larger requested quizzes while meaningfully shrinking the pool
# from the value that reproduced full content fabrication.
BROAD_TOP_K = 12  # generate_assignment and each side of compare_topics
SUMMARIZE_TOP_K = 20  # summarize needs comprehensive coverage of one document, not just top-k nearest neighbors

RRF_K = 60  # standard constant from the original Reciprocal Rank Fusion paper
CANDIDATE_POOL_MIN = 20
CANDIDATE_POOL_MULTIPLIER = 4
