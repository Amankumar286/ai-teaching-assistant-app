"""Local sentence-transformers embedding wrapper (BGE).

BGE models are asymmetric: retrieval quality depends on adding a query
instruction prefix ("Represent this sentence for searching relevant
passages:") to queries but NOT to indexed documents. embed_query() and
embed_documents() are kept as separate entry points specifically so this
distinction can never be accidentally collapsed by a caller.
"""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BGE_QUERY_PREFIX, EMBEDDING_MODEL_NAME

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_documents(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    vectors = _get_model().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    vector = _get_model().encode(BGE_QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()
