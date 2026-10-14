"""BM25 (sparse) + Chroma (dense) fused via Reciprocal Rank Fusion.

This is the module the rest of the app should call for indexing/search from
now on -- it keeps the BM25 corpus and the Chroma collection in sync on the
same chunk ids (retriever.chunk_id), which is what lets RRF recognize "the
same chunk" across both rankers' result lists.

rank_bm25's BM25Okapi has no incremental-add API, so the BM25 "index" here
is really a persisted list of {id, text, metadata, tokens} plus an in-memory
BM25Okapi rebuilt whenever that list changes -- cheap at the corpus sizes
this project will ever see.
"""

import pickle
import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    BM25_INDEX_PATH,
    CANDIDATE_POOL_MIN,
    CANDIDATE_POOL_MULTIPLIER,
    DEFAULT_TOP_K,
    RRF_K,
)

from app.rag import retriever
from app.rag.chunker import Chunk

# Uzbek text uses several different Unicode characters as its apostrophe
# ("tutuq belgisi") depending on how a document was authored/exported --
# o'zbek / o‘zbek / o’zbek / oʻzbekiston are the same word. Normalizing them
# to one canonical form before tokenizing keeps exact-term BM25 matching
# from silently fragmenting across documents.
_APOSTROPHE_VARIANTS = "'’‘ʻʼ`´"
_CANONICAL_APOSTROPHE = "ʻ"
_APOSTROPHE_TRANS = str.maketrans({c: _CANONICAL_APOSTROPHE for c in _APOSTROPHE_VARIANTS})
_TOKEN_RE = re.compile(rf"[\w{_CANONICAL_APOSTROPHE}]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    normalized = text.lower().translate(_APOSTROPHE_TRANS)
    return _TOKEN_RE.findall(normalized)


def _matches_where(metadata: dict, where: Optional[dict]) -> bool:
    if not where:
        return True
    return all(metadata.get(k) == v for k, v in where.items())


class BM25Store:
    def __init__(self):
        self._entries: List[dict] = []  # {"id", "text", "metadata", "tokens"}
        self._id_to_index: dict = {}
        self._bm25 = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        if BM25_INDEX_PATH.exists():
            with open(BM25_INDEX_PATH, "rb") as f:
                self._entries = pickle.load(f)
        self._loaded = True
        self._rebuild()

    def _rebuild(self):
        self._id_to_index = {e["id"]: i for i, e in enumerate(self._entries)}
        if self._entries:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([e["tokens"] for e in self._entries])
        else:
            self._bm25 = None

    def _persist(self):
        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(self._entries, f)

    def add_chunks(self, chunks: List[Chunk], ids: List[str]):
        self._ensure_loaded()
        for cid, chunk in zip(ids, chunks):
            entry = {
                "id": cid,
                "text": chunk.text,
                "metadata": chunk.to_metadata(),
                "tokens": tokenize(chunk.text),
            }
            if cid in self._id_to_index:
                self._entries[self._id_to_index[cid]] = entry
            else:
                self._id_to_index[cid] = len(self._entries)
                self._entries.append(entry)
        self._rebuild()
        self._persist()

    def search(self, query_text: str, top_k: int, where: Optional[dict] = None) -> List[dict]:
        self._ensure_loaded()
        if not self._entries or self._bm25 is None:
            return []

        candidate_indices = range(len(self._entries))
        if where:
            candidate_indices = [i for i in candidate_indices if _matches_where(self._entries[i]["metadata"], where)]
            if not candidate_indices:
                return []

        scores = self._bm25.get_scores(tokenize(query_text))
        # A score of ~0 means "shares no discriminating term with the query"
        # -- for a rare/exact-term query, most of the corpus ties at exactly
        # 0. Ranking those ties (rank 2, 3, ...) would hand them a real RRF
        # contribution as if they were meaningfully ordered, when they're
        # actually just as irrelevant as a document that was never ranked at
        # all -- which can drown out the one genuine match. Only real
        # (>0) matches count as ranked hits.
        nonzero = [i for i in candidate_indices if scores[i] > 1e-9]
        ranked = sorted(nonzero, key=lambda i: scores[i], reverse=True)[:top_k]

        hits = []
        for i in ranked:
            e = self._entries[i]
            hits.append({"id": e["id"], "text": e["text"], "metadata": e["metadata"], "score": float(scores[i])})
        return hits

    def clear(self):
        self._entries = []
        self._id_to_index = {}
        self._bm25 = None
        self._loaded = True
        self._persist()

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._entries)


_store: Optional[BM25Store] = None


def _get_store() -> BM25Store:
    global _store
    if _store is None:
        _store = BM25Store()
    return _store


def rrf_fuse(dense_hits: List[dict], sparse_hits: List[dict], top_k: int, k: int = RRF_K) -> List[dict]:
    """Reciprocal Rank Fusion: score(id) = sum over rankers of 1/(k + rank).
    A chunk absent from one ranker's candidate pool simply contributes 0
    from that side rather than being penalized."""
    scores: dict = {}
    payload: dict = {}
    ranks: dict = {}

    for rank, hit in enumerate(dense_hits, start=1):
        cid = hit["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        payload.setdefault(cid, hit)
        ranks.setdefault(cid, {})["dense_rank"] = rank

    for rank, hit in enumerate(sparse_hits, start=1):
        cid = hit["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        payload.setdefault(cid, hit)
        ranks.setdefault(cid, {})["bm25_rank"] = rank

    ordered_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]

    results = []
    for cid in ordered_ids:
        hit = dict(payload[cid])
        hit["score"] = scores[cid]
        hit["dense_rank"] = ranks[cid].get("dense_rank")
        hit["bm25_rank"] = ranks[cid].get("bm25_rank")
        results.append(hit)
    return results


def index_chunks(chunks: List[Chunk]) -> int:
    if not chunks:
        return 0
    ids = [retriever.chunk_id(c, i) for i, c in enumerate(chunks)]
    retriever.index_chunks(chunks, ids=ids)
    _get_store().add_chunks(chunks, ids)
    return len(chunks)


def search_dense_only(query_text: str, top_k: int = DEFAULT_TOP_K, where: Optional[dict] = None) -> List[dict]:
    return retriever.query(query_text, top_k=top_k, where=where)


def search_bm25_only(query_text: str, top_k: int = DEFAULT_TOP_K, where: Optional[dict] = None) -> List[dict]:
    return _get_store().search(query_text, top_k=top_k, where=where)


def search(query_text: str, top_k: int = DEFAULT_TOP_K, where: Optional[dict] = None) -> List[dict]:
    pool_size = max(top_k * CANDIDATE_POOL_MULTIPLIER, CANDIDATE_POOL_MIN)
    dense_hits = retriever.query(query_text, top_k=pool_size, where=where)
    sparse_hits = _get_store().search(query_text, top_k=pool_size, where=where)
    return rrf_fuse(dense_hits, sparse_hits, top_k=top_k)


def sample_spread(where: dict, n: int) -> List[dict]:
    """Returns up to n chunks matching `where`, evenly spread across the
    matched scope's page/slide order -- for genuine whole-document (or
    whole-chapter) coverage instead of a similarity-ranked cluster around
    whatever text most resembles the query wording.

    Found live: summarize()'s matched-reference path still ran an ordinary
    similarity search even once scoped to one file via `where`, so
    "summarize Mastering Pytorch 2ed.pdf" kept returning a summary of one
    narrow ONNX-export section -- with a DIFFERENT page citation each
    run -- instead of surveying the whole 559-page book. Filtering to the
    right file doesn't fix that: the chunks within that file are still
    ranked by similarity to the literal query text, which clusters around
    one topic rather than sampling the book's actual breadth."""
    items = retriever.get_all_for_where(where)
    if not items:
        return []

    def sort_key(item: dict):
        meta = item["metadata"]
        page = meta.get("page_number")
        if page is not None:
            return page
        return meta.get("slide_number") or 0

    items.sort(key=sort_key)

    if len(items) <= n:
        return items

    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def clear_knowledge_base():
    retriever.clear_knowledge_base()
    _get_store().clear()


def document_count() -> int:
    return retriever.document_count()


def bm25_document_count() -> int:
    return _get_store().count()


def list_indexed_files() -> List[str]:
    return retriever.list_indexed_files()


def list_indexed_chapters() -> List[str]:
    return retriever.list_indexed_chapters()


def list_indexed_lecture_numbers() -> List[int]:
    return retriever.list_indexed_lecture_numbers()
