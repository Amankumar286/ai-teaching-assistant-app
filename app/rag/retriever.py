"""Dense retrieval over a persistent local Chroma collection.

Embeddings are computed explicitly via embeddings.py rather than through
Chroma's `embedding_function` hook: Chroma calls that hook identically for
both `add` and `query`, with no way to tell the two apart, but BGE's
asymmetric query/document prefixing needs exactly that distinction (see
embeddings.py). Precomputing vectors ourselves and passing them straight to
Chroma sidesteps the problem entirely.
"""

import hashlib
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CHROMA_DIR, DEFAULT_TOP_K

from app.rag.chunker import Chunk
from app.rag.embeddings import embed_documents, embed_query

_COLLECTION_NAME = "teaching_assistant"
_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def chunk_id(chunk: Chunk, index: int) -> str:
    """Stable id from (file, page/slide, position) so re-indexing the same
    file upserts in place instead of duplicating. Public so hybrid.py can
    compute the same ids once and share them with the BM25 store -- RRF
    fusion depends on both rankers agreeing on chunk identity."""
    key = f"{chunk.file_name}|{chunk.page_number}|{chunk.slide_number}|{index}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def index_chunks(chunks: List[Chunk], ids: Optional[List[str]] = None) -> int:
    if not chunks:
        return 0
    collection = _get_collection()
    ids = ids if ids is not None else [chunk_id(c, i) for i, c in enumerate(chunks)]
    documents = [c.text for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]
    embeddings = embed_documents(documents)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(chunks)


def query(text: str, top_k: int = DEFAULT_TOP_K, where: Optional[dict] = None) -> List[dict]:
    """Returns up to top_k hits as {"id", "text", "metadata", "score"} dicts,
    score a cosine similarity in roughly [0, 1] (higher = more relevant).
    Returns [] on an empty knowledge base rather than raising."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    result = collection.query(
        query_embeddings=[embed_query(text)],
        n_results=min(top_k, count),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    ids = result.get("ids") or [[]]
    docs = result.get("documents") or [[]]
    metas = result.get("metadatas") or [[]]
    dists = result.get("distances") or [[]]
    for hit_id, doc_text, meta, dist in zip(ids[0], docs[0], metas[0], dists[0]):
        hits.append({"id": hit_id, "text": doc_text, "metadata": meta, "score": 1.0 - dist})
    return hits


def clear_knowledge_base():
    """Backs the UI's 'Clear Knowledge Base' button."""
    global _client, _collection
    _get_collection()
    _client.delete_collection(name=_COLLECTION_NAME)
    _collection = None
    _get_collection()


def document_count() -> int:
    return _get_collection().count()


def list_indexed_files() -> List[str]:
    """Distinct file_name values currently indexed -- backs the UI's
    'Indexed Documents' list."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    result = collection.get(include=["metadatas"])
    names = {m["file_name"] for m in result.get("metadatas", []) if m.get("file_name")}
    return sorted(names)


def list_indexed_chapters() -> List[str]:
    """Distinct chapter values currently indexed -- used by summarize's
    reference resolution to match e.g. 'chapter 3' to an indexed chapter."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    result = collection.get(include=["metadatas"])
    chapters = {m["chapter"] for m in result.get("metadatas", []) if m.get("chapter")}
    return sorted(chapters)


def get_all_for_where(where: dict) -> List[dict]:
    """Returns every indexed chunk matching `where`, unranked by
    similarity -- used for whole-document/chapter sampling where genuine
    breadth of coverage matters more than similarity to a query string
    (see hybrid.sample_spread)."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    result = collection.get(where=where, include=["documents", "metadatas"])
    ids = result.get("ids", [])
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])
    return [{"id": i, "text": d, "metadata": m, "score": 1.0} for i, d, m in zip(ids, docs, metas)]


def list_indexed_lecture_numbers() -> List[int]:
    """Distinct lecture_number values currently indexed -- lets summarize
    match 'Lecture 8' to the right file by number rather than fragile
    filename-string similarity (e.g. 'Lecture_08.pptx' vs 'Lecture 8' don't
    share a clean substring or fuzzy-match reliably)."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    result = collection.get(include=["metadatas"])
    numbers = {m["lecture_number"] for m in result.get("metadatas", []) if m.get("lecture_number") is not None}
    return sorted(numbers)
