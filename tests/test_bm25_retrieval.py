"""Real, deterministic pytest coverage for hybrid.tokenize and
hybrid.BM25Store -- pure keyword search via rank_bm25, no embeddings/Chroma
needed (app.rag.retriever/embeddings only import those lazily inside
functions we never call here, per their own source). Companion to the
manual scripts/test_retriever.py, which eyeballs *dense* relevance on real
documents; this file locks down the deterministic BM25 side.

Uses a throwaway BM25Store() instance with BM25_INDEX_PATH redirected to a
tmp_path, so this never touches the app's real persisted index.
"""

import pytest

from app.rag import hybrid
from app.rag.chunker import Chunk


def make_chunk(text, file_name="doc.pdf", **extra):
    return Chunk(text=text, file_name=file_name, file_type="pdf", **extra)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(hybrid, "BM25_INDEX_PATH", tmp_path / "bm25_index.pkl")
    return hybrid.BM25Store()


# --- tokenize(): Uzbek apostrophe normalization -----------------------------

def test_tokenize_normalizes_all_apostrophe_variants_identically():
    variants = ["o'zbek", "o‘zbek", "o’zbek", "oʻzbek"]
    tokenized = [hybrid.tokenize(v) for v in variants]

    assert all(tokens == tokenized[0] for tokens in tokenized), tokenized
    assert tokenized[0] == ["oʻzbek"]  # canonical form


def test_tokenize_empty_string_returns_empty_list():
    assert hybrid.tokenize("") == []
    assert hybrid.tokenize(None) == []


def test_tokenize_is_case_insensitive():
    assert hybrid.tokenize("Kiberxavfsizlik") == hybrid.tokenize("kiberxavfsizlik")


# --- BM25Store: indexing + search -------------------------------------------

def test_search_ranks_matching_chunk_above_unrelated_ones(store):
    chunks = [
        make_chunk("Shifrlash asoslari va kriptografik algoritmlar"),
        make_chunk("Atmosfera qatlamlari va havo bosimi"),
        make_chunk("Kompyuter tarmoqlari va protokollar"),
    ]
    store.add_chunks(chunks, ids=["c0", "c1", "c2"])

    hits = store.search("shifrlash algoritmlar", top_k=3)

    assert hits, "expected at least one match"
    assert hits[0]["id"] == "c0"


def test_search_with_no_vocabulary_overlap_returns_no_hits(store):
    # Per BM25Store.search's own comment: a score of ~0 (shares no
    # discriminating term with the query) must NOT be ranked as a real hit,
    # even though every entry technically gets *some* score from rank_bm25.
    store.add_chunks(
        [make_chunk("Shifrlash asoslari va kriptografik algoritmlar")],
        ids=["c0"],
    )

    hits = store.search("kvant kompyuterlari blokcheyn", top_k=3)

    assert hits == []


def test_search_respects_top_k(store):
    # 3 chunks share the query terms, 2 are unrelated distractors -- without
    # the distractors, EVERY corpus document would share the query's exact
    # vocabulary, driving BM25's idf negative for those terms (a real
    # rank_bm25 property: a term present in the whole tiny corpus carries no
    # discriminating power) and producing zero hits regardless of top_k.
    matching = [make_chunk(f"takrorlanuvchi so'z mavzu raqam {i}") for i in range(3)]
    distractors = [make_chunk("ob-havo bashorati"), make_chunk("moliyaviy hisobot")]
    store.add_chunks(matching + distractors, ids=[f"m{i}" for i in range(3)] + ["d0", "d1"])

    hits = store.search("takrorlanuvchi so'z mavzu", top_k=2)

    assert len(hits) == 2
    assert all(h["id"].startswith("m") for h in hits)


def test_search_where_filter_narrows_to_matching_metadata(store):
    chunks = [
        make_chunk("mashina o'rganish asoslari", file_name="ml.pdf"),
        make_chunk("mashina o'rganish algoritmlari", file_name="dl.pdf"),
        make_chunk("ob-havo bashorati usullari", file_name="wx.pdf"),  # distractor, see above
    ]
    store.add_chunks(chunks, ids=["c0", "c1", "c2"])

    hits = store.search("mashina o'rganish", top_k=5, where={"file_name": "ml.pdf"})

    assert [h["id"] for h in hits] == ["c0"]


def test_add_chunks_with_repeated_id_overwrites_not_duplicates(store):
    # 2 distractors, not 1: with a term unique to just 1 of only 2 total
    # corpus documents, df=1 of N=2 sits exactly on this idf formula's
    # zero-crossing (log(N-df+0.5)-log(df+0.5) with N=2,df=1 is exactly 0),
    # which the store's own `> 1e-9` guard then filters out as "no hits" --
    # a 3rd document pushes it cleanly positive.
    store.add_chunks(
        [make_chunk("ob-havo bashorati"), make_chunk("moliyaviy hisobot")],
        ids=["distractor1", "distractor2"],
    )
    store.add_chunks([make_chunk("birinchi versiya")], ids=["c0"])
    store.add_chunks([make_chunk("ikkinchi versiya")], ids=["c0"])

    assert store.count() == 3  # 2 distractors + c0, not 4 -- c0 was overwritten
    hits = store.search("ikkinchi versiya", top_k=1)
    assert hits[0]["text"] == "ikkinchi versiya"


def test_clear_empties_the_store(store):
    store.add_chunks([make_chunk("bir narsa")], ids=["c0"])
    assert store.count() == 1

    store.clear()

    assert store.count() == 0
    assert store.search("bir narsa", top_k=5) == []


def test_persists_across_store_instances(tmp_path, monkeypatch):
    index_path = tmp_path / "bm25_index.pkl"
    monkeypatch.setattr(hybrid, "BM25_INDEX_PATH", index_path)

    first = hybrid.BM25Store()
    first.add_chunks(
        # 2 distractors, not 1 -- see the df=1-of-N=2 zero-crossing note above.
        [make_chunk("saqlangan ma'lumot"), make_chunk("ob-havo bashorati"), make_chunk("moliyaviy hisobot")],
        ids=["c0", "distractor1", "distractor2"],
    )

    second = hybrid.BM25Store()  # fresh instance, same redirected path
    assert second.count() == 3
    assert second.search("saqlangan ma'lumot", top_k=1)[0]["id"] == "c0"
