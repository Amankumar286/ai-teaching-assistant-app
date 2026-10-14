"""Real, deterministic pytest coverage for hybrid.rrf_fuse -- pure Reciprocal
Rank Fusion math, no models/network/index needed. Companion to the manual
scripts/test_hybrid.py, which eyeballs fused *relevance* on real documents;
this file locks down the fusion *formula* itself so a refactor can't
silently change how scores/ranks combine without a test noticing.
"""

from app.rag.hybrid import rrf_fuse

K = 60  # matches config.RRF_K


def hit(cid, **extra):
    return {"id": cid, "text": f"text-{cid}", "metadata": {}, **extra}


def test_empty_inputs_return_empty():
    assert rrf_fuse([], [], top_k=5) == []


def test_chunk_present_in_both_rankers_outscores_single_ranker_hit():
    dense = [hit("a"), hit("b")]
    sparse = [hit("b"), hit("c")]

    results = rrf_fuse(dense, sparse, top_k=3)
    ids = [r["id"] for r in results]

    # "b" is rank 2 in both lists: 1/(K+2) + 1/(K+2). "a" and "c" each only
    # score once (rank 1 in one list each: 1/(K+1)), so "b" must not win on
    # raw rank-1 status alone -- fusion, not either ranker, decides order.
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_hit_missing_from_one_ranker_is_not_penalized_to_zero():
    # "only_dense" appears solely in the dense list -- per rrf_fuse's own
    # docstring it should still be scored and returned (just without a BM25
    # contribution), not dropped or zeroed out.
    dense = [hit("only_dense")]
    sparse = []

    results = rrf_fuse(dense, sparse, top_k=5)

    assert len(results) == 1
    r = results[0]
    assert r["id"] == "only_dense"
    assert r["dense_rank"] == 1
    assert r["bm25_rank"] is None
    assert r["score"] == 1.0 / (K + 1)


def test_exact_score_matches_reciprocal_rank_fusion_formula():
    dense = [hit("x"), hit("y")]
    sparse = [hit("y"), hit("x")]

    results = rrf_fuse(dense, sparse, top_k=2)
    scores = {r["id"]: r["score"] for r in results}

    # x: dense rank 1, sparse rank 2. y: dense rank 2, sparse rank 1.
    assert scores["x"] == 1.0 / (K + 1) + 1.0 / (K + 2)
    assert scores["y"] == 1.0 / (K + 2) + 1.0 / (K + 1)
    # Symmetric ranks -> tied score -> both present, order between ties is
    # whatever sorted() gives (stable), just confirm neither is dropped.
    assert set(scores) == {"x", "y"}


def test_top_k_truncates_after_fusion_not_before():
    # Both rankers agree "c" is worst, but each ranker's OWN top_k might
    # have already excluded it upstream -- rrf_fuse itself must respect
    # top_k on the fused/combined result, not assume its inputs are already
    # sized correctly.
    dense = [hit("a"), hit("b"), hit("c")]
    sparse = [hit("a"), hit("b"), hit("c")]

    results = rrf_fuse(dense, sparse, top_k=2)

    assert len(results) == 2
    assert {r["id"] for r in results} == {"a", "b"}


def test_custom_k_constant_changes_scores():
    dense = [hit("a")]
    sparse = []

    default_k_score = rrf_fuse(dense, sparse, top_k=1)[0]["score"]
    custom_k_score = rrf_fuse(dense, sparse, top_k=1, k=1)[0]["score"]

    assert default_k_score == 1.0 / (K + 1)
    assert custom_k_score == 1.0 / (1 + 1)
    assert custom_k_score > default_k_score
