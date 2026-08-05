import pytest

from retrieval.retrieve import hybrid_retrieve, reciprocal_rank_fusion


def test_rank_based_not_score_based():
    # reciprocal_rank_fusion only ever sees ordered doc_id lists, never raw scores —
    # swapping which doc is rank 1 vs rank 2 must flip the fused order accordingly.
    fused_a = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    fused_b = reciprocal_rank_fusion([["b", "a", "c"]], k=60)
    assert [doc_id for doc_id, _ in fused_a] == ["a", "b", "c"]
    assert [doc_id for doc_id, _ in fused_b] == ["b", "a", "c"]


def test_k_60_hand_computed_scores():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 63)


def test_duplicate_document_accumulates_both_sources():
    fused = dict(reciprocal_rank_fusion([["a"], ["a"]], k=60))
    assert fused["a"] == pytest.approx(2 / 61)


def test_document_present_in_only_one_source_still_appears():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["a"]], k=60))
    assert set(fused) == {"a", "b"}
    assert fused["b"] == pytest.approx(1 / 62)


def test_tie_break_is_deterministic_by_doc_id():
    # "x" and "y" each appear only once, both at rank 1 of a single-element list,
    # in separate fusion calls -> identical fused score -> break tie by doc_id ascending.
    fused = reciprocal_rank_fusion([["y"], ["x"]], k=60)
    assert fused == [("x", pytest.approx(1 / 61)), ("y", pytest.approx(1 / 61))]

    fused_again = reciprocal_rank_fusion([["y"], ["x"]], k=60)
    assert fused == fused_again


def test_hybrid_retrieve_schema_and_fusion():
    bm25_rows = [
        {"query_id": "q1", "doc_id": "a", "rank": 1, "score": 5.0},
        {"query_id": "q1", "doc_id": "b", "rank": 2, "score": 3.0},
        {"query_id": "q2", "doc_id": "c", "rank": 1, "score": 4.0},
    ]
    dense_rows = [
        {"query_id": "q1", "doc_id": "b", "rank": 1, "score": 0.9},
        {"query_id": "q1", "doc_id": "a", "rank": 2, "score": 0.8},
        {"query_id": "q2", "doc_id": "d", "rank": 1, "score": 0.7},
    ]

    rows = hybrid_retrieve(bm25_rows, dense_rows, k=60)

    q1_rows = sorted((r for r in rows if r["query_id"] == "q1"), key=lambda r: r["rank"])
    assert [r["doc_id"] for r in q1_rows] == ["a", "b"]
    assert [r["rank"] for r in q1_rows] == [1, 2]
    expected_a = dict(reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60))["a"]
    assert q1_rows[0]["score"] == pytest.approx(expected_a)

    q2_rows = sorted((r for r in rows if r["query_id"] == "q2"), key=lambda r: r["rank"])
    assert [r["doc_id"] for r in q2_rows] == ["c", "d"]
