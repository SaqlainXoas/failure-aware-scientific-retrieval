"""Ranking metrics: Recall@k, MRR@10, nDCG@10.

Shared semantics: relevance means grade > 0 (grade 0 is never treated as
relevant); ranks are one-indexed; k must be positive; nDCG uses graded
relevance for gain, not just membership; a query with no positive qrels
returns 0.0 for every metric.
"""

import math


def recall_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k ranked list; 0.0 if there are none."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not relevant_doc_ids:
        return 0.0
    hits = len(set(ranked_doc_ids[:k]) & relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def mrr_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int = 10) -> float:
    """Reciprocal rank of the first relevant doc within the top-k, else 0.0."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_doc_ids: list[str], qrels_for_query: dict[str, int], k: int = 10) -> float:
    """Standard nDCG@k using graded relevance from qrels (positive grades only contribute gain)."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    dcg = sum(
        qrels_for_query.get(doc_id, 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1)
    )
    ideal_gains = sorted((grade for grade in qrels_for_query.values() if grade > 0), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal_gains, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(ranked_doc_ids: list[str], qrels_for_query: dict[str, int]) -> dict[str, float]:
    """Computes Recall@5/10/50, MRR@10, and nDCG@10 for one query's ranking."""
    relevant_doc_ids = {doc_id for doc_id, grade in qrels_for_query.items() if grade > 0}
    return {
        "recall@5": recall_at_k(ranked_doc_ids, relevant_doc_ids, 5),
        "recall@10": recall_at_k(ranked_doc_ids, relevant_doc_ids, 10),
        "recall@50": recall_at_k(ranked_doc_ids, relevant_doc_ids, 50),
        "mrr@10": mrr_at_k(ranked_doc_ids, relevant_doc_ids, 10),
        "ndcg@10": ndcg_at_k(ranked_doc_ids, qrels_for_query, 10),
    }
