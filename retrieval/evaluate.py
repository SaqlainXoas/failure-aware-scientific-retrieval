"""Ranking metrics: Recall@k, MRR@10, nDCG@10.

Shared semantics: relevance means grade > 0 (grade 0 is never treated as
relevant); ranks are one-indexed; k must be positive; nDCG uses graded
relevance for gain, not just membership; a query with no positive qrels
returns 0.0 for every metric.
"""

import math
from typing import Any


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


CATEGORIES = [
    "already_successful",
    "rescued_by_reranker",
    "degraded_by_reranker",
    "unchanged_failure",
    "no_opportunity",
]


def candidate_success_50(gold_doc_ids: set[str], candidate_doc_ids: list[str]) -> bool:
    """True iff gold(q) intersects the top-50 candidate set."""
    return bool(gold_doc_ids & set(candidate_doc_ids))


def final_success_10(gold_doc_ids: set[str], ranked_doc_ids: list[str]) -> bool:
    """True iff gold(q) intersects the given top-10 ranking."""
    return bool(gold_doc_ids & set(ranked_doc_ids))


def label_transition(
    gold_doc_ids: set[str],
    first_stage_ranked: list[str],
    reranked_ranked: list[str],
    candidate_depth: int = 50,
) -> str:
    """Assigns one of the five transition categories.

    The no-candidate check is the first and only exit for candidate_success_50=0, so a
    candidate-set failure can never be mislabeled as a reranker rescue.
    """
    if not candidate_success_50(gold_doc_ids, first_stage_ranked[:candidate_depth]):
        return "no_opportunity"
    in_first_10 = final_success_10(gold_doc_ids, first_stage_ranked[:10])
    in_reranked_10 = final_success_10(gold_doc_ids, reranked_ranked[:10])
    if in_first_10 and in_reranked_10:
        return "already_successful"
    if not in_first_10 and in_reranked_10:
        return "rescued_by_reranker"
    if in_first_10 and not in_reranked_10:
        return "degraded_by_reranker"
    return "unchanged_failure"


def decomposition_metrics(labels: list[str]) -> dict[str, Any]:
    """Aggregates failure-decomposition rates from per-query transition labels.

    Every rate except the two below is denominated over *all* queries in the split:
    candidate_set_failure_rate / reranking_failure_rate / final_success_rate partition them
    exactly. The two conditional rates are named for their denominator —
    conditional_conversion_rate is final success among queries with a candidate-set opportunity,
    and share_of_failures_from_candidate_set is the fraction of *final failures* that the
    reranker never had a chance to fix — the headline quantity of the study. Both return 0.0
    rather than raising when their denominator is empty.

    Raw per-category counts and n_queries are returned alongside the rates so the
    "transitions sum to the query count" invariant is checkable from the saved artifact.
    """
    unknown = sorted(set(labels) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown transition label(s): {unknown}")

    n = len(labels)
    counts = {category: labels.count(category) for category in CATEGORIES}
    n_opportunity = n - counts["no_opportunity"]
    n_final_success = counts["already_successful"] + counts["rescued_by_reranker"]
    n_rerank_failure = counts["degraded_by_reranker"] + counts["unchanged_failure"]
    n_final_failure = counts["no_opportunity"] + n_rerank_failure

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "candidate_set_failure_rate": rate(counts["no_opportunity"], n),
        "reranking_failure_rate": rate(n_rerank_failure, n),
        "final_success_rate": rate(n_final_success, n),
        "opportunity_rate": rate(n_opportunity, n),
        "rescue_rate": rate(counts["rescued_by_reranker"], n),
        "degradation_rate": rate(counts["degraded_by_reranker"], n),
        "conditional_conversion_rate": rate(n_final_success, n_opportunity),
        "share_of_failures_from_candidate_set": rate(counts["no_opportunity"], n_final_failure),
        "counts": counts,
        "n_queries": n,
        "n_final_failures": n_final_failure,
    }
