"""
Retrieval evaluation metrics for the WhoLM RAG pipeline.

Computes Precision@K, Recall@K, MRR, and NDCG@K over a set of queries
with known relevant document IDs.
"""

import logging
import math
from typing import List, Dict, Any, Set, Union

logger = logging.getLogger(__name__)


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Precision@K: fraction of top-k retrieved documents that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of ground-truth relevant document IDs
        k: Cutoff rank

    Returns:
        Precision score in [0, 1]
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Recall@K: fraction of relevant documents that appear in top-k.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of ground-truth relevant document IDs
        k: Cutoff rank

    Returns:
        Recall score in [0, 1]
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Mean Reciprocal Rank (MRR): 1/rank of the first relevant document.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of ground-truth relevant document IDs

    Returns:
        MRR score in [0, 1]
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    Assumes binary relevance (1 if relevant, 0 otherwise).

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of ground-truth relevant document IDs
        k: Cutoff rank

    Returns:
        NDCG score in [0, 1]
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because rank starts at 1, log2(1)=0

    # Ideal DCG: all relevant docs ranked at the top
    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Hit Rate@K: 1 if any relevant document appears in top-k, 0 otherwise.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of ground-truth relevant document IDs
        k: Cutoff rank

    Returns:
        1.0 or 0.0
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & relevant_ids else 0.0


class RetrievalEvaluator:
    """
    Evaluates retrieval quality across a dataset of queries with known relevant documents.
    
    Usage:
        evaluator = RetrievalEvaluator(k_values=[3, 5, 10])
        
        for query_data in eval_dataset:
            retrieved = rag_pipeline.query(query_data["question"])
            retrieved_ids = [r["doc_id"] for r in retrieved]
            relevant_ids = set(query_data["relevant_doc_ids"])
            evaluator.add_result(retrieved_ids, relevant_ids)
        
        metrics = evaluator.compute_metrics()
    """

    def __init__(self, k_values: List[int] = None):
        """
        Args:
            k_values: List of K values for Precision@K, Recall@K, NDCG@K
        """
        self.k_values = k_values or [3, 5, 10]
        self.results: List[Dict[str, Any]] = []

    def add_result(self, retrieved_ids: List[str], relevant_ids: Union[Set[str], List[str]],
                   query: str = None) -> Dict[str, float]:
        """
        Add a single query result for evaluation.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set or list of ground-truth relevant document IDs
            query: Optional query text for logging

        Returns:
            Dict of per-query metrics
        """
        if isinstance(relevant_ids, list):
            relevant_ids = set(relevant_ids)

        metrics = {"query": query or f"query_{len(self.results)}"}

        # Compute metrics for each K
        for k in self.k_values:
            metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_ids, relevant_ids, k)

        metrics["mrr"] = mean_reciprocal_rank(retrieved_ids, relevant_ids)
        metrics["num_retrieved"] = len(retrieved_ids)
        metrics["num_relevant"] = len(relevant_ids)

        self.results.append(metrics)
        return metrics

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute averaged metrics across all queries.

        Returns:
            Dict with averaged metric values
        """
        if not self.results:
            logger.warning("No results to evaluate")
            return {}

        n = len(self.results)
        avg_metrics = {"num_queries": n}

        # Aggregate all numeric metrics
        metric_keys = [k for k in self.results[0].keys()
                      if k not in ("query", "num_retrieved", "num_relevant")]

        for key in metric_keys:
            values = [r[key] for r in self.results]
            avg_metrics[f"avg_{key}"] = sum(values) / n

        # Add totals
        avg_metrics["avg_num_retrieved"] = sum(r["num_retrieved"] for r in self.results) / n
        avg_metrics["avg_num_relevant"] = sum(r["num_relevant"] for r in self.results) / n

        return avg_metrics

    def get_per_query_results(self) -> List[Dict[str, Any]]:
        """Get individual query results for detailed analysis."""
        return self.results

    def reset(self):
        """Clear all stored results."""
        self.results = []

    def summary_table(self) -> str:
        """
        Generate a formatted summary table of evaluation results.

        Returns:
            Formatted string table
        """
        metrics = self.compute_metrics()
        if not metrics:
            return "No evaluation results."

        lines = [
            "=" * 50,
            "  RETRIEVAL EVALUATION RESULTS",
            "=" * 50,
            f"  Queries evaluated: {metrics['num_queries']}",
            f"  Avg retrieved:     {metrics['avg_num_retrieved']:.1f}",
            f"  Avg relevant:      {metrics['avg_num_relevant']:.1f}",
            "-" * 50,
        ]

        for k in self.k_values:
            lines.append(f"  Precision@{k}:  {metrics.get(f'avg_precision@{k}', 0):.4f}")
            lines.append(f"  Recall@{k}:     {metrics.get(f'avg_recall@{k}', 0):.4f}")
            lines.append(f"  NDCG@{k}:       {metrics.get(f'avg_ndcg@{k}', 0):.4f}")
            lines.append(f"  Hit Rate@{k}:   {metrics.get(f'avg_hit_rate@{k}', 0):.4f}")
            lines.append("-" * 50)

        lines.append(f"  MRR:            {metrics.get('avg_mrr', 0):.4f}")
        lines.append("=" * 50)

        return "\n".join(lines)
