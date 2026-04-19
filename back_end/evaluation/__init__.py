"""
WhoLM Evaluation Framework

Provides retrieval and generation evaluation metrics for the RAG pipeline.
"""

from .retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    RetrievalEvaluator
)

from .generation_metrics import (
    faithfulness_score,
    answer_relevance_score,
    GenerationEvaluator
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "RetrievalEvaluator",
    "faithfulness_score",
    "answer_relevance_score",
    "GenerationEvaluator",
]
