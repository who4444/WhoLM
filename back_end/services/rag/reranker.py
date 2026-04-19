import torch
import logging
from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)

# Auto-detect device — don't crash on CPU-only machines
_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
_use_fp16 = _device != 'cpu'

logger.info(f"Reranker using device: {_device}, fp16: {_use_fp16}")

_reranker_model = None

def _get_reranker():
    """Lazy-load the reranker model to avoid startup cost and allow CPU fallback."""
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"Loading reranker model on {_device}...")
        _reranker_model = FlagReranker(
            'BAAI/bge-reranker-v2-m3',
            devices=_device,
            use_fp16=_use_fp16
        )
        logger.info("Reranker model loaded successfully")
    return _reranker_model


def reranker(query, contexts, top_k):
    """
    Rerank contexts by relevance to query using cross-encoder.
    
    Args:
        query: The search query
        contexts: List of candidate text passages
        top_k: Number of top results to return
        
    Returns:
        List of top-k contexts sorted by relevance
    """
    if not contexts:
        return []

    model = _get_reranker()
    pairs = [(query, context) for context in contexts]

    scores = model.compute_score(pairs)

    # compute_score returns a single float if only one pair
    if isinstance(scores, (int, float)):
        scores = [scores]

    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_k_contexts = [contexts[i] for i in top_k_indices]
    return top_k_contexts
