"""
Simplified retriever interface for backwards compatibility.
Uses the RAG pipeline internally.
"""
from typing import List, Dict, Optional
from .qdrant_rag_pipeline import QdrantRAGPipeline

# Global RAG pipeline instance (Qdrant-backed)
_rag_pipeline: Optional[QdrantRAGPipeline] = None


def initialize_rag(bm25_weight: float = 0.5, dense_weight: float = 0.5,
                   reranker_top_k: int = 3,
                   qdrant_url: str = "http://localhost:6333",
                   collection_name: str = "documents",
                   embedding_dim: int = 1024) -> QdrantRAGPipeline:
    """
    Initialize the RAG pipeline.
    
    Args:
        bm25_weight: Weight for BM25 in hybrid retriever
        dense_weight: Weight for dense embeddings
        reranker_top_k: Number of results after reranking
        
    Returns:
        RAGPipeline instance
    """
    global _rag_pipeline
    _rag_pipeline = QdrantRAGPipeline(
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        embedding_dim=embedding_dim,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        reranker_top_k=reranker_top_k,
    )
    return _rag_pipeline


def retrieve(query: str, top_k: int = 10) -> List[Dict]:
    """
    Simple retrieve function using global RAG pipeline.
    
    Args:
        query: Query text
        top_k: Number of candidates before reranking
        
    Returns:
        List of reranked results
    """
    if _rag_pipeline is None:
        raise RuntimeError("RAG pipeline not initialized. Call initialize_rag() first")
    
    return _rag_pipeline.query(query, retriever_top_k=top_k)


def get_pipeline() -> Optional[QdrantRAGPipeline]:
    """Get the global RAG pipeline instance."""
    return _rag_pipeline
