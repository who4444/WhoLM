
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from .bm25_index import BM25Index

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and dense embeddings.
    Scores results from both methods and merges them using weighted combination.
    """
    
    def __init__(self, bm25_weight: float = 0.5, dense_weight: float = 0.5):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_weight: Weight for BM25 scores (0-1)
            dense_weight: Weight for dense embedding scores (0-1)
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bm25_index = BM25Index()
        self.embeddings = {}
        self.embedding_dim = None
        
        # Normalize weights
        total_weight = bm25_weight + dense_weight
        self.bm25_weight = bm25_weight / total_weight
        self.dense_weight = dense_weight / total_weight
        
    def build_index(self, documents: List[str], embeddings: np.ndarray,
                   doc_ids: List[str], metadata: List[Dict] = None) -> None:
        """
        Build hybrid index with both BM25 and dense embeddings.
        
        Args:
            documents: List of document texts
            embeddings: Dense embeddings array (n_docs, embedding_dim)
            doc_ids: Document identifiers
            metadata: Optional metadata for each document
        """
        if len(documents) != len(embeddings) or len(documents) != len(doc_ids):
            raise ValueError("Documents, embeddings, and doc_ids must have same length")
        
        # Build BM25 index
        self.bm25_index.build_index(documents, doc_ids, metadata)
        
        # Store embeddings
        self.embedding_dim = embeddings.shape[1]
        for doc_id, embedding in zip(doc_ids, embeddings):
            self.embeddings[doc_id] = embedding / np.linalg.norm(embedding)
        
        logger.info(f"Hybrid index built with {len(documents)} documents, "
                   f"embedding_dim={self.embedding_dim}")
    
    def retrieve(self, query: str, query_embedding: np.ndarray, 
                top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k results using hybrid approach.
        
        Args:
            query: Query text
            query_embedding: Dense embedding for query
            top_k: Number of results to return
            
        Returns:
            List of result dicts with combined scores
        """
        # BM25 retrieval
        bm25_doc_ids, bm25_scores = self.bm25_index.retrieve(query, top_k * 2)
        
        # Dense retrieval
        dense_results = self._dense_retrieve(query_embedding, top_k * 2)
        
        # Combine results
        combined_scores = self._combine_scores(bm25_doc_ids, bm25_scores, 
                                               dense_results)
        
        # Get top-k from combined scores
        sorted_doc_ids = sorted(combined_scores.keys(), 
                               key=lambda x: combined_scores[x], 
                               reverse=True)[:top_k]
        
        results = []
        for doc_id in sorted_doc_ids:
            doc = self.bm25_index.get_document(doc_id)
            if doc:
                doc['hybrid_score'] = combined_scores[doc_id]
                doc['bm25_score'] = combined_scores.get(f"{doc_id}_bm25", 0)
                doc['dense_score'] = combined_scores.get(f"{doc_id}_dense", 0)
                results.append(doc)
        
        return results
    
    def _dense_retrieve(self, query_embedding: np.ndarray, 
                       top_k: int) -> Dict[str, Tuple[str, float]]:
        """
        Dense embedding-based retrieval using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Dict mapping doc_id to (doc_id, score)
        """
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        scores = {}
        for doc_id, embedding in self.embeddings.items():
            # Cosine similarity
            score = np.dot(query_embedding, embedding)
            scores[doc_id] = score
        
        # Get top-k
        top_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        
        return {doc_id: scores[doc_id] for doc_id in top_ids}
    
    def _combine_scores(self, bm25_doc_ids: List[str], bm25_scores: List[float],
                       dense_results: Dict[str, float]) -> Dict[str, float]:
        """
        Combine BM25 and dense scores using weighted combination and min-max normalization.
        
        Args:
            bm25_doc_ids: Document IDs from BM25
            bm25_scores: BM25 scores
            dense_results: Dense retrieval results (doc_id -> score)
            
        Returns:
            Dict of combined scores
        """
        combined = {}
        
        # Normalize BM25 scores
        if bm25_scores:
            bm25_max = max(bm25_scores) if bm25_scores else 1
            bm25_min = min(bm25_scores) if bm25_scores else 0
            bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1
        
        # Normalize dense scores
        if dense_results:
            dense_scores = list(dense_results.values())
            dense_max = max(dense_scores) if dense_scores else 1
            dense_min = min(dense_scores) if dense_scores else 0
            dense_range = dense_max - dense_min if dense_max != dense_min else 1
        
        # Add BM25 scores
        for doc_id, score in zip(bm25_doc_ids, bm25_scores):
            normalized_bm25 = (score - bm25_min) / bm25_range if bm25_range > 0 else 0
            combined[doc_id] = normalized_bm25 * self.bm25_weight
        
        # Add dense scores
        for doc_id, score in dense_results.items():
            normalized_dense = (score - dense_min) / dense_range if dense_range > 0 else 0
            if doc_id in combined:
                combined[doc_id] += normalized_dense * self.dense_weight
            else:
                combined[doc_id] = normalized_dense * self.dense_weight
        
        return combined
    
    def update_index(self, documents: List[str], embeddings: np.ndarray,
                    doc_ids: List[str], metadata: List[Dict] = None) -> None:
        """
        Update index with new documents.
        
        Args:
            documents: New documents to add
            embeddings: Dense embeddings for new documents
            doc_ids: Document identifiers
            metadata: Optional metadata
        """
        # Update BM25 index
        self.bm25_index.update_index(documents, doc_ids, metadata)
        
        # Update embeddings
        for doc_id, embedding in zip(doc_ids, embeddings):
            self.embeddings[doc_id] = embedding / np.linalg.norm(embedding)
        
        logger.info(f"Hybrid index updated, now contains "
                   f"{len(self.bm25_index.doc_ids)} total documents")
    
    def set_weights(self, bm25_weight: float, dense_weight: float) -> None:
        """
        Update the weights for combining BM25 and dense scores.
        
        Args:
            bm25_weight: New weight for BM25 (0-1)
            dense_weight: New weight for dense embeddings (0-1)
        """
        total = bm25_weight + dense_weight
        self.bm25_weight = bm25_weight / total
        self.dense_weight = dense_weight / total
        logger.info(f"Weights updated: BM25={self.bm25_weight:.3f}, "
                   f"Dense={self.dense_weight:.3f}")
