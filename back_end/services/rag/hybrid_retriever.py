
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from .bm25_index import BM25Index

logger = logging.getLogger(__name__)

# Default persist directory for BM25 index
DEFAULT_BM25_PERSIST_PATH = "bm25_data/bm25_index.pkl"


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and dense embeddings.
    Scores results from both methods and merges them using weighted combination.
    Supports disk persistence for the BM25 index.
    """
    
    def __init__(self, bm25_weight: float = 0.5, dense_weight: float = 0.5,
                 persist_path: str = None):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_weight: Weight for BM25 scores (0-1)
            dense_weight: Weight for dense embedding scores (0-1)
            persist_path: Path for BM25 index persistence (auto-loads if exists)
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.embeddings = {}
        self.embedding_dim = None

        # Normalize weights
        total_weight = bm25_weight + dense_weight
        self.bm25_weight = bm25_weight / total_weight
        self.dense_weight = dense_weight / total_weight

        # Initialize BM25 with persistence
        self.persist_path = persist_path or DEFAULT_BM25_PERSIST_PATH
        self.bm25_index = BM25Index(persist_path=self.persist_path)

        logger.info(f"HybridRetriever initialized (BM25={self.bm25_weight:.2f}, "
                    f"Dense={self.dense_weight:.2f}, "
                    f"BM25 docs={self.bm25_index.index_size})")
        
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
        
        # Build BM25 index (auto-persists)
        self.bm25_index.build_index(documents, doc_ids, metadata)
        
        # Store embeddings
        self.embedding_dim = embeddings.shape[1]
        for doc_id, embedding in zip(doc_ids, embeddings):
            norm = np.linalg.norm(embedding)
            if norm > 0:
                self.embeddings[doc_id] = embedding / norm
            else:
                self.embeddings[doc_id] = embedding
        
        logger.info(f"Hybrid index built with {len(documents)} documents, "
                   f"embedding_dim={self.embedding_dim}")

    def add_documents(self, documents: List[str], doc_ids: List[str],
                     metadata: List[Dict] = None) -> None:
        """
        Add new documents to the BM25 index incrementally.
        Only updates BM25 (dense embeddings are stored in Qdrant).
        
        Args:
            documents: New document texts
            doc_ids: Document identifiers
            metadata: Optional metadata for each document
        """
        self.bm25_index.add_documents(documents, doc_ids, metadata)
    
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
        
        # Dense retrieval (only if embeddings are stored locally)
        dense_results = {}
        if self.embeddings:
            dense_results = self._dense_retrieve(query_embedding, top_k * 2)
        
        # Combine results
        combined_scores = self._combine_scores(bm25_doc_ids, bm25_scores, 
                                               dense_results)
        
        # Get top-k from combined scores
        # Filter out internal score tracking keys (those with _bm25 or _dense suffix)
        score_keys = [k for k in combined_scores.keys() 
                     if not k.endswith("_bm25") and not k.endswith("_dense")]
        sorted_doc_ids = sorted(score_keys, 
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

    def bm25_only_retrieve(self, query: str, top_k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Retrieve using BM25 only (no dense embeddings needed).
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            Tuple of (doc_ids, scores)
        """
        return self.bm25_index.retrieve(query, top_k)
    
    def _dense_retrieve(self, query_embedding: np.ndarray, 
                       top_k: int) -> Dict[str, float]:
        """
        Dense embedding-based retrieval using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Dict mapping doc_id to score
        """
        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        scores = {}
        for doc_id, embedding in self.embeddings.items():
            # Cosine similarity
            score = float(np.dot(query_embedding, embedding))
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
            Dict of combined scores (includes _bm25 and _dense suffixed keys for tracking)
        """
        combined = {}
        
        # Normalize BM25 scores
        bm25_min, bm25_max, bm25_range = 0, 1, 1
        if bm25_scores:
            bm25_max = max(bm25_scores)
            bm25_min = min(bm25_scores)
            bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1
        
        # Normalize dense scores
        dense_min, dense_max, dense_range = 0, 1, 1
        if dense_results:
            dense_scores_list = list(dense_results.values())
            dense_max = max(dense_scores_list)
            dense_min = min(dense_scores_list)
            dense_range = dense_max - dense_min if dense_max != dense_min else 1
        
        # Add BM25 scores
        for doc_id, score in zip(bm25_doc_ids, bm25_scores):
            normalized_bm25 = (score - bm25_min) / bm25_range if bm25_range > 0 else 0
            combined[doc_id] = normalized_bm25 * self.bm25_weight
            combined[f"{doc_id}_bm25"] = normalized_bm25
        
        # Add dense scores
        for doc_id, score in dense_results.items():
            normalized_dense = (score - dense_min) / dense_range if dense_range > 0 else 0
            if doc_id in combined:
                combined[doc_id] += normalized_dense * self.dense_weight
            else:
                combined[doc_id] = normalized_dense * self.dense_weight
            combined[f"{doc_id}_dense"] = normalized_dense
        
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
            norm = np.linalg.norm(embedding)
            if norm > 0:
                self.embeddings[doc_id] = embedding / norm
            else:
                self.embeddings[doc_id] = embedding
        
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

    def save(self) -> bool:
        """Save the BM25 index to disk."""
        return self.bm25_index.save_to_disk()

    def load(self) -> bool:
        """Load the BM25 index from disk."""
        return self.bm25_index.load_from_disk()

    @property
    def bm25_ready(self) -> bool:
        """Whether the BM25 index is populated and ready for queries."""
        return self.bm25_index.is_ready

    @property
    def bm25_doc_count(self) -> int:
        """Number of documents in the BM25 index."""
        return self.bm25_index.index_size
