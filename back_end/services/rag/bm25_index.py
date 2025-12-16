
import logging
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 index for keyword-based retrieval.
    Combines keyword matching with dense embeddings for hybrid RAG.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.
        
        Args:
            k1: Controls non-linear term frequency normalization (saturation point)
            b: Controls to what degree document length normalizes tf values
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.metadata = []
        
    def build_index(self, documents: List[str], doc_ids: List[str], 
                   metadata: List[Dict] = None) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document texts to index
            doc_ids: List of document identifiers
            metadata: Optional list of metadata dicts for each document
        """
        if not documents:
            logger.warning("No documents provided for BM25 indexing")
            return
        
        if len(documents) != len(doc_ids):
            raise ValueError("Documents and doc_ids must have same length")
        
        # Tokenize documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        
        self.documents = documents
        self.doc_ids = doc_ids
        self.metadata = metadata if metadata else [{} for _ in documents]
        
        logger.info(f"BM25 index built with {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """
        Retrieve top-k documents using BM25 scoring.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            Tuple of (doc_ids, scores)
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built yet")
            return [], []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        top_doc_ids = [self.doc_ids[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        
        return top_doc_ids, top_scores
    
    def retrieve_with_metadata(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k documents with all metadata.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of result dicts with doc_id, score, text, and metadata
        """
        doc_ids, scores = self.retrieve(query, top_k)
        
        results = []
        for doc_id, score in zip(doc_ids, scores):
            idx = self.doc_ids.index(doc_id)
            results.append({
                'doc_id': doc_id,
                'score': float(score),
                'text': self.documents[idx],
                'metadata': self.metadata[idx]
            })
        
        return results
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text by splitting on whitespace and lowercasing.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        return text.lower().split()
    
    def update_index(self, documents: List[str], doc_ids: List[str],
                    metadata: List[Dict] = None) -> None:
        """
        Update index with new documents. Rebuilds entire index.
        
        Args:
            documents: New documents to add
            doc_ids: Document identifiers for new documents
            metadata: Optional metadata for new documents
        """
        # Combine with existing documents
        all_documents = self.documents + documents
        all_doc_ids = self.doc_ids + doc_ids
        all_metadata = self.metadata + (metadata if metadata else [{} for _ in documents])
        
        self.build_index(all_documents, all_doc_ids, all_metadata)
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        Get a document by its ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dict with document info or None if not found
        """
        if doc_id not in self.doc_ids:
            return None
        
        idx = self.doc_ids.index(doc_id)
        return {
            'doc_id': doc_id,
            'text': self.documents[idx],
            'metadata': self.metadata[idx]
        }
