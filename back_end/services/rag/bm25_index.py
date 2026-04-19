
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
import numpy as np
import re

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 index for keyword-based retrieval with disk persistence.
    Combines keyword matching with dense embeddings for hybrid RAG.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, persist_path: str = None):
        """
        Initialize BM25 index.
        
        Args:
            k1: Controls non-linear term frequency normalization (saturation point)
            b: Controls to what degree document length normalizes tf values
            persist_path: Optional path to persist the index to disk
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.metadata = []
        self.persist_path = persist_path

        # Auto-load from disk if persist_path exists
        if persist_path and Path(persist_path).exists():
            self.load_from_disk(persist_path)
        
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
        
        self.documents = list(documents)
        self.doc_ids = list(doc_ids)
        self.metadata = list(metadata) if metadata else [{} for _ in documents]
        
        logger.info(f"BM25 index built with {len(documents)} documents")

        # Auto-persist if path is set
        if self.persist_path:
            self.save_to_disk(self.persist_path)
    
    def add_documents(self, documents: List[str], doc_ids: List[str],
                     metadata: List[Dict] = None) -> None:
        """
        Add new documents to the index incrementally.
        Rebuilds the BM25 model with all documents (existing + new).
        
        Args:
            documents: New documents to add
            doc_ids: Document identifiers for new documents
            metadata: Optional metadata for new documents
        """
        if not documents:
            return

        # Skip documents that already exist (by doc_id)
        new_docs = []
        new_ids = []
        new_meta = []
        existing_ids = set(self.doc_ids)

        for i, doc_id in enumerate(doc_ids):
            if doc_id not in existing_ids:
                new_docs.append(documents[i])
                new_ids.append(doc_id)
                new_meta.append((metadata[i] if metadata else {}))

        if not new_docs:
            logger.debug("All documents already in BM25 index, skipping")
            return

        # Combine with existing documents
        all_documents = self.documents + new_docs
        all_doc_ids = self.doc_ids + new_ids
        all_metadata = self.metadata + new_meta
        
        self.build_index(all_documents, all_doc_ids, all_metadata)
        logger.info(f"BM25 index updated: added {len(new_docs)} new documents, "
                   f"total {len(all_documents)}")

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
        top_scores = [float(scores[i]) for i in top_indices]
        
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
        Tokenize text: lowercase, strip punctuation, split on whitespace.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Lowercase and strip common punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [token for token in text.split() if len(token) > 1]
    
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

    def save_to_disk(self, path: str = None) -> bool:
        """
        Persist the BM25 index to disk.
        
        Args:
            path: File path to save to (uses self.persist_path if None)
            
        Returns:
            True if saved successfully
        """
        save_path = path or self.persist_path
        if not save_path:
            logger.warning("No persist path specified for BM25 index")
            return False

        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "documents": self.documents,
                "doc_ids": self.doc_ids,
                "metadata": self.metadata,
                "k1": self.k1,
                "b": self.b,
            }
            with open(save_path, "wb") as f:
                pickle.dump(data, f)

            logger.info(f"BM25 index saved to {save_path} ({len(self.documents)} documents)")
            return True
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
            return False

    def load_from_disk(self, path: str = None) -> bool:
        """
        Load the BM25 index from disk.
        
        Args:
            path: File path to load from (uses self.persist_path if None)
            
        Returns:
            True if loaded successfully
        """
        load_path = path or self.persist_path
        if not load_path or not Path(load_path).exists():
            logger.warning(f"BM25 index file not found: {load_path}")
            return False

        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)

            self.k1 = data.get("k1", self.k1)
            self.b = data.get("b", self.b)

            documents = data["documents"]
            doc_ids = data["doc_ids"]
            metadata = data["metadata"]

            # Rebuild BM25 from loaded data (don't call build_index to avoid
            # re-persisting)
            tokenized_docs = [self._tokenize(doc) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
            self.documents = documents
            self.doc_ids = doc_ids
            self.metadata = metadata

            logger.info(f"BM25 index loaded from {load_path} ({len(documents)} documents)")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    @property
    def index_size(self) -> int:
        """Number of documents in the index."""
        return len(self.documents)

    @property
    def is_ready(self) -> bool:
        """Whether the index has been built and has documents."""
        return self.bm25 is not None and len(self.documents) > 0
