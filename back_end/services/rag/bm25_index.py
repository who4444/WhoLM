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
    Uses dirty-flag batching: incremental adds are buffered and only trigger
    a full rebuild when the pending count exceeds the batch threshold or a
    retrieval is requested.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, persist_path: str = None,
                 batch_threshold: int = 50):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.metadata: List[Dict] = []
        self.persist_path = persist_path
        self.batch_threshold = batch_threshold

        # Pending buffer for incremental adds — avoids O(n) rebuild per add
        self._pending_docs: List[str] = []
        self._pending_ids: List[str] = []
        self._pending_meta: List[Dict] = []
        self._dirty = False

        if persist_path and Path(persist_path).exists():
            self.load_from_disk(persist_path)

    def _ensure_fresh(self):
        """Rebuild BM25 model if there are pending documents not yet indexed."""
        if self._dirty and self._pending_ids:
            all_docs = self.documents + self._pending_docs
            all_ids = self.doc_ids + self._pending_ids
            all_meta = self.metadata + self._pending_meta
            self._build_internal(all_docs, all_ids, all_meta)
            logger.debug(f"BM25 index rebuilt with {len(self._pending_ids)} new docs "
                         f"(total: {len(all_docs)})")

    def _build_internal(self, documents: List[str], doc_ids: List[str],
                        metadata: List[Dict]) -> None:
        """Internal build without triggering persistence."""
        tokenized = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self.documents = list(documents)
        self.doc_ids = list(doc_ids)
        self.metadata = list(metadata)
        self._pending_docs.clear()
        self._pending_ids.clear()
        self._pending_meta.clear()
        self._dirty = False

    def build_index(self, documents: List[str], doc_ids: List[str],
                   metadata: List[Dict] = None) -> None:
        """Build BM25 index from documents (full rebuild, clears pending)."""
        if not documents:
            logger.warning("No documents provided for BM25 indexing")
            return

        if len(documents) != len(doc_ids):
            raise ValueError("Documents and doc_ids must have same length")

        meta = list(metadata) if metadata else [{} for _ in documents]
        self._build_internal(documents, doc_ids, meta)

        logger.info(f"BM25 index built with {len(documents)} documents")

        if self.persist_path:
            self.save_to_disk(self.persist_path)

    def add_documents(self, documents: List[str], doc_ids: List[str],
                     metadata: List[Dict] = None) -> None:
        """Add documents incrementally. Buffers them; only rebuilds if threshold
        is crossed, otherwise defers to next retrieve() or explicit flush."""
        if not documents:
            return

        existing_ids = set(self.doc_ids) | set(self._pending_ids)
        new_docs, new_ids, new_meta = [], [], []
        for i, doc_id in enumerate(doc_ids):
            if doc_id not in existing_ids:
                new_docs.append(documents[i])
                new_ids.append(doc_id)
                new_meta.append((metadata[i] if metadata else {}))

        if not new_docs:
            logger.debug("All documents already in BM25 index, skipping")
            return

        self._pending_docs.extend(new_docs)
        self._pending_ids.extend(new_ids)
        self._pending_meta.extend(new_meta)
        self._dirty = True

        pending_total = len(self._pending_ids)
        if pending_total >= self.batch_threshold:
            self._ensure_fresh()
            logger.info(f"BM25 index: flushed {pending_total} pending docs "
                        f"(total: {len(self.documents)})")
        else:
            logger.debug(f"BM25 index: {pending_total} pending docs buffered "
                         f"(threshold={self.batch_threshold})")

    def flush(self) -> None:
        """Force rebuild with all pending documents."""
        self._ensure_fresh()

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Retrieve top-k documents using BM25 scoring. Triggers flush if dirty."""
        self._ensure_fresh()

        if self.bm25 is None:
            logger.warning("BM25 index not built yet")
            return [], []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        top_doc_ids = [self.doc_ids[i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]

        return top_doc_ids, top_scores

    def retrieve_with_metadata(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k documents with all metadata."""
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
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [token for token in text.split() if len(token) > 1]

    def update_index(self, documents: List[str], doc_ids: List[str],
                    metadata: List[Dict] = None) -> None:
        """Update index with new documents. Uses incremental add."""
        self.add_documents(documents, doc_ids, metadata)

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get a document by its ID (searches both committed and pending)."""
        if doc_id in self.doc_ids:
            idx = self.doc_ids.index(doc_id)
            return {'doc_id': doc_id, 'text': self.documents[idx], 'metadata': self.metadata[idx]}

        if doc_id in self._pending_ids:
            idx = self._pending_ids.index(doc_id)
            return {'doc_id': doc_id, 'text': self._pending_docs[idx],
                    'metadata': self._pending_meta[idx]}

        return None

    def save_to_disk(self, path: str = None) -> bool:
        """Persist the BM25 index to disk (flushes pending docs first)."""
        self._ensure_fresh()

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
        """Load the BM25 index from disk."""
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

            tokenized = [self._tokenize(doc) for doc in documents]
            self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
            self.documents = documents
            self.doc_ids = doc_ids
            self.metadata = metadata
            self._pending_docs.clear()
            self._pending_ids.clear()
            self._pending_meta.clear()
            self._dirty = False

            logger.info(f"BM25 index loaded from {load_path} ({len(documents)} documents)")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    @property
    def index_size(self) -> int:
        """Number of committed + pending documents."""
        return len(self.documents) + len(self._pending_ids)

    @property
    def is_ready(self) -> bool:
        """Whether the index has documents available for queries."""
        return self.bm25 is not None and (len(self.documents) > 0 or len(self._pending_ids) > 0)

    @property
    def pending_count(self) -> int:
        """Number of documents in the pending buffer."""
        return len(self._pending_ids)
