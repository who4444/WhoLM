
import logging
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import hashlib
import json

from database.qdrant import QdrantDB
from config.config import Config

logger = logging.getLogger(__name__)


class VectorIngester:
    """
    Handles ingestion of different types of embeddings into Qdrant collections.
    Supports separate collections for different modalities.
    """

    def __init__(self, qdrant_url: str = None):
        """
        Initialize vector ingester with Qdrant connection.

        Args:
            qdrant_url: Qdrant server URL (uses Config default if None)
        """
        self.qdrant_url = qdrant_url or Config.QDRANT_URL

        # Initialize single QdrantDB instance (handles all collections)
        self.db = QdrantDB(url=self.qdrant_url)

        logger.info("VectorIngester initialized with Qdrant collections: text_documents, frame_embeddings, conversations, conversation_contexts")

    def push_text_embeddings(self, texts: List[str], embeddings: np.ndarray,
                           batch_size: int = 100) -> Dict:
        """
        Push text embeddings to Qdrant.

        Args:
            texts: List of original text strings
            embeddings: Text embeddings array (n_texts, 1024)
            batch_size: Batch size for upsert operations

        Returns:
            Dict with operation results
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have same length")

        logger.info(f"Pushing {len(texts)} text embeddings to Qdrant")

        # Prepare payloads
        payloads = []
        for i, text in enumerate(texts):
            payload = {
                "modality": "text",
                "type": "documents",
                "text": text,
                "doc_id": self._generate_doc_id(text, f"text_{i}"),
                "chunk_index": i
            }

            payloads.append(payload)

        # Generate unique IDs
        ids = [self._generate_point_id(payload["doc_id"]) for payload in payloads]

        # Push in batches
        return self._batch_upsert(self.db, ids, embeddings, payloads, batch_size, collection_type="document")

    def push_frame_embeddings(self, frame_paths: List[str], embeddings: np.ndarray,
                           batch_size: int = 100) -> Dict:
        """
        Push frame embeddings to Qdrant.

        Args:
            frame_paths: List of paths to frame images
            embeddings: Frame embeddings array (n_frames, 512)
            batch_size: Batch size for upsert operations

        Returns:
            Dict with operation results
        """
        if len(frame_paths) != len(embeddings):
            raise ValueError("frame_paths and embeddings must have same length")


        logger.info(f"Pushing {len(frame_paths)} frame embeddings to Qdrant")

        # Prepare payloads
        payloads = []
        for i, frame_path in enumerate(frame_paths):
            frame_path = Path(frame_path)

            # Extract frame metadata from path/filename
            frame_info = self._parse_frame_path(frame_path)

            payload = {
                "modality": "visual",
                "frame_path": str(frame_path),
                "frame_filename": frame_path.name,
                "video_id": frame_info["video_id"],
                "frame_num": frame_info["frame_num"],
                "timestamp": frame_info["timestamp"],
                "doc_id": self._generate_doc_id(str(frame_path), f"frame_{i}")
            }

            payloads.append(payload)

        # Generate unique IDs
        ids = [self._generate_point_id(payload["doc_id"]) for payload in payloads]

        # Push in batches
        return self._batch_upsert(self.db, ids, embeddings, payloads, batch_size, collection_type="frame")

    def push_video_transcript(self, video_id: str, transcript_chunks: List[Dict],
                            embeddings: np.ndarray, batch_size: int = 100) -> Dict:
        """
        Push video transcript embeddings with rich metadata.

        Args:
            video_id: ID of the source video
            transcript_chunks: List of dicts with 'text', 'start', 'end' keys
            embeddings: Text embeddings for transcript chunks
            batch_size: Batch size for upsert operations

        Returns:
            Dict with operation results
        """
        if len(transcript_chunks) != len(embeddings):
            raise ValueError("transcript_chunks and embeddings must have same length")

        logger.info(f"Pushing {len(transcript_chunks)} transcript chunks for video: {video_id}")

        # Prepare payloads
        payloads = []
        for i, chunk in enumerate(transcript_chunks):
            payload = {
                "modality": "text",
                "type": "transcript",
                "video_id": video_id,
                "text": chunk["text"],
                "start_time": chunk.get("start", 0),
                "end_time": chunk.get("end", 0),
                "duration": chunk.get("end", 0) - chunk.get("start", 0),
                "text_length": len(chunk["text"]),
                "chunk_index": i,
                "doc_id": self._generate_doc_id(f"{video_id}_{i}", chunk["text"])
            }
            payloads.append(payload)

        # Generate unique IDs
        ids = [self._generate_point_id(payload["doc_id"]) for payload in payloads]

        # Push in batches
        return self._batch_upsert(self.db, ids, embeddings, payloads, batch_size, collection_type="document")

    def push_document_embeddings(self, doc_id: str, text_chunks: List[str],
                               embeddings: np.ndarray, 
                               batch_size: int = 100) -> Dict:
        """
        Push document text embeddings with document-level metadata.

        Args:
            doc_id: ID of the source document
            text_chunks: List of text chunks from the document
            embeddings: Text embeddings for chunks
            batch_size: Batch size for upsert operations

        Returns:
            Dict with operation results
        """
        if len(text_chunks) != len(embeddings):
            raise ValueError("text_chunks and embeddings must have same length")

        logger.info(f"Pushing {len(text_chunks)} document chunks for: {doc_id}")

        # Prepare payloads
        payloads = []
        for i, chunk in enumerate(text_chunks):
            payload = {
                "modality": "text",
                "type": "document",
                "text": chunk,
                "text_length": len(chunk),
                "chunk_index": i,
                "doc_id": self._generate_doc_id(f"{doc_id}_{i}", chunk)
            }

            payloads.append(payload)

        # Generate unique IDs
        ids = [self._generate_point_id(payload["doc_id"]) for payload in payloads]

        # Push in batches
        return self._batch_upsert(self.db, ids, embeddings, payloads, batch_size, collection_type="document")

    def search_text(self, query_embedding: np.ndarray, limit: int = 10,
                   filters: Dict = None) -> List[Dict]:
        """
        Search text collection.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum results
            filters: Optional metadata filters

        Returns:
            Search results
        """
        if filters:
            return self.db.search_with_filter(query_embedding, filters, limit, collection_type="document")
        else:
            return self.db.text_search(query_embedding, limit, collection_type="document")

    def search_frames(self, query_embedding: np.ndarray, limit: int = 10,
                     filters: Dict = None) -> List[Dict]:
        """
        Search frame collection.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum results
            filters: Optional metadata filters

        Returns:
            Search results
        """
        if filters:
            return self.db.search_with_filter(query_embedding, filters, limit, collection_type="frame")
        else:
            return self.db.frame_search(query_embedding, limit, collection_type="frame")

    def get_collection_stats(self) -> Dict:
        """
        Get statistics for all collections.

        Returns:
            Dict with collection statistics
        """
        text_stats = self.db.get_collection_info(collection_type="document")
        frame_stats = self.db.get_collection_info(collection_type="frame")

        return {
            "text_collection": text_stats,
            "frame_collection": frame_stats
        }

    def clear_all_collections(self) -> Dict:
        """
        Clear all collections (use with caution).

        Returns:
            Dict with operation results
        """
        text_result = self.db.clear_collection(collection_type="document")
        frame_result = self.db.clear_collection(collection_type="frame")

        return {
            "text_collection_cleared": text_result,
            "frame_collection_cleared": frame_result
        }

    def _batch_upsert(self, collection: QdrantDB, ids: List[int],
                     embeddings: np.ndarray, payloads: List[Dict],
                     batch_size: int, collection_type: str = "document") -> Dict:
        """
        Upsert embeddings in batches.

        Args:
            collection: QdrantDB collection instance
            ids: Point IDs
            embeddings: Embeddings array
            payloads: Payload dicts
            batch_size: Batch size

        Returns:
            Dict with operation results
        """
        total_points = len(ids)
        successful_batches = 0
        failed_batches = 0
        errors = []

        for i in range(0, total_points, batch_size):
            end_idx = min(i + batch_size, total_points)

            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_payloads = payloads[i:end_idx]

            try:
                success = collection.upsert(batch_ids, batch_embeddings, batch_payloads, collection_type=collection_type)
                if success:
                    successful_batches += 1
                else:
                    failed_batches += 1
                    errors.append(f"Batch {i//batch_size + 1} failed")
            except Exception as e:
                failed_batches += 1
                errors.append(f"Batch {i//batch_size + 1}: {str(e)}")

        return {
            "total_points": total_points,
            "successful_batches": successful_batches,
            "failed_batches": failed_batches,
            "batch_size": batch_size,
            "errors": errors[:10]  # Limit error messages
        }

    @staticmethod
    def _generate_doc_id(content: str, fallback: str) -> str:
        """
        Generate a unique document ID from content.

        Args:
            content: Content to hash
            fallback: Fallback ID if hashing fails

        Returns:
            Unique document ID
        """
        try:
            # Create hash from content
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
            return f"doc_{content_hash}"
        except:
            return fallback

    @staticmethod
    def _generate_point_id(doc_id: str) -> int:
        """
        Generate a numeric point ID from document ID.

        Args:
            doc_id: Document identifier

        Returns:
            Numeric point ID
        """
        try:
            # Use hash of doc_id to generate consistent numeric ID
            return int(hashlib.md5(doc_id.encode('utf-8')).hexdigest()[:16], 16)
        except:
            # Fallback to random ID
            return hash(doc_id) % 2**31

    @staticmethod
    def _parse_frame_path(frame_path: Path) -> Dict:
        """
        Parse frame path to extract metadata.

        Args:
            frame_path: Path to frame file

        Returns:
            Dict with parsed metadata
        """
        # Default values
        info = {
            "video_name": frame_path.parent.name,
            "frame_num": 0,
            "timestamp": 0.0
        }

        # Try to parse frame number from filename
        import re
        match = re.search(r"frame_(\d+)", frame_path.name)
        if match:
            frame_num = int(match.group(1))
            info["frame_num"] = frame_num
            # Assume 1 fps for timestamp calculation
            info["timestamp"] = frame_num / 1.0

        return info


# Convenience functions for easy usage
def push_text_to_qdrant(texts: List[str], embeddings: np.ndarray,
                       metadata: List[Dict] = None) -> Dict:
    """
    Convenience function to push text embeddings.

    Args:
        texts: List of text strings
        embeddings: Text embeddings array
        metadata: Optional metadata

    Returns:
        Operation results
    """
    ingester = VectorIngester()
    return ingester.push_text_embeddings(texts, embeddings, metadata)


def push_frames_to_qdrant(frame_paths: List[str], embeddings: np.ndarray,
                         metadata: List[Dict] = None) -> Dict:
    """
    Convenience function to push frame embeddings.

    Args:
        frame_paths: List of frame file paths
        embeddings: Frame embeddings array
        metadata: Optional metadata

    Returns:
        Operation results
    """
    ingester = VectorIngester()
    return ingester.push_frame_embeddings(frame_paths, embeddings, metadata)


def push_transcript_to_qdrant(video_name: str, transcript_chunks: List[Dict],
                            embeddings: np.ndarray) -> Dict:
    """
    Convenience function to push video transcript embeddings.

    Args:
        video_name: Source video name
        transcript_chunks: Transcript chunks with timing
        embeddings: Text embeddings

    Returns:
        Operation results
    """
    ingester = VectorIngester()
    return ingester.push_video_transcript(video_name, transcript_chunks, embeddings)