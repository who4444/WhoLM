import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, SparseVectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client import models
from config.config import Config

logger = logging.getLogger(__name__)


class QdrantDB:
    """
    Qdrant vector database integration for storing and retrieving embeddings.
    """
    
    def __init__(self, url: str = "http://localhost:6333", 
                 document_collection: str = "text_documents",
                 frame_collection: str = "frame_embeddings",
                 doc_dim: int = 1024,
                 frame_dim: int = 512,
                 timeout: int = 30):
        """
        Initialize Qdrant client and create collections if needed.
        
        Args:
            url: Qdrant server URL
            document_collection: Name of text documents collection
            frame_collection: Name of frame embeddings collection
            doc_dim: Dimension of document embeddings
            frame_dim: Dimension of frame embeddings
            timeout: Request timeout in seconds
        """
        self.client = QdrantClient(url=url, timeout=timeout)
        self.document_collection = Config.QDRANT_DOC_COLLECTION
        self.vdocument_dim = Config.QDRANT_TEXT_EMBEDDING_DIM
        self.frame_collection = Config.QDRANT_FRAME_COLLECTION
        self.frame_dim = Config.QDRANT_FRAME_EMBEDDING_DIM

        # Create document collection if it doesn't exist
        if not self.client.collection_exists(self.document_collection):
            self.client.create_collection(
                collection_name=self.document_collection,            
                vectors_config=VectorParams(size=self.vdocument_dim, distance=Distance.COSINE),
                sparse_vectors_config = {"sparse_vector": models.SparseVectorParams()}
            )
            logger.info(f"Created collection '{self.document_collection}'")
        
        # Create frame collection if it doesn't exist
        if not self.client.collection_exists(self.frame_collection):
            self.client.create_collection(
                collection_name=self.frame_collection,            
                vectors_config=VectorParams(size=self.frame_dim, distance=Distance.COSINE),
                sparse_vectors_config = {"sparse_vector": models.SparseVectorParams()}
            )
            logger.info(f"Created collection '{self.frame_collection}'")
        
        # Create conversations collection if it doesn't exist
        if not self.client.collection_exists("conversations"):
            self.client.create_collection(
                collection_name="conversations",
                vectors_config=VectorParams(size=self.vdocument_dim, distance=Distance.COSINE)
            )
            logger.info("Created collection 'conversations'")
        
        # Create conversation_contexts collection if it doesn't exist
        if not self.client.collection_exists("conversation_contexts"):
            self.client.create_collection(
                collection_name="conversation_contexts",
                vectors_config=VectorParams(size=self.vdocument_dim, distance=Distance.COSINE)
            )
            logger.info("Created collection 'conversation_contexts'")
    
    def upsert(self, ids: List[int], vectors: np.ndarray, payloads: List[Dict], collection_type: str = "document") -> bool:
        """
        Insert or update vectors with their payloads.
        
        Args:
            ids: List of point IDs
            vectors: Embeddings array (n, vector_dim)
            payloads: List of payload dicts with document metadata
            collection_type: Type of collection - "document" or "frame"
            
        Returns:
            Success status
        """
        if len(ids) != len(vectors) or len(ids) != len(payloads):
            raise ValueError("ids, vectors, and payloads must have same length")
                
        # Create points
        points = [
            PointStruct(
                id=int(ids[i]),
                vector=vectors[i].tolist() if isinstance(vectors[i], np.ndarray) else vectors[i],
                payload=payloads[i]
            )
            for i in range(len(ids))
        ]
        
        try:
            collection_name = self.document_collection if collection_type == "document" else self.frame_collection
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} points to '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error upserting points: {e}")
            return False
    
    def text_search(self, query_vector: np.ndarray, limit: int = 10, collection_type: str = "document") -> List[Dict]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: Query embedding
            limit: Maximum number of results
            collection_type: Type of collection - "document" or "frame"
            
        Returns:
            List of search results with score and payload
        """
        # Normalize query vector
        query_vector = query_vector / np.linalg.norm(query_vector)
        collection_name = self._get_collection_name(collection_type)
        
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector.tolist(),
                limit=limit
            )
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results.points
            ]
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    

    def frame_search(self, query_vector: np.ndarray, limit: int = 10, collection_type: str = "frame") -> List[Dict]:
        """
        Search for similar frame vectors using cosine similarity.
        
        Args:
            query_vector: Query embedding
            limit: Maximum number of results
            collection_type: Type of collection - "document" or "frame"
            
        Returns:
            List of search results with score and payload
        """
        # Normalize query vector
        query_vector = query_vector / np.linalg.norm(query_vector)
        collection_name = self._get_collection_name(collection_type)
        
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector.tolist(),
                limit=limit
            )
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results.points
            ]
        except Exception as e:
            logger.error(f"Error searching frame vectors: {e}")
            return []
    
    def _get_collection_name(self, collection_type: str) -> str:
        """
        Get collection name based on type.
        
        Args:
            collection_type: Type of collection - "document" or "frame"
            
        Returns:
            Collection name
        """
        return self.document_collection if collection_type == "document" else self.frame_collection
    
    def search_both_collections(self, text_query_vector: np.ndarray, frame_query_vector: np.ndarray, 
                               limit: int = 10) -> List[Dict]:
        """
        Search both text and frame collections simultaneously.
        Returns combined results sorted by score.
        
        Args:
            text_query_vector: Query embedding for text (BGE-M3, 1024-dim)
            frame_query_vector: Query embedding for frames (CLIP, 512-dim)
            limit: Maximum number of results per collection
            
        Returns:
            List of search results from both collections with collection_type metadata
        """
        combined_results = []
        
        # Normalize text query vector
        text_query_vector = text_query_vector / np.linalg.norm(text_query_vector)
        
        # Search document collection
        try:
            doc_results = self.client.query_points(
                collection_name=self.document_collection,
                query=text_query_vector.tolist(),
                limit=limit
            )
            for result in doc_results.points:
                combined_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                    "collection_type": "document"
                })
        except Exception as e:
            logger.error(f"Error searching document collection: {e}")
        
        # Normalize frame query vector
        frame_query_vector = frame_query_vector / np.linalg.norm(frame_query_vector)
        
        # Search frame collection
        try:
            frame_results = self.client.query_points(
                collection_name=self.frame_collection,
                query=frame_query_vector.tolist(),
                limit=limit
            )
            for result in frame_results.points:
                combined_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                    "collection_type": "frame"
                })
        except Exception as e:
            logger.error(f"Error searching frame collection: {e}")
        
        # Sort by score descending and limit total results
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:limit * 2]

    def search_with_filter(self, query_vector: np.ndarray, 
                          filter_dict: Dict, limit: int = 10, collection_type: str = "document") -> List[Dict]:
        """
        Search with optional metadata filtering.
        
        Args:
            query_vector: Query embedding
            filter_dict: Filter conditions (e.g., {"video_name": "video_1"})
            limit: Maximum number of results
            collection_type: Type of collection - "document" or "frame"
            
        Returns:
            Filtered search results with collection_type
        """
        # Normalize query vector
        query_vector = query_vector / np.linalg.norm(query_vector)
        collection_name = self._get_collection_name(collection_type)
        
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector.tolist(),
                limit=limit
            )
            
            # Client-side filtering
            filtered = []
            for result in results.points:
                payload = result.payload
                if all(payload.get(k) == v for k, v in filter_dict.items()):
                    filtered.append({
                        "id": result.id,
                        "score": result.score,
                        "payload": result.payload,
                        "collection_type": collection_type
                    })
                if len(filtered) >= limit:
                    break
            
            return filtered
        except Exception as e:
            logger.error(f"Error searching with filter: {e}")
            return []
    
    def get_point(self, point_id: int, collection_type: str = "document") -> Optional[Dict]:
        """
        Retrieve a single point by ID.
        
        Args:
            point_id: Point ID
            collection_type: Type of collection - "document" or "frame"
            
        Returns:
            Point data with vector and payload
        """
        collection_name = self._get_collection_name(collection_type)
        
        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id]
            )
            
            if points:
                point = points[0]
                return {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving point: {e}")
            return None
    
    def delete_points(self, ids: List[int], collection_type: str = "document") -> bool:
        """
        Delete points by IDs.
        
        Args:
            ids: List of point IDs to delete
            collection_type: Type of collection - "document" or "frame"
            
        Returns:
            Success status
        """
        collection_name = self._get_collection_name(collection_type)
        
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=[int(id) for id in ids]
            )
            logger.info(f"Deleted {len(ids)} points from '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting points: {e}")
            return False
    
    def clear_collection(self, collection_type: str = "document") -> bool:
        """
        Delete all points in the collection.
        
        Args:
            collection_type: Type of collection - "document" or "frame"
            
        Returns:
            Success status
        """
        collection_name = self._get_collection_name(collection_type)
        vector_dim = self.vdocument_dim if collection_type == "document" else self.frame_dim
        
        try:
            self.client.delete_collection(collection_name=collection_name)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
            )
            logger.info(f"Cleared collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def get_collection_info(self, collection_type: str = "document") -> Optional[Dict]:
        """
        Get collection statistics.
        
        Args:
            collection_type: Type of collection - "document" or "frame"
            
        Returns:
            Collection info with point count and vector config
        """
        collection_name = self._get_collection_name(collection_type)
        vector_dim = self.vdocument_dim if collection_type == "document" else self.frame_dim
        
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": info.name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "vector_size": vector_dim,
                "status": str(info.status)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
    
    def batch_search(self, query_vectors: np.ndarray, limit: int = 10, collection_type: str = "document") -> List[List[Dict]]:
        """
        Search with multiple query vectors.
        
        Args:
            query_vectors: Array of query embeddings (n, vector_dim)
            limit: Maximum results per query
            collection_type: Type of collection - "document" or "frame"
            
        Returns:
            List of result lists for each query
        """
        results = []
        for query_vector in query_vectors:
            results.append(self.text_search(query_vector, limit, collection_type))
        return results
    