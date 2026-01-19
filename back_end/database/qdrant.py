import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, SparseVectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client import models

logger = logging.getLogger(__name__)


class QdrantDB:
    """
    Qdrant vector database integration for storing and retrieving embeddings.
    """
    
    def __init__(self, url: str = "http://localhost:6333", 
                 vector_collection: str = "documents",
                 vector_dim: int = 1024,
                 timeout: int = 30):
        """
        Initialize Qdrant client and create collections if needed.
        
        Args:
            url: Qdrant server URL
            vector_collection: Name of vector collection
            vector_dim: Dimension of vectors
            timeout: Request timeout in seconds
        """
        self.client = QdrantClient(url=url, timeout=timeout)
        self.vector_collection = vector_collection
        self.vector_dim = vector_dim
        
        # Create vector collection if it doesn't exist
        if not self.client.collection_exists(self.vector_collection):
            self.client.create_collection(
                collection_name=self.vector_collection,            
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
                sparse_vectors_config = {"sparse_vector": models.SparseVectorParams()}
            )
            logger.info(f"Created collection '{self.vector_collection}'")
    
    def upsert(self, ids: List[int], vectors: np.ndarray, payloads: List[Dict]) -> bool:
        """
        Insert or update vectors with their payloads.
        
        Args:
            ids: List of point IDs
            vectors: Embeddings array (n, vector_dim)
            payloads: List of payload dicts with document metadata
            
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
            self.client.upsert(
                collection_name=self.vector_collection,
                points=points
            )
            logger.info(f"Upserted {len(points)} points to '{self.vector_collection}'")
            return True
        except Exception as e:
            logger.error(f"Error upserting points: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, limit: int = 10) -> List[Dict]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: Query embedding
            limit: Maximum number of results
            
        Returns:
            List of search results with score and payload
        """
        # Normalize query vector
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        try:
            results = self.client.search(
                collection_name=self.vector_collection,
                query_vector=query_vector.tolist(),
                limit=limit
            )
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def search_with_filter(self, query_vector: np.ndarray, 
                          filter_dict: Dict, limit: int = 10) -> List[Dict]:
        """
        Search with optional metadata filtering.
        
        Args:
            query_vector: Query embedding
            filter_dict: Filter conditions (e.g., {"video_name": "video_1"})
            limit: Maximum number of results
            
        Returns:
            Filtered search results
        """
        # Normalize query vector
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        try:
            results = self.client.search(
                collection_name=self.vector_collection,
                query_vector=query_vector.tolist(),
                limit=limit
            )
            
            # Client-side filtering
            filtered = []
            for result in results:
                payload = result.payload
                if all(payload.get(k) == v for k, v in filter_dict.items()):
                    filtered.append({
                        "id": result.id,
                        "score": result.score,
                        "payload": result.payload
                    })
                if len(filtered) >= limit:
                    break
            
            return filtered
        except Exception as e:
            logger.error(f"Error searching with filter: {e}")
            return []
    
    def get_point(self, point_id: int) -> Optional[Dict]:
        """
        Retrieve a single point by ID.
        
        Args:
            point_id: Point ID
            
        Returns:
            Point data with vector and payload
        """
        try:
            points = self.client.retrieve(
                collection_name=self.vector_collection,
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
    
    def delete_points(self, ids: List[int]) -> bool:
        """
        Delete points by IDs.
        
        Args:
            ids: List of point IDs to delete
            
        Returns:
            Success status
        """
        try:
            self.client.delete(
                collection_name=self.vector_collection,
                points_selector=[int(id) for id in ids]
            )
            logger.info(f"Deleted {len(ids)} points from '{self.vector_collection}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting points: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Delete all points in the collection.
        
        Returns:
            Success status
        """
        try:
            self.client.delete_collection(collection_name=self.vector_collection)
            self.client.create_collection(
                collection_name=self.vector_collection,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
            )
            logger.info(f"Cleared collection '{self.vector_collection}'")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def get_collection_info(self) -> Optional[Dict]:
        """
        Get collection statistics.
        
        Returns:
            Collection info with point count and vector config
        """
        try:
            info = self.client.get_collection(self.vector_collection)
            return {
                "name": info.name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "vector_size": self.vector_dim,
                "status": str(info.status)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
    
    def batch_search(self, query_vectors: np.ndarray, limit: int = 10) -> List[List[Dict]]:
        """
        Search with multiple query vectors.
        
        Args:
            query_vectors: Array of query embeddings (n, vector_dim)
            limit: Maximum results per query
            
        Returns:
            List of result lists for each query
        """
        results = []
        for query_vector in query_vectors:
            results.append(self.search(query_vector, limit))
        return results