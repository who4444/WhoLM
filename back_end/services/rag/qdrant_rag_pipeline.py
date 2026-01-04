
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from .hybrid_retriever import HybridRetriever
from .reranker import reranker
from ingestion.embeddings.text_encoder import model as text_encoder_model
from database.qdrant import QdrantDB

logger = logging.getLogger(__name__)


class QdrantRAGPipeline:
    """
    RAG pipeline that uses Qdrant for persistent storage of embeddings
    and hybrid retrieval (BM25 + dense).
    """
    
    def __init__(self, qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "documents",
                 embedding_dim: int = 1024,
                 bm25_weight: float = 0.5,
                 dense_weight: float = 0.5,
                 reranker_top_k: int = 3):
        """
        Initialize Qdrant-integrated RAG pipeline.
        
        Args:
            qdrant_url: Qdrant server URL
            collection_name: Name of vector collection
            embedding_dim: Dimension of embeddings (1024 for BGE-M3)
            bm25_weight: Weight for BM25 in hybrid retriever
            dense_weight: Weight for dense embeddings
            reranker_top_k: Number of results after reranking
        """
        self.qdrant_db = QdrantDB(
            url=qdrant_url,
            vector_collection=collection_name,
            vector_dim=embedding_dim
        )
        self.hybrid_retriever = HybridRetriever(
            bm25_weight=bm25_weight,
            dense_weight=dense_weight
        )
        self.reranker_top_k = reranker_top_k
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.next_point_id = 0
    
    def add_transcripts(self, documents: List[str], doc_ids: List[str],
                     metadata: List[Dict] = None) -> Dict:
        """
        Add documents to RAG system with embeddings stored in Qdrant.
        
        Args:
            documents: List of document texts
            doc_ids: List of document identifiers
            metadata: Optional metadata for each document
            
        Returns:
            Dict with add operation results
        """
        if not documents:
            logger.warning("No documents provided")
            return {"success": False, "message": "No documents provided"}
        
        if len(documents) != len(doc_ids):
            raise ValueError("documents and doc_ids must have same length")
        
        logger.info(f"Adding {len(documents)} documents to RAG system")
        
        try:
            # Encode documents
            embeddings_output = text_encoder_model.encode(
                documents,
                batch_size=16,
                return_dense=True,
                return_sparse=False
            )
            
            # Extract dense embeddings
            if hasattr(embeddings_output, 'tolist'):
                embeddings = np.array(embeddings_output.tolist())
            else:
                embeddings = np.array(embeddings_output)
            
            # Prepare payloads for Qdrant
            point_ids = list(range(self.next_point_id, self.next_point_id + len(documents)))
            payloads = [
                {
                    "doc_id": doc_ids[i],
                    "text": documents[i],
                    **(metadata[i] if metadata else {})
                }
                for i in range(len(documents))
            ]
            
            # Store in Qdrant
            success = self.qdrant_db.upsert(point_ids, embeddings, payloads)
            
            if success:
                # Update hybrid retriever's BM25 index
                self.hybrid_retriever.bm25_index.update_index(
                    documents, doc_ids, metadata
                )
                
                # Store embeddings in memory for hybrid retrieval
                for doc_id, embedding in zip(doc_ids, embeddings):
                    self.hybrid_retriever.embeddings[doc_id] = embedding / np.linalg.norm(embedding)
                
                self.next_point_id += len(documents)
                self.hybrid_retriever.embedding_dim = self.embedding_dim
                
                logger.info(f"Successfully added {len(documents)} documents")
                return {
                    "success": True,
                    "documents_added": len(documents),
                    "total_documents": self.next_point_id
                }
            else:
                return {"success": False, "message": "Failed to store embeddings in Qdrant"}
                
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"success": False, "message": str(e)}
    
    def query(self, query_text: str, retriever_top_k: int = 10,
             video_filter: Optional[str] = None) -> List[Dict]:
        """
        Execute RAG query: retrieve from Qdrant and rerank.
        
        Args:
            query_text: User query
            retriever_top_k: Number of candidates to retrieve
            video_filter: Optional video name to filter results
            
        Returns:
            List of reranked results with scores
        """
        logger.info(f"Processing query: {query_text}")
        
        # Encode query
        query_embedding = text_encoder_model.encode(
            [query_text],
            batch_size=1,
            return_dense=True,
            return_sparse=False
        )[0]
        
        # Retrieve from Qdrant
        qdrant_results = self.qdrant_db.search(query_embedding, limit=retriever_top_k)
        
        if not qdrant_results:
            logger.warning("No results from Qdrant")
            return []
        
        # Convert Qdrant results to hybrid retriever format
        candidates = [
            {
                "doc_id": result["payload"]["doc_id"],
                "text": result["payload"]["text"],
                "metadata": {k: v for k, v in result["payload"].items() 
                           if k not in ["doc_id", "text"]},
                "hybrid_score": result["score"],
                "dense_score": result["score"]
            }
            for result in qdrant_results
        ]
        
        # Rerank results
        logger.debug(f"Reranking {len(candidates)} candidates")
        reranked = self._rerank_results(query_text, candidates)
        
        return reranked
    
    def hybrid_query(self, query_text: str, retriever_top_k: int = 10) -> List[Dict]:
        """
        Execute true hybrid query combining BM25 and Qdrant dense retrieval.
        Uses Qdrant for dense similarity and BM25 index for keyword matching.
        
        Args:
            query_text: User query
            retriever_top_k: Number of candidates before reranking
            
        Returns:
            List of hybrid-ranked results
        """
        logger.info(f"Processing hybrid query: {query_text}")
        
        # Encode query
        query_embedding = text_encoder_model.encode(
            [query_text],
            batch_size=1,
            return_dense=True,
            return_sparse=False
        )[0]
        
        # Dense retrieval from Qdrant
        qdrant_results = self.qdrant_db.search(query_embedding, limit=retriever_top_k * 2)
        dense_doc_ids = [r["payload"]["doc_id"] for r in qdrant_results]
        
        # BM25 retrieval from memory index
        bm25_doc_ids, bm25_scores = self.hybrid_retriever.bm25_index.retrieve(
            query_text, retriever_top_k * 2
        )
        
        # Combine results
        combined = {}
        
        # Normalize and add Qdrant scores
        if qdrant_results:
            scores = [r["score"] for r in qdrant_results]
            max_score = max(scores)
            min_score = min(scores)
            score_range = max_score - min_score if max_score != min_score else 1
            
            for result in qdrant_results:
                doc_id = result["payload"]["doc_id"]
                norm_score = (result["score"] - min_score) / score_range
                combined[doc_id] = {
                    "dense_score": norm_score,
                    "bm25_score": 0,
                    "data": result["payload"]
                }
        
        # Normalize and add BM25 scores
        if bm25_scores:
            max_bm25 = max(bm25_scores)
            min_bm25 = min(bm25_scores)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
            
            for doc_id, bm25_score in zip(bm25_doc_ids, bm25_scores):
                norm_bm25 = (bm25_score - min_bm25) / bm25_range
                if doc_id in combined:
                    combined[doc_id]["bm25_score"] = norm_bm25
                else:
                    doc = self.hybrid_retriever.bm25_index.get_document(doc_id)
                    combined[doc_id] = {
                        "dense_score": 0,
                        "bm25_score": norm_bm25,
                        "data": {
                            "doc_id": doc_id,
                            "text": doc["text"],
                            **doc["metadata"]
                        }
                    }
        
        # Calculate final hybrid scores
        w_dense = self.hybrid_retriever.dense_weight
        w_bm25 = self.hybrid_retriever.bm25_weight
        
        scored_results = []
        for doc_id, scores in combined.items():
            hybrid_score = (scores["dense_score"] * w_dense) + (scores["bm25_score"] * w_bm25)
            scored_results.append({
                "doc_id": doc_id,
                "text": scores["data"]["text"],
                "metadata": {k: v for k, v in scores["data"].items() 
                           if k not in ["doc_id", "text"]},
                "hybrid_score": hybrid_score,
                "dense_score": scores["dense_score"],
                "bm25_score": scores["bm25_score"]
            })
        
        # Sort by hybrid score and get top-k
        scored_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        candidates = scored_results[:retriever_top_k]
        
        # Rerank
        logger.debug(f"Reranking {len(candidates)} candidates")
        reranked = self._rerank_results(query_text, candidates)
        
        return reranked
    
    def _rerank_results(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Query text
            candidates: Candidate results
            
        Returns:
            Reranked results
        """
        if not candidates:
            return []
        
        candidate_texts = [c["text"] for c in candidates]
        reranked_texts = reranker(query, candidate_texts, self.reranker_top_k)
        
        reranked_results = []
        for reranked_text in reranked_texts:
            for candidate in candidates:
                if candidate["text"] == reranked_text:
                    reranked_results.append(candidate)
                    break
        
        return reranked_results[:self.reranker_top_k]
    
    def get_stats(self) -> Dict:
        """
        Get RAG system statistics.
        
        Returns:
            Dict with system statistics
        """
        collection_info = self.qdrant_db.get_collection_info()
        return {
            "collection": self.collection_name,
            "total_documents": collection_info["points_count"] if collection_info else 0,
            "embedding_dim": self.embedding_dim,
            "bm25_documents": len(self.hybrid_retriever.bm25_index.doc_ids),
            "reranker_top_k": self.reranker_top_k,
            "bm25_weight": self.hybrid_retriever.bm25_weight,
            "dense_weight": self.hybrid_retriever.dense_weight
        }
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents from RAG system.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Success status
        """
        logger.info(f"Deleting {len(doc_ids)} documents")
        
        try:
            # Find point IDs corresponding to doc_ids
            # This requires searching through payloads - simplified approach
            logger.warning("Document deletion requires rebuilding index")
            return False
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def clear_all(self) -> bool:
        """
        Clear all documents from RAG system.
        
        Returns:
            Success status
        """
        try:
            self.qdrant_db.clear_collection()
            self.hybrid_retriever.bm25_index.documents = []
            self.hybrid_retriever.bm25_index.doc_ids = []
            self.hybrid_retriever.bm25_index.metadata = []
            self.hybrid_retriever.embeddings = {}
            self.next_point_id = 0
            logger.info("Cleared all documents from RAG system")
            return True
        except Exception as e:
            logger.error(f"Error clearing system: {e}")
            return False
