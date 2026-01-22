
import logging
from typing import List, Dict, Optional
import numpy as np
import torch
from .hybrid_retriever import HybridRetriever
from .reranker import reranker
from ingestion.embeddings.text_encoder import encode_texts
import open_clip
from database.qdrant import QdrantDB
from config.config import Config

logger = logging.getLogger(__name__)


class QdrantRAGPipeline:
    """
    Simplified RAG pipeline for querying and retrieving from Qdrant collections.
    Handles both dense retrieval and hybrid retrieval (BM25 + dense).
    Note: Document ingestion is handled by VectorIngester in processing_functions.py
    """
    
    def __init__(self, qdrant_url: str = "http://localhost:6333",
                 text_collection: str = "text_documents",
                 frame_collection: str = "frame_embeddings",
                 embedding_dim: int = 1024,
                 frame_embedding_dim: int = 512,
                 bm25_weight: float = 0.5,
                 dense_weight: float = 0.5,
                 reranker_top_k: int = 3):
        """
        Initialize RAG pipeline for querying.
        
        Args:
            qdrant_url: Qdrant server URL
            text_collection: Name of text vector collection
            frame_collection: Name of frame vector collection
            embedding_dim: Dimension of text embeddings (1024 for BGE-M3)
            frame_embedding_dim: Dimension of frame embeddings (512 for CLIP)
            bm25_weight: Weight for BM25 in hybrid retriever
            dense_weight: Weight for dense embeddings
            reranker_top_k: Number of results after reranking
        """
        self.text_db = QdrantDB(
            url=qdrant_url,
            document_collection=text_collection,
            frame_collection=frame_collection,
            doc_dim=embedding_dim,
            frame_dim=frame_embedding_dim
        )
        # Use same instance for both since QdrantDB now handles both collections
        self.frame_db = self.text_db
        
        # Initialize CLIP model for frame encoding
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            Config.CLIP_MODEL_NAME, 
            pretrained='openai'
        )
        self.clip_tokenizer = open_clip.get_tokenizer(Config.CLIP_MODEL_NAME)
        self.clip_model.eval()
        
        self.hybrid_retriever = HybridRetriever(
            bm25_weight=bm25_weight,
            dense_weight=dense_weight
        )
        self.reranker_top_k = reranker_top_k
        self.text_collection = text_collection
        self.frame_collection = frame_collection
        self.embedding_dim = embedding_dim
        self.frame_embedding_dim = frame_embedding_dim
    
    def query(self, query_text: str, retriever_top_k: int = 10, search_frames: bool = False) -> List[Dict]:
        """
        Execute RAG query: retrieve from Qdrant and rerank.
        Searches text collection (BGE-M3) or frame collection (CLIP) separately.
        
        Args:
            query_text: User query
            retriever_top_k: Number of candidates to retrieve before reranking
            search_frames: If True, search frame collection with CLIP encoder. If False, search text with BGE-M3
            
        Returns:
            List of reranked results with scores
        """
        logger.info(f"Processing query: {query_text} (search_frames={search_frames})")
        
        if search_frames:
            # Use CLIP's text encoder for frame embeddings
            text_tokens = self.clip_tokenizer([query_text])
            with torch.no_grad():
                query_embedding = self.clip_model.encode_text(text_tokens)
            query_embedding = query_embedding.cpu().numpy()[0]
            results = self.text_db.frame_search(query_embedding, limit=retriever_top_k)
        else:
            # Use BGE-M3 encoder for text documents
            query_embedding = encode_texts([query_text])[0]
            query_embedding = np.array(query_embedding)
            results = self.text_db.text_search(query_embedding, limit=retriever_top_k, collection_type="document")
        
        if not results:
            logger.warning("No results found for query")
            return []
        
        # Convert to candidate format
        candidates = [
            {
                "doc_id": result["payload"].get("doc_id", ""),
                "text": result["payload"].get("text", ""),
                "metadata": {k: v for k, v in result["payload"].items() 
                           if k not in ["doc_id", "text"]},
                "score": result["score"]
            }
            for result in results
        ]
        
        # Rerank results
        logger.debug(f"Reranking {len(candidates)} candidates")
        reranked = self._rerank_results(query_text, candidates)
        
        return reranked
    
    def hybrid_query(self, query_text: str, retriever_top_k: int = 10) -> List[Dict]:
        """
        Execute hybrid query combining BM25 and Qdrant dense retrieval.
        Uses Qdrant for dense similarity and BM25 index for keyword matching.
        
        Args:
            query_text: User query
            retriever_top_k: Number of candidates before reranking
            
        Returns:
            List of hybrid-ranked results
        """
        logger.info(f"Processing hybrid query: {query_text}")
        
        # Encode query
        query_embedding = encode_texts([query_text])[0]
        query_embedding = np.array(query_embedding)
        
        # Dense retrieval from Qdrant (text collection only for hybrid)
        qdrant_results = self.text_db.text_search(query_embedding, collection_type="document", limit=retriever_top_k * 2)
        
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
                doc_id = result["payload"].get("doc_id", "")
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
                    if doc:
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
            candidates: Candidate results with 'text' field
            
        Returns:
            Reranked results (top-k)
        """
        if not candidates:
            return []
        
        candidate_texts = [c["text"] for c in candidates]
        reranked_texts = reranker(query, candidate_texts, self.reranker_top_k)
        
        # Match reranked texts back to candidates
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
        text_info = self.text_db.get_collection_info()
        frame_info = self.frame_db.get_collection_info()
        
        return {
            "text_collection": self.text_collection,
            "frame_collection": self.frame_collection,
            "total_text_documents": text_info.get("points_count", 0) if text_info else 0,
            "total_frame_documents": frame_info.get("points_count", 0) if frame_info else 0,
            "text_embedding_dim": self.embedding_dim,
            "frame_embedding_dim": self.frame_embedding_dim,
            "reranker_top_k": self.reranker_top_k
        }
