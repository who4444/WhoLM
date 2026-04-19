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
    RAG pipeline for querying and retrieving from Qdrant collections.
    Supports dense-only and hybrid (BM25 + dense) retrieval modes.
    Note: Document ingestion is handled by VectorIngester in processing_functions.py
    """
    
    def __init__(self, qdrant_url: str = "http://localhost:6333",
                 text_collection: str = "text_documents",
                 frame_collection: str = "frame_embeddings",
                 embedding_dim: int = 1024,
                 frame_embedding_dim: int = 768,
                 bm25_weight: float = 0.5,
                 dense_weight: float = 0.5,
                 reranker_top_k: int = 5,
                 bm25_persist_path: str = None):
        """
        Initialize RAG pipeline for querying.
        
        Args:
            qdrant_url: Qdrant server URL
            text_collection: Name of text vector collection
            frame_collection: Name of frame vector collection
            embedding_dim: Dimension of text embeddings (1024 for BGE-M3)
            frame_embedding_dim: Dimension of frame embeddings (768 for CLIP)
            bm25_weight: Weight for BM25 in hybrid retriever
            dense_weight: Weight for dense embeddings
            reranker_top_k: Number of results after reranking
            bm25_persist_path: Path to persist BM25 index
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
            pretrained= Config.CLIP_PRETRAINED
        )
        self.clip_tokenizer = open_clip.get_tokenizer(Config.CLIP_MODEL_NAME)
        self.clip_model.eval()
        
        self.hybrid_retriever = HybridRetriever(
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            persist_path=bm25_persist_path
        )
        self.reranker_top_k = reranker_top_k
        self.text_collection = text_collection
        self.frame_collection = frame_collection
        self.embedding_dim = embedding_dim
        self.frame_embedding_dim = frame_embedding_dim

        logger.info(f"QdrantRAGPipeline initialized (hybrid BM25 docs: "
                    f"{self.hybrid_retriever.bm25_doc_count})")
    
    def query(self, query_text: str, retriever_top_k: int = 10,
              use_hybrid: bool = True) -> List[Dict]:
        """
        Execute RAG query: retrieve from text + frame collections, optionally
        combine with BM25 keyword scores, then rerank.
        
        Args:
            query_text: User query
            retriever_top_k: Number of candidates to retrieve before reranking
            use_hybrid: If True, combine BM25 + dense retrieval (default: True)
            
        Returns:
            List of reranked results with scores from both collections
        """
        logger.info(f"Processing query (hybrid={use_hybrid}): {query_text}")
        
        # Encode query with BGE-M3 for text documents
        text_embedding = encode_texts([query_text])[0]
        text_embedding = np.array(text_embedding)
        
        # Encode query with CLIP's text encoder for frame embeddings
        text_tokens = self.clip_tokenizer([query_text])
        with torch.no_grad():
            frame_embedding = self.clip_model.encode_text(text_tokens)
        frame_embedding = frame_embedding.cpu().numpy()[0]
        
        # --- Dense retrieval from Qdrant (both collections) ---
        all_results = self.text_db.search_both_collections(
            text_query_vector=text_embedding,
            frame_query_vector=frame_embedding,
            limit=retriever_top_k
        )
        
        # --- BM25 retrieval (text collection only) ---
        bm25_candidates = {}
        if use_hybrid and self.hybrid_retriever.bm25_ready:
            bm25_doc_ids, bm25_scores = self.hybrid_retriever.bm25_only_retrieve(
                query_text, top_k=retriever_top_k * 2
            )
            
            # Normalize BM25 scores to [0, 1]
            if bm25_scores:
                max_bm25 = max(bm25_scores)
                min_bm25 = min(bm25_scores)
                bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
                for doc_id, score in zip(bm25_doc_ids, bm25_scores):
                    bm25_candidates[doc_id] = (score - min_bm25) / bm25_range

            logger.debug(f"BM25 returned {len(bm25_candidates)} candidates")
        elif use_hybrid:
            logger.debug("Hybrid requested but BM25 index is empty, falling back to dense-only")

        if not all_results and not bm25_candidates:
            logger.warning("No results found for query")
            return []
        
        # --- Merge dense + BM25 scores ---
        candidate_map = {}  # doc_id -> candidate dict

        # Process Qdrant dense results
        # Normalize dense scores per collection for fair comparison
        doc_results = [r for r in all_results if r.get("collection_type") == "document"]
        frame_results = [r for r in all_results if r.get("collection_type") == "frame"]

        for result_set in [doc_results, frame_results]:
            if not result_set:
                continue
            scores = [r["score"] for r in result_set]
            max_s = max(scores)
            min_s = min(scores)
            range_s = max_s - min_s if max_s != min_s else 1

            for result in result_set:
                doc_id = result["payload"].get("doc_id", "")
                norm_dense_score = (result["score"] - min_s) / range_s

                candidate_map[doc_id] = {
                    "doc_id": doc_id,
                    "text": result["payload"].get("text", ""),
                    "metadata": {k: v for k, v in result["payload"].items()
                                if k not in ["doc_id", "text"]},
                    "dense_score": norm_dense_score,
                    "bm25_score": 0.0,
                    "collection_type": result["collection_type"]
                }

        # Merge BM25 scores into candidates
        bm25_w = self.hybrid_retriever.bm25_weight
        dense_w = self.hybrid_retriever.dense_weight

        for doc_id, bm25_norm_score in bm25_candidates.items():
            if doc_id in candidate_map:
                candidate_map[doc_id]["bm25_score"] = bm25_norm_score
            else:
                # BM25-only result: get text from BM25 index
                bm25_doc = self.hybrid_retriever.bm25_index.get_document(doc_id)
                if bm25_doc:
                    candidate_map[doc_id] = {
                        "doc_id": doc_id,
                        "text": bm25_doc["text"],
                        "metadata": bm25_doc.get("metadata", {}),
                        "dense_score": 0.0,
                        "bm25_score": bm25_norm_score,
                        "collection_type": "document"
                    }

        # Compute final hybrid score
        for candidate in candidate_map.values():
            if use_hybrid and self.hybrid_retriever.bm25_ready:
                candidate["score"] = (
                    candidate["dense_score"] * dense_w +
                    candidate["bm25_score"] * bm25_w
                )
            else:
                candidate["score"] = candidate["dense_score"]

        # Sort by score and take top candidates
        candidates = sorted(candidate_map.values(), 
                           key=lambda x: x["score"], reverse=True)
        candidates = candidates[:retriever_top_k]
        
        # Rerank results
        if candidates:
            logger.debug(f"Reranking {len(candidates)} candidates")
            reranked = self._rerank_results(query_text, candidates)
            return reranked
        
        return candidates
    
    def add_to_bm25_index(self, texts: List[str], doc_ids: List[str],
                          metadata: List[Dict] = None) -> None:
        """
        Add documents to the BM25 index (called during ingestion).
        
        Args:
            texts: Document texts
            doc_ids: Document identifiers
            metadata: Optional metadata for each document
        """
        self.hybrid_retriever.add_documents(texts, doc_ids, metadata)
        logger.info(f"Added {len(texts)} documents to BM25 index "
                    f"(total: {self.hybrid_retriever.bm25_doc_count})")

    def rebuild_bm25_from_qdrant(self, batch_size: int = 100) -> Dict:
        """
        Rebuild the BM25 index by scrolling through all documents in Qdrant.
        Use this to populate BM25 for data that was ingested before hybrid was active.
        
        Args:
            batch_size: Number of points to fetch per scroll request
            
        Returns:
            Dict with rebuild statistics
        """
        logger.info("Rebuilding BM25 index from Qdrant text_documents collection...")
        
        all_texts = []
        all_doc_ids = []
        all_metadata = []
        
        offset = None
        total_fetched = 0
        
        try:
            while True:
                # Scroll through the document collection
                results, next_offset = self.text_db.client.scroll(
                    collection_name=self.text_collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not results:
                    break
                
                for point in results:
                    payload = point.payload or {}
                    text = payload.get("text", "")
                    doc_id = payload.get("doc_id", "")
                    
                    if text and doc_id:
                        all_texts.append(text)
                        all_doc_ids.append(doc_id)
                        meta = {k: v for k, v in payload.items() 
                               if k not in ["text", "doc_id"]}
                        all_metadata.append(meta)
                
                total_fetched += len(results)
                offset = next_offset
                
                if next_offset is None:
                    break
                    
                logger.debug(f"Scrolled {total_fetched} points so far...")
            
            if all_texts:
                # Build fresh BM25 index
                self.hybrid_retriever.bm25_index.build_index(
                    all_texts, all_doc_ids, all_metadata
                )
                logger.info(f"BM25 index rebuilt with {len(all_texts)} documents")
            else:
                logger.warning("No text documents found in Qdrant to rebuild BM25 index")
            
            return {
                "success": True,
                "documents_indexed": len(all_texts),
                "total_scrolled": total_fetched
            }
            
        except Exception as e:
            logger.error(f"Error rebuilding BM25 index: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_indexed": len(all_texts)
            }

    def _rerank_results(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Rerank candidates using cross-encoder.
        Returns results with reranker scores, preserving all metadata.
        
        Args:
            query: Query text
            candidates: Candidate results with 'text' field
            
        Returns:
            Reranked results (top-k)
        """
        if not candidates:
            return []
        
        candidate_texts = [c.get("text", "") for c in candidates]
        
        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(candidate_texts) if t.strip()]
        if not valid_indices:
            return candidates[:self.reranker_top_k]
        
        valid_texts = [candidate_texts[i] for i in valid_indices]
        valid_candidates = [candidates[i] for i in valid_indices]
        
        # Get reranked texts
        reranked_texts = reranker(query, valid_texts, self.reranker_top_k)
        
        # Match reranked texts back to candidates by index (more robust than text matching)
        reranked_results = []
        used = set()
        for reranked_text in reranked_texts:
            for i, candidate in enumerate(valid_candidates):
                if i not in used and candidate.get("text", "") == reranked_text:
                    reranked_results.append(candidate)
                    used.add(i)
                    break
        
        return reranked_results[:self.reranker_top_k]
    
    def format_sources(self, results: List[Dict]) -> List[Dict]:
        """
        Format search results for frontend display with readable source information.
        
        Args:
            results: Search results from query()
            
        Returns:
            List of formatted source dictionaries for frontend
        """
        formatted_sources = []
        
        for result in results:
            metadata = result.get("metadata", {})
            modality = metadata.get("modality", "unknown")
            
            if modality == "text":
                source_type = metadata.get("type", "document").upper()
                
                # Handle different text types
                if metadata.get("type") == "transcript":
                    # Transcript source
                    video_id = metadata.get("video_id", "Unknown Video")
                    start_time = metadata.get("start_time", 0)
                    end_time = metadata.get("end_time", 0)
                    
                    # Format timestamp
                    def format_time(seconds):
                        mins = int(seconds) // 60
                        secs = int(seconds) % 60
                        return f"{mins}:{secs:02d}"
                    
                    time_range = f"{format_time(start_time)} - {format_time(end_time)}"
                    
                    formatted_sources.append({
                        "type": "Transcript",
                        "source": video_id,
                        "time": time_range,
                        "content": result.get("text", "")[:200] + "...",
                        "metadata": metadata
                    })
                else:
                    # Document source
                    doc_id = metadata.get("doc_id", "Unknown Document")
                    formatted_sources.append({
                        "type": "Document",
                        "source": doc_id,
                        "content": result.get("text", "")[:200] + "...",
                        "metadata": metadata
                    })
            
            elif modality == "visual":
                # Frame source
                video_id = metadata.get("video_id", "Unknown Video")
                frame_num = metadata.get("frame_num", 0)
                timestamp = metadata.get("timestamp", 0)
                
                # Format timestamp
                mins = int(timestamp) // 60
                secs = int(timestamp) % 60
                time_str = f"{mins}:{secs:02d}"
                
                formatted_sources.append({
                    "type": "Frame",
                    "source": f"{video_id} (Frame {frame_num})",
                    "time": time_str,
                    "frame_num": frame_num,
                    "metadata": metadata
                })
            
            else:
                # Unknown source
                formatted_sources.append({
                    "type": "Unknown",
                    "source": "Unknown Source",
                    "content": result.get("text", "")[:200] + "...",
                    "metadata": metadata
                })
        
        return formatted_sources
    
    def get_stats(self) -> Dict:
        """
        Get RAG system statistics including BM25 index status.
        
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
            "reranker_top_k": self.reranker_top_k,
            "bm25_index_size": self.hybrid_retriever.bm25_doc_count,
            "bm25_ready": self.hybrid_retriever.bm25_ready,
            "hybrid_weights": {
                "bm25": self.hybrid_retriever.bm25_weight,
                "dense": self.hybrid_retriever.dense_weight
            }
        }
