from typing import List, Dict, Any, Union
from logging import getLogger

logger = getLogger(__name__)


class PayloadWrapper:
    """Wrapper to make dict payloads compatible with hit.payload access."""
    def __init__(self, data: Dict[str, Any]):
        self.payload = data
    
    @property
    def score(self) -> float:
        return self.payload.get("score", 0.0)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def _get_score_str(hit: Any, include_score: bool) -> str:
    """Get formatted score string if include_score is True."""
    if include_score and hasattr(hit, 'score'):
        return f" (Score: {hit.score:.3f})"
    return ""


def build_document_context(hits: List[Any], include_score: bool = False) -> str:
    """
    Build formatted context from document embeddings.
    
    Expected payload: {"modality": "text", "type": "documents", "text": str, "doc_id": str, "chunk_index": int}
    """
    if not hits:
        logger.warning("No document hits provided")
        return ""
    
    context_parts = []
    
    for i, hit in enumerate(hits, start=1):
        try:
            payload = hit.payload
            text = payload.get("text", "").strip()
            
            if not text:
                continue
            
            doc_id = payload.get("doc_id", "unknown")
            chunk_idx = payload.get("chunk_index", 0)
            
            header = f"[{i}] Document {doc_id} (Chunk {chunk_idx})"
            header += _get_score_str(hit, include_score)
            
            context_parts.append(f"{header}:\n{text}")
            
        except Exception as e:
            logger.error(f"Error processing document hit {i}: {str(e)}")
            continue
    
    return "\n\n---\n\n".join(context_parts)


def build_transcript_context(hits: List[Any], include_score: bool = False) -> str:
    """
    Build formatted context from transcript embeddings.
    
    Expected payload: {
        "modality": "text", "type": "transcript", "text": str,
        "video_id": str, "start_time": float, "end_time": float,
        "chunk_index": int
    }
    """
    if not hits:
        logger.warning("No transcript hits provided")
        return ""
    
    context_parts = []
    
    for i, hit in enumerate(hits, start=1):
        try:
            payload = hit.payload
            text = payload.get("text", "").strip()
            
            if not text:
                continue
            
            video_id = payload.get("video_id", "unknown")
            start_time = payload.get("start_time", 0)
            end_time = payload.get("end_time", 0)
            
            time_str = f"[{_format_timestamp(start_time)} - {_format_timestamp(end_time)}]"
            header = f"{time_str} (Video: {video_id})"
            header += _get_score_str(hit, include_score)
            
            context_parts.append(f"{header}:\n{text}")
            
        except Exception as e:
            logger.error(f"Error processing transcript hit {i}: {str(e)}")
            continue
    
    return "\n\n---\n\n".join(context_parts)


def build_frame_context(hits: List[Any], include_score: bool = False) -> str:
    """
    Build formatted context from frame embeddings.
    
    Expected payload: {
        "modality": "visual", "frame_path": str, "frame_filename": str,
        "video_id": str, "frame_num": int, "timestamp": float
    }
    """
    if not hits:
        logger.warning("No frame hits provided")
        return ""
    
    context_parts = []
    
    for i, hit in enumerate(hits, start=1):
        try:
            payload = hit.payload
            
            frame_path = payload.get("frame_path", "unknown")
            frame_num = payload.get("frame_num", 0)
            timestamp = payload.get("timestamp", 0)
            video_id = payload.get("video_id", "unknown")
            
            time_str = _format_timestamp(timestamp)
            header = f"[Frame {frame_num} @ {time_str}] (Video: {video_id})"
            header += _get_score_str(hit, include_score)
            
            context_parts.append(f"{header}:\n{frame_path}")
            
        except Exception as e:
            logger.error(f"Error processing frame hit {i}: {str(e)}")
            continue
    
    return "\n\n---\n\n".join(context_parts)


def build_hybrid_context(hits: List[Any], include_score: bool = False) -> str:
    """
    Build formatted context from mixed modality hits (documents, transcripts, frames).
    Intelligently routes each hit based on its modality.
    """
    if not hits:
        logger.warning("No hits provided")
        return ""
    
    context_parts = []
    
    for i, hit in enumerate(hits, start=1):
        try:
            payload = hit.payload
            modality = payload.get("modality", "").lower()
            content_type = payload.get("type", "").lower()
            
            # Route based on modality and type
            if modality == "visual":
                formatted = _format_frame_hit(payload, i, hit, include_score)
            elif content_type == "transcript":
                formatted = _format_transcript_hit(payload, i, hit, include_score)
            else:  # text/document
                formatted = _format_text_hit(payload, i, hit, include_score)
            
            if formatted:
                context_parts.append(formatted)
                
        except Exception as e:
            logger.error(f"Error processing hybrid hit {i}: {str(e)}")
            continue
    
    return "\n\n---\n\n".join(context_parts)


def _format_text_hit(payload: Dict[str, Any], index: int, hit: Any, include_score: bool) -> str:
    """Format a text/document hit."""
    text = payload.get("text", "").strip()
    if not text:
        return ""
    
    doc_id = payload.get("doc_id", "unknown")
    chunk_idx = payload.get("chunk_index", 0)
    
    header = f"[{index}] {doc_id} (Chunk {chunk_idx})"
    header += _get_score_str(hit, include_score)
    
    return f"{header}:\n{text}"


def _format_transcript_hit(payload: Dict[str, Any], index: int, hit: Any, include_score: bool) -> str:
    """Format a transcript hit."""
    text = payload.get("text", "").strip()
    if not text:
        return ""
    
    video_id = payload.get("video_id", "unknown")
    start_time = payload.get("start_time", 0)
    end_time = payload.get("end_time", 0)
    
    time_str = f"[{_format_timestamp(start_time)} - {_format_timestamp(end_time)}]"
    header = f"{time_str} (Video: {video_id})"
    header += _get_score_str(hit, include_score)
    
    return f"{header}:\n{text}"


def _format_frame_hit(payload: Dict[str, Any], index: int, hit: Any, include_score: bool) -> str:
    """Format a frame hit."""
    frame_path = payload.get("frame_path", "unknown")
    frame_num = payload.get("frame_num", 0)
    timestamp = payload.get("timestamp", 0)
    video_id = payload.get("video_id", "unknown")
    
    time_str = _format_timestamp(timestamp)
    header = f"[Frame {frame_num} @ {time_str}] (Video: {video_id})"
    header += _get_score_str(hit, include_score)
    
    return f"{header}:\n{frame_path}"


def convert_rag_results_to_hits(results: List[Dict[str, Any]]) -> List[Any]:
    """
    Convert RAG pipeline results (dicts) to hit-like objects compatible with context builders.
    
    Args:
        results: List of result dicts from rag_pipeline.query()
        
    Returns:
        List of hit-like objects with payload attributes
    """
    hits = []
    for result in results:
        # Create a wrapper object that has .payload and .score attributes
        wrapper = PayloadWrapper(result.get("metadata", {}))
        # Add text and collection info to payload
        wrapper.payload["text"] = result.get("text", "")
        wrapper.payload["collection_type"] = result.get("collection_type", "")
        wrapper.payload["doc_id"] = result.get("doc_id", "")
        wrapper.payload["score"] = result.get("score", 0.0)
        wrapper.score = result.get("score", 0.0)
        hits.append(wrapper)
    
    return hits


def build_rag_context(results: List[Dict[str, Any]], include_score: bool = True) -> str:
    """
    Build formatted context from RAG pipeline results.
    Intelligently handles mixed modality results from rag_pipeline.query().
    
    Args:
        results: List of result dicts from rag_pipeline.query()
        include_score: Whether to include retrieval scores
        
    Returns:
        Formatted context string for chatbot prompt
    """
    if not results:
        logger.warning("No RAG results provided")
        return ""
    
    # Convert dicts to hit-like objects
    hits = convert_rag_results_to_hits(results)
    
    # Use hybrid context builder to handle mixed modalities
    return build_hybrid_context(hits, include_score=include_score)


def extract_citations_from_context(context: str) -> List[Dict[str, str]]:
    """
    Extract structured citations from formatted context string.
    Parses the formatted context output and creates citation references.
    
    Args:
        context: Formatted context string from build_rag_context
        
    Returns:
        List of citation dictionaries with source info
    """
    citations = []
    
    # Split context by separator
    sections = context.split("\n\n---\n\n")
    
    for i, section in enumerate(sections, 1):
        lines = section.strip().split("\n", 1)
        if len(lines) >= 1:
            header = lines[0]
            content = lines[1] if len(lines) > 1 else ""
            
            citation = {
                "id": i,
                "header": header,
                "content": content[:100] + "..." if len(content) > 100 else content
            }
            citations.append(citation)
    
    return citations