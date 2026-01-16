from typing import List, Dict, Any
from logging import getLogger

logger = getLogger(__name__)


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