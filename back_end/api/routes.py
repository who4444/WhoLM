
import os
import uuid
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
from supabase import create_client, Client

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks

from .models import ChatRequest, ChatResponse, UploadResponse, YouTubeUploadRequest, SupabaseUploadRequest, ProcessRequest
from services.chatbot.chatbot_with_memory import WhoLM
from ingestion.processing_functions import process_video_upload, process_document_upload
from database.postgres import PostgresDB
from config.config import Config

logger = logging.getLogger(__name__)

router = APIRouter()
chatbot = WhoLM(qdrant_url=Config.QDRANT_URL)

# Initialize Supabase client
supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

# Initialize Postgres for persistent content tracking
try:
    content_db = PostgresDB()
    logger.info("PostgresDB initialized for content tracking")
except Exception as e:
    logger.warning(f"PostgresDB not available, content tracking will be limited: {e}")
    content_db = None

@router.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "WhoLM API is running", "version": "1.0.0"}


@router.post("/upload/get-url")
async def get_upload_url(request: SupabaseUploadRequest):
    """Generate presigned URL for Supabase upload"""
    try:
        # Validate file type based on content type
        allowed_video_types = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo", "video/webm"}
        allowed_doc_types = {"application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"}

        if request.content_type in allowed_video_types:
            content_type = "video"
        elif request.content_type in allowed_doc_types:
            content_type = "document"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {request.content_type}"
            )

        # Generate unique object key
        ext = Path(request.filename).suffix.lower()
        if not ext:
            raise HTTPException(status_code=400, detail="File extension missing")

        object_key = f"uploads/{uuid.uuid4()}{ext}"
        content_id = str(uuid.uuid4())

        # Generate signed URL for upload
        signed_url = supabase.storage.from_(Config.SUPABASE_BUCKET_NAME).create_signed_upload_url(
            object_key  
        )

        # Store upload metadata in database
        if content_db:
            content_db.insert_content(
                content_id=content_id,
                name=request.filename,
                content_type=content_type,
                storage_path=object_key,
                status="pending_upload"
            )

        return {
            "upload_url": signed_url["signedUrl"],
            "storage_path": object_key,
            "content_id": content_id,
            "expires_in": 900
        }

    except Exception as e:
        logger.error(f"Error generating upload URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/process", response_model=UploadResponse)
async def process_uploaded_file(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Process a file that has been uploaded to Supabase"""
    try:
        # Look up upload metadata from database
        upload_metadata = None
        if content_db:
            upload_metadata = content_db.get_content_by_id(request.content_id)

        if not upload_metadata:
            raise HTTPException(status_code=404, detail="Upload not found")

        # Update status in database
        if content_db:
            content_db.update_content_status(request.content_id, "processing")

        # Process in background
        upload_type = upload_metadata.get("content_type", "")
        storage_path = upload_metadata.get("storage_path", "")
        if upload_type == "video":
            background_tasks.add_task(process_supabase_video_background, request.content_id, storage_path, request.name, chatbot.rag_pipeline)
        else:
            background_tasks.add_task(process_supabase_document_background, request.content_id, storage_path, request.name, chatbot.rag_pipeline)

        return UploadResponse(
            success=True,
            message=f"File '{request.name}' processing started.",
            content_id=request.content_id
        )

    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/youtube", response_model=UploadResponse)
async def upload_youtube_video(
    background_tasks: BackgroundTasks,
    request: YouTubeUploadRequest
):
    """Upload and process a YouTube video"""
    try:
        youtube_url = request.youtube_url

        # Generate content ID
        content_id = str(uuid.uuid4())

        # Persist content record to database
        video_display_name = f"YouTube Video ({youtube_url.split('v=')[-1][:11]})"
        if content_db:
            content_db.insert_content(
                content_id=content_id,
                name=video_display_name,
                content_type="youtube_video",
                youtube_url=youtube_url,
                status="processing"
            )

        # Process YouTube video in background
        background_tasks.add_task(process_youtube_video_background, content_id, youtube_url, chatbot.rag_pipeline)

        return UploadResponse(
            success=True,
            message=f"YouTube video processing started. This may take several minutes.",
            content_id=content_id
        )

    except Exception as e:
        logger.error(f"Error uploading YouTube video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat questions"""
    try:
        import asyncio

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Offload synchronous chatbot.chat() to thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: chatbot.chat(session_id=session_id, user_input=request.question)
        )

        return ChatResponse(
            success=True,
            answer=response.get("answer", "No answer provided"),
            sources=response.get("sources", []),
            context_used=response.get("context_used"),
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return ChatResponse(
            success=False,
            error=str(e)
        )


@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get conversation history for a session"""
    try:
        history = chatbot.memory.get_session_history(session_id)
        return history
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def get_sessions():
    """Get all active sessions"""
    try:
        # Get sessions from chatbot memory
        sessions = list(chatbot.memory.active_contexts.keys())
        
        session_info = []
        for session_id in sessions:
            context = chatbot.memory.get_context(session_id)
            if context:
                recent_messages = chatbot.memory.get_recent_messages(session_id, limit=1)
                session_info.append({
                    "session_id": session_id,
                    "user_id": context.user_id,
                    "topic": context.topic,
                    "last_activity": context.last_activity.isoformat() if context.last_activity else None,
                    "recent_message": recent_messages[0].content if recent_messages else None
                })
        
        return session_info
    except Exception as e:
        logger.error(f"Error getting sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/content")
async def get_uploaded_content():
    """Get list of uploaded content from database"""
    try:
        if content_db:
            rows = content_db.get_all_content()
            # Convert to frontend-compatible format
            result = []
            for row in rows:
                result.append({
                    "id": row["content_id"],
                    "content_id": row["content_id"],
                    "name": row["name"],
                    "type": row["content_type"],
                    "youtube_url": row.get("youtube_url"),
                    "upload_time": row["upload_time"].isoformat() if row.get("upload_time") else None,
                    "status": row["status"],
                    "error": row.get("error"),
                })
            return result
        return []
    except Exception as e:
        logger.error(f"Error getting content list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/content/{content_id}")
async def delete_content(content_id: str):
    """Delete uploaded content from database"""
    try:
        if content_db:
            deleted = content_db.delete_content(content_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Content not found")
            return {"success": True, "message": "Content deleted successfully"}
        raise HTTPException(status_code=404, detail="Content not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/content/{content_id}/status")
async def get_content_status(content_id: str):
    """Get processing status for uploaded content from database"""
    try:
        if content_db:
            record = content_db.get_content_by_id(content_id)
            if record:
                return {"status": record["status"], "error": record.get("error")}
        # Fallback to in-memory status
        from ingestion.processing_functions import get_processing_status
        status = get_processing_status(content_id)
        return status
    except Exception as e:
        logger.error(f"Error getting content status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/rebuild-bm25")
async def rebuild_bm25_index():
    """Rebuild the BM25 index from existing Qdrant data."""
    try:
        result = chatbot.rag_pipeline.rebuild_bm25_from_qdrant()
        if result.get("success"):
            return {
                "success": True,
                "message": f"BM25 index rebuilt with {result['documents_indexed']} documents",
                "details": result
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    except Exception as e:
        logger.error(f"Error rebuilding BM25 index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/rag-stats")
async def get_rag_stats():
    """Get RAG pipeline statistics including BM25 index status."""
    try:
        return chatbot.rag_pipeline.get_stats()
    except Exception as e:
        logger.error(f"Error getting RAG stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper to update content status in database
def _update_content_db_status(content_id: str, status: str, error: str = None, result: dict = None):
    """Update content status in Postgres. Silently skips if DB unavailable."""
    if content_db:
        try:
            content_db.update_content_status(content_id, status, error=error, processing_result=result)
        except Exception as e:
            logger.warning(f"Failed to update content status in DB: {e}")


# Background processing functions
def process_supabase_video_background(content_id: str, storage_path: str, video_name: str,
                                       rag_pipeline=None):
    """Process video from Supabase in background"""
    temp_path = None
    try:
        logger.info(f"Starting background processing for Supabase video: {video_name}")
        _update_content_db_status(content_id, "downloading")

        # Download file from Supabase to temporary location
        temp_file = tempfile.NamedTemporaryFile(suffix=Path(storage_path).suffix, delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        response = supabase.storage.from_(Config.SUPABASE_BUCKET_NAME).download(storage_path)
        with open(temp_path, 'wb') as f:
            f.write(response)

        _update_content_db_status(content_id, "processing")

        # Process the video using the existing pipeline
        result = process_video_upload(temp_path, video_name, content_id, rag_pipeline=rag_pipeline)

        if result.get("success"):
            _update_content_db_status(content_id, "completed", result=result)
        else:
            _update_content_db_status(content_id, "failed", error=result.get("error", "Unknown error"))

        logger.info(f"Completed background processing for Supabase video: {video_name}")

    except Exception as e:
        logger.error(f"Error in background video processing: {str(e)}")
        _update_content_db_status(content_id, "failed", error=str(e))
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def process_supabase_document_background(content_id: str, storage_path: str, doc_name: str,
                                          rag_pipeline=None):
    """Process document from Supabase in background"""
    temp_path = None
    try:
        logger.info(f"Starting background processing for Supabase document: {doc_name}")
        _update_content_db_status(content_id, "downloading")

        # Download file from Supabase to temporary location
        tmp_dir = Config.TEMP_DIR
        os.makedirs(tmp_dir, exist_ok=True)
        temp_path = os.path.join(tmp_dir, f"{uuid.uuid4()}{Path(storage_path).suffix}")
        
        response = supabase.storage.from_(Config.SUPABASE_BUCKET_NAME).download(storage_path)
        with open(temp_path, 'wb') as f:
            f.write(response)

        _update_content_db_status(content_id, "processing")

        # Process the document using the existing pipeline
        result = process_document_upload(temp_path, doc_name, content_id, rag_pipeline=rag_pipeline)

        if result.get("success"):
            _update_content_db_status(content_id, "completed", result=result)
        else:
            _update_content_db_status(content_id, "failed", error=result.get("error", "Unknown error"))

        logger.info(f"Completed background processing for Supabase document: {doc_name}")

    except Exception as e:
        logger.error(f"Error in background Supabase document processing: {str(e)}")
        _update_content_db_status(content_id, "failed", error=str(e))
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def process_youtube_video_background(content_id: str, youtube_url: str, rag_pipeline=None):
    """Process YouTube video in background"""
    try:
        logger.info(f"Starting background processing for YouTube video: {youtube_url}")
        _update_content_db_status(content_id, "downloading")

        # Extract video ID from URL
        import re
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
        if not video_id_match:
            raise ValueError("Invalid YouTube URL")

        video_id = video_id_match.group(1)
        video_name = f"youtube_{video_id}"

        # Download video using yt-dlp
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, f"{video_name}.mp4")

        # Download video
        import subprocess
        cmd = [
            "yt-dlp",
            "-f", "best[height<=720]",  # Limit quality for faster processing
            "-o", video_path,
            youtube_url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to download video: {result.stderr}")

        _update_content_db_status(content_id, "processing")

        # Process the video using the existing pipeline
        result = process_video_upload(video_path, video_name, content_id, rag_pipeline=rag_pipeline)

        if result.get("success"):
            _update_content_db_status(content_id, "completed", result=result)
        else:
            _update_content_db_status(content_id, "failed", error=result.get("error", "Unknown error"))

        # Clean up temp directory
        shutil.rmtree(temp_dir)

        logger.info(f"Completed background processing for YouTube video: {video_id}")

    except Exception as e:
        logger.error(f"Error in background YouTube video processing: {str(e)}")
        _update_content_db_status(content_id, "failed", error=str(e))