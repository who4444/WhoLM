
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
from config.config import Config

logger = logging.getLogger(__name__)

router = APIRouter()
chatbot = WhoLM(qdrant_url=Config.QDRANT_URL)

# Initialize Supabase client
supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

# In-memory storage for uploaded content (in production, use a database)
uploaded_content = []

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

        # Store upload metadata
        upload_metadata = {
            "content_id": content_id,
            "filename": request.filename,
            "content_type": content_type,
            "storage_path": object_key,
            "upload_time": datetime.now().isoformat(),
            "status": "pending_upload"
        }
        uploaded_content.append(upload_metadata)

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
        # Find the upload metadata
        upload_metadata = None
        for content in uploaded_content:
            if content.get("content_id") == request.content_id:
                upload_metadata = content
                break

        if not upload_metadata:
            raise HTTPException(status_code=404, detail="Upload not found")

        # Update status
        upload_metadata["status"] = "processing"
        upload_metadata["name"] = request.name

        # Create content info for tracking
        content_info = {
            "id": request.content_id,
            "name": request.name,
            "type": upload_metadata["content_type"],
            "storage_path": upload_metadata["storage_path"],
            "upload_time": upload_metadata["upload_time"],
            "status": "processing"
        }
        # Add to uploaded content list for frontend display
        uploaded_content.append(content_info)

        # Process in background
        if upload_metadata["content_type"] == "video":
            background_tasks.add_task(process_supabase_video_background, request.content_id, upload_metadata["storage_path"], request.name)
        else:
            background_tasks.add_task(process_supabase_document_background, request.content_id, upload_metadata["storage_path"], request.name)

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

        # Add to uploaded content list
        content_info = {
            "id": content_id,
            "name": f"YouTube Video ({youtube_url.split('v=')[-1][:11]})",
            "type": "youtube_video",
            "youtube_url": youtube_url,
            "upload_time": datetime.now().isoformat(),
            "status": "processing"
        }
        uploaded_content.append(content_info)

        # Process YouTube video in background
        background_tasks.add_task(process_youtube_video_background, content_id, youtube_url)

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
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Ask the chatbot
        response = chatbot.chat(session_id=session_id, user_input=request.question)

        return ChatResponse(
            success=True,
            answer=response.get("answer", "No answer provided"),
            sources=response.get("sources", []),
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
    """Get list of uploaded content"""
    try:
        return uploaded_content
    except Exception as e:
        logger.error(f"Error getting content list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/content/{content_id}")
async def delete_content(content_id: str):
    """Delete uploaded content"""
    try:
        global uploaded_content
        content_to_delete = None

        for content in uploaded_content:
            if content.get("id") == content_id or content.get("content_id") == content_id:
                content_to_delete = content
                break

        if not content_to_delete:
            raise HTTPException(status_code=404, detail="Content not found")

        # Remove from list
        uploaded_content = [c for c in uploaded_content if c.get("id") != content_id and c.get("content_id") != content_id]

        # Clean up file if it exists
        file_path = content_to_delete.get("file_path")
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)

        return {"success": True, "message": "Content deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Background processing functions
def process_supabase_video_background(content_id: str, storage_path: str, video_name: str):
    """Process video from Supabase in background"""
    temp_path = None
    try:
        logger.info(f"Starting background processing for Supabase video: {video_name}")

        # Update status
        for content in uploaded_content:
            if content.get("id") == content_id or content.get("content_id") == content_id:
                content["status"] = "downloading"
                break

        # Download file from Supabase to temporary location
        temp_file = tempfile.NamedTemporaryFile(suffix=Path(storage_path).suffix, delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        response = supabase.storage.from_(Config.SUPABASE_BUCKET_NAME).download(storage_path)
        with open(temp_path, 'wb') as f:
            f.write(response)

        # Update status
        for content in uploaded_content:
            if content.get("id") == content_id or content.get("content_id") == content_id:
                content["status"] = "processing"
                break

        # Process the video using the existing pipeline
        result = process_video_upload(temp_path, video_name)

        # Update status
        for content in uploaded_content:
            if content.get("id") == content_id or content.get("content_id") == content_id:
                if result.get("success"):
                    content["status"] = "completed"
                    content["processing_result"] = result
                else:
                    content["status"] = "failed"
                    content["error"] = result.get("error", "Unknown error")
                break

        logger.info(f"Completed background processing for Supabase video: {video_name}")

    except Exception as e:
        logger.error(f"Error in background S3 video processing: {str(e)}")
        # Update status to failed
        for content in uploaded_content:
            if content.get("id") == content_id or content.get("content_id") == content_id:
                content["status"] = "failed"
                content["error"] = str(e)
                break
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def process_supabase_document_background(content_id: str, storage_path: str, doc_name: str):
    """Process document from Supabase in background"""
    try:
        logger.info(f"Starting background processing for Supabase document: {doc_name}")

        # Update status
        for content in uploaded_content:
            if content.get("id") == content_id or content.get("content_id") == content_id:
                content["status"] = "downloading"
                break

        # Download file from Supabase to temporary location
        tmp_dir = Config.TEMP_DIR
        os.makedirs(tmp_dir, exist_ok=True)
        temp_path = os.path.join(tmp_dir, f"{uuid.uuid4()}{Path(storage_path).suffix}")
        
        response = supabase.storage.from_(Config.SUPABASE_BUCKET_NAME).download(storage_path)
        with open(temp_path, 'wb') as f:
            f.write(response)

        # Update status
        for content in uploaded_content:
            if content.get("id") == content_id or content.get("content_id") == content_id:
                content["status"] = "processing"
                break

        # Process the document using the existing pipeline
        result = process_document_upload(temp_path, doc_name)

        # Update status
        for content in uploaded_content:
            if content.get("id") == content_id or content.get("content_id") == content_id:
                if result.get("success"):
                    content["status"] = "completed"
                    content["processing_result"] = result
                else:
                    content["status"] = "failed"
                    content["error"] = result.get("error", "Unknown error")
                break

        logger.info(f"Completed background processing for Supabase document: {doc_name}")

    except Exception as e:
        logger.error(f"Error in background Supabase document processing: {str(e)}")
        # Update status to failed
        for content in uploaded_content:
            if content.get("id") == content_id or content.get("content_id") == content_id:
                content["status"] = "failed"
                content["error"] = str(e)
                break
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def process_youtube_video_background(content_id: str, youtube_url: str):
    """Process YouTube video in background"""
    try:
        logger.info(f"Starting background processing for YouTube video: {youtube_url}")

        # Update status
        for content in uploaded_content:
            if content.get("id") == content_id:
                content["status"] = "downloading"
                break

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

        # Update status
        for content in uploaded_content:
            if content.get("id") == content_id:
                content["status"] = "processing"
                content["file_path"] = video_path
                break

        # Process the video using the existing pipeline
        result = process_video_upload(video_path, video_name)

        # Update status
        for content in uploaded_content:
            if content.get("id") == content_id:
                if result.get("success"):
                    content["status"] = "completed"
                    content["processing_result"] = result
                else:
                    content["status"] = "failed"
                    content["error"] = result.get("error", "Unknown error")
                break

        # Clean up temp directory
        shutil.rmtree(temp_dir)

        logger.info(f"Completed background processing for YouTube video: {video_id}")

    except Exception as e:
        logger.error(f"Error in background YouTube video processing: {str(e)}")
        # Update status to failed
        for content in uploaded_content:
            if content.get("id") == content_id:
                content["status"] = "failed"
                content["error"] = str(e)
                break