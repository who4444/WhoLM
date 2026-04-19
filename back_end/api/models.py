
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000, description="User question")
    session_id: Optional[str] = Field(None, max_length=255)


class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = []
    context_used: Optional[str] = None
    session_id: Optional[str] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    success: bool
    message: str
    content_id: Optional[str] = None
    error: Optional[str] = None


class SupabaseUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=500)
    content_type: str = Field(..., max_length=255)


class ProcessRequest(BaseModel):
    content_id: str = Field(..., max_length=255)
    name: str = Field(..., min_length=1, max_length=500)


class YouTubeUploadRequest(BaseModel):
    youtube_url: str = Field(..., min_length=10, max_length=500,
                            pattern=r'^https?://(www\.)?(youtube\.com|youtu\.be)/.+')