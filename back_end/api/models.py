
from typing import List, Dict, Optional, Any
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = []
    session_id: Optional[str] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    success: bool
    message: str
    content_id: Optional[str] = None
    error: Optional[str] = None


class SupabaseUploadRequest(BaseModel):
    filename: str
    content_type: str


class ProcessRequest(BaseModel):
    content_id: str
    name: str


class YouTubeUploadRequest(BaseModel):
    youtube_url: str