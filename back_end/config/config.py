from typing import Tuple
import os
from dotenv import load_dotenv
load_dotenv()

class Config:

    # Processing:
    BATCH_SIZE = 64
    WORKER_NUM = 4

    # Paths
    KEYFRAMES_DIR: str = ""
    TEMP_DIR: str = "tmp"

    # CLIP configs
    MAX_IMAGE_SIZE: Tuple[int, int] = (512, 512)
    CLIP_VECTOR_DIM = 768
    CLIP_MODEL_NAME = 'ViT-B-16-SigLIP-512'
    CLIP_PRETRAINED = 'webli'

    # Qdrant configs
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_DOC_COLLECTION: str = "text_documents"  
    QDRANT_FRAME_COLLECTION: str = "frame_embeddings"  
    QDRANT_TEXT_EMBEDDING_DIM: int = 1024  
    QDRANT_FRAME_EMBEDDING_DIM: int = 768
    
    # RAG configuration
    RAG_BM25_WEIGHT: float = 0.5
    RAG_DENSE_WEIGHT: float = 0.5
    RAG_RERANKER_TOP_K: int = 3
    RAG_RETRIEVER_TOP_K: int = 10   

    # Supabase configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
    SUPABASE_BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "video-qa-bucket")

    # Gemini configuration
    GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

    # PostgreSQL configuration
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

    # Other configurations
    DEDUP_THRESHOLD: float = 0.9
    DEDUP_WINDOW : int = 5  
    VIDEO_KEYFRAME_FPS: float = 1.0  # Extract 1 frame per second