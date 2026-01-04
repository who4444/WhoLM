from typing import Tuple
class Config:

    # Processing:
    BATCH_SIZE = 64
    WORKER_NUM = 4

    # Paths
    KEYFRAMES_DIR: str = ""
    TEMP_DIR: str = "back_end/tmp"

    # CLIP configs
    MAX_IMAGE_SIZE: Tuple[int, int] = (512, 512)
    CLIP_VECTOR_DIM = 512
    CLIP_MODEL_NAME = 'ViT-B-16-SigLIP-384'
    CLIP_PRETRAINED = 'webli'

    # Qdrant configs
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_DOC_COLLECTION_NAME: str = "documents"
    QDRANT_VD_COLLECTION_NAME: str = "videos"
    QDRANT_TEXT_EMBEDDING_DIM: int = 1024  
    QDRANT_IMAGE_EMBEDDING_DIM: int = 512
    
    # RAG configuration
    RAG_BM25_WEIGHT: float = 0.5
    RAG_DENSE_WEIGHT: float = 0.5
    RAG_RERANKER_TOP_K: int = 3
    RAG_RETRIEVER_TOP_K: int = 10   

    # AWS S3 configuration
    S3_BUCKET = ""
    AWS_REGION = ""

    # Other configurations
    DEDUP_THRESHOLD: float = 0.9
    DEDUP_WINDOW : int = 5  
    VIDEO_KEYFRAME_FPS: float = 1.0  # Extract 1 frame per second