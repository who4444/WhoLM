import logging
import os
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import numpy as np

from ingestion.vector_ingester import VectorIngester
from ingestion.processing.video_processing.audio_processor import AudioProcessor
from ingestion.processing.video_processing.frame_processor import FrameExtractor
from ingestion.processing.document_processing.document_processor import DocumentProcessor
from ingestion.embeddings.text_encoder import encode_texts
from ingestion.embeddings.frame_encoder import FrameEncoder
from config.config import Config

logger = logging.getLogger(__name__)

def process_video_upload(file_path: str, video_name: str) -> Dict[str, Any]:
    """
    Process an uploaded video file: extract audio & frames → encode embeddings → store in Qdrant.

    Args:
        file_path: Path to the uploaded video file
        video_name: Name of the video

    Returns:
        Dict with processing results
    """
    try:
        logger.info(f"Processing video: {video_name}")

        # Initialize processing components
        ingester = VectorIngester()
        audio_processor = AudioProcessor()
        frame_extractor = FrameExtractor()
        frame_encoder = FrameEncoder()

        # Step 1: Extract audio and transcribe
        logger.info("Step 1: Extracting audio and transcribing...")
        transcript_result = audio_processor.process_audio(file_path, Config.TEMP_DIR)

        # Step 2: Extract frames
        logger.info("Step 2: Extracting video frames...")
        frames_result = frame_extractor.process_video(file_path, Config.TEMP_DIR)

        # Step 3: Generate embeddings and store
        logger.info("Step 3: Generating and storing embeddings...")

        # Transcript embeddings: encode → store in Qdrant
        if transcript_result:
            logger.info("Processing transcript embeddings...")
            transcript_texts = [chunk["text"] for chunk in transcript_result]
            
            # Encode transcript texts
            transcript_embeddings = encode_texts(transcript_texts)
            transcript_embeddings_array = np.array(transcript_embeddings)
            
            # Push transcript embeddings to Qdrant
            ingester.push_video_transcript(
                video_id=video_name,
                transcript_chunks=transcript_result,
                embeddings=transcript_embeddings_array
            )
            logger.info(f"Stored {len(transcript_texts)} transcript chunks")
            
        # Frame embeddings: encode → store in Qdrant
        if frames_result:
            logger.info("Processing frame embeddings...")
            frame_embeddings = frame_encoder.process_single_video(frames_result)

            # Store frame embeddings to Qdrant
            ingester.push_frame_embeddings(frames_result, frame_embeddings)
            logger.info(f"Stored {len(frames_result)} frame embeddings")

        logger.info(f"Successfully processed video: {video_name}")

        return {
            "success": True,
            "video_name": video_name,
            "transcript_chunks": len(transcript_result) if transcript_result else 0,
            "frames_extracted": len(frames_result) if frames_result else 0,
            "message": f"Video '{video_name}' processed successfully"
        }

    except Exception as e:
        logger.error(f"Error processing video {video_name}: {str(e)}")
        return {"success": False, "error": str(e)}

def process_document_upload(file_path: str, doc_name: str) -> Dict[str, Any]:
    """
    Process an uploaded document file: extract text → encode embeddings → store in Qdrant.

    Args:
        file_path: Path to the uploaded document file
        doc_name: Name of the document

    Returns:
        Dict with processing results
    """
    try:
        logger.info(f"Processing document: {doc_name}")

        # Initialize components
        ingester = VectorIngester()
        doc_processor = DocumentProcessor()

        # Step 1: Document Processing - Extract text from document
        logger.info("Step 1: Processing document content...")
        doc_result = doc_processor.process_file(file_path)

        if not doc_result.get("success"):
            return {"success": False, "error": "Failed to process document"}

        text_chunks = doc_result.get("text_chunks", [])

        if not text_chunks:
            return {"success": False, "error": "No text content found in document"}

        logger.info(f"Extracted {len(text_chunks)} text chunks from document")

        # Step 2: Text Encoding - Generate embeddings for text chunks
        logger.info("Step 2: Generating text embeddings...")
        embeddings = encode_texts(text_chunks)
        embeddings_array = np.array(embeddings)
        logger.info(f"Generated embeddings with shape: {embeddings_array.shape}")

        # Step 3: Vector Storage - Push embeddings to Qdrant
        logger.info("Step 3: Storing embeddings in Qdrant...")
        ingest_result = ingester.push_document_embeddings(
            doc_id=doc_name,
            text_chunks=text_chunks,
            embeddings=embeddings_array,
            batch_size=100
        )

        if not ingest_result:
            return {"success": False, "error": "Failed to store embeddings in Qdrant"}

        logger.info(f"Successfully processed and stored document: {doc_name}")
        logger.debug(f"Ingestion result: {ingest_result}")

        return {
            "success": True,
            "doc_name": doc_name,
            "text_chunks": len(text_chunks),
            "embeddings_stored": ingest_result.get("successful_batches", 0) * ingest_result.get("batch_size", 0),
            "message": f"Document '{doc_name}' processed successfully with {len(text_chunks)} chunks"
        }

    except Exception as e:
        logger.error(f"Error processing document {doc_name}: {str(e)}")
        return {"success": False, "error": str(e)}

def get_processing_status(content_id: str) -> Dict[str, Any]:
    """
    Get processing status for uploaded content.

    Args:
        content_id: ID of the uploaded content

    Returns:
        Dict with status information
    """
    # This would typically query a database
    # For now, return a placeholder
    return {
        "content_id": content_id,
        "status": "unknown",
        "message": "Status tracking not implemented yet"
    }