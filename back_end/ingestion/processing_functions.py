import logging
import os
from pathlib import Path
from typing import Dict, Any, List
import tempfile

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
    Process an uploaded video file.

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
        logger.info("Extracting audio and transcribing...")
        transcript_result = audio_processor.process_audio(file_path, Config.TEMP_DIR)


        # Step 2: Extract frames
        logger.info("Extracting video frames...")
        frames_result = frame_extractor.process_video(file_path, Config.TEMP_DIR)


        # Step 3: Generate embeddings
        logger.info("Generating embeddings...")

        # Text embeddings for transcript
        if transcript_result:
            transcript_texts = [chunk["text"] for chunk in transcript_result]
            transcript_embeddings = encode_texts(transcript_texts)

            # Store transcript embeddings
            ingester.push_video_transcript(video_name, transcript_result, transcript_embeddings)
        # Frame embeddings
        if frames_result:
            frame_embeddings = frame_encoder.process_single_video(frames_result)

            # Store frame embeddings
            ingester.push_frame_embeddings(frames_result, frame_embeddings)

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
    Process an uploaded document file.

    Args:
        file_path: Path to the uploaded document file
        doc_name: Name of the document

    Returns:
        Dict with processing results
    """
    try:
        logger.info(f"Processing document: {doc_name}")

        # Initialize vector ingester
        ingester = VectorIngester()
        doc_processor = DocumentProcessor()

        # Step 1: Process document and extract text
        logger.info("Processing document content...")
        doc_result = DocumentProcessor.process_file(file_path)

        if not doc_result.get("success"):
            return {"success": False, "error": "Failed to process document"}

        text_chunks = doc_result.get("text_chunks", [])

        if not text_chunks:
            return {"success": False, "error": "No text content found in document"}

        # Step 2: Generate embeddings
        logger.info("Generating text embeddings...")
        embeddings = encode_texts(text_chunks)

        # Step 3: Store in vector database
        ingester.push_document_embeddings(doc_name, text_chunks, embeddings)

        logger.info(f"Successfully processed document: {doc_name}")

        return {
            "success": True,
            "doc_name": doc_name,
            "text_chunks": len(text_chunks),
            "message": f"Document '{doc_name}' processed successfully"
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