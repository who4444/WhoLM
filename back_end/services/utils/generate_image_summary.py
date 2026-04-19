"""
Generate frame/image captions using Google Gemini Vision.

Used during ingestion to convert visual frames into text descriptions,
so the LLM receives semantic captions instead of useless file paths.
"""

import logging
import base64
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazy-load the Gemini client."""
    global _client
    if _client is None:
        try:
            from google import genai
            from config.config import Config

            _client = genai.Client(api_key=Config.GOOGLE_GEMINI_API_KEY)
            logger.info("Gemini VLM client initialized for frame captioning")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    return _client


def _encode_image(image_path: str) -> Optional[dict]:
    """Load image file and prepare it for Gemini API."""
    path = Path(image_path)
    if not path.exists():
        logger.warning(f"Image not found: {image_path}")
        return None

    # Detect MIME type
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    with open(path, "rb") as f:
        image_bytes = f.read()

    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": base64.b64encode(image_bytes).decode("utf-8"),
        }
    }


def generate_frame_caption(image_path: str, context_hint: str = "") -> str:
    """
    Generate a text caption for a single video frame using Gemini Vision.

    Args:
        image_path: Path to the frame image file
        context_hint: Optional hint about the video content for better captions

    Returns:
        Generated caption string, or empty string on failure
    """
    try:
        client = _get_client()
        image_part = _encode_image(image_path)
        if image_part is None:
            return ""

        prompt = (
            "Describe this video frame in 1-2 detailed sentences. "
            "Focus on: visible objects, text, people, actions, and visual elements. "
            "Be specific and factual."
        )
        if context_hint:
            prompt += f"\nContext: This frame is from a video about: {context_hint}"

        from config.config import Config
        response = client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=[prompt, image_part],
        )

        caption = response.text.strip() if response.text else ""
        logger.debug(f"Caption for {Path(image_path).name}: {caption[:80]}...")
        return caption

    except Exception as e:
        logger.error(f"Failed to generate caption for {image_path}: {e}")
        return ""


def generate_batch_captions(image_paths: List[str], context_hint: str = "",
                            max_batch: int = 50) -> List[str]:
    """
    Generate captions for multiple frames.

    Args:
        image_paths: List of frame image paths
        context_hint: Optional video context hint
        max_batch: Maximum frames to caption (excess are skipped with empty captions)

    Returns:
        List of caption strings (same order as input)
    """
    captions = []
    total = min(len(image_paths), max_batch)

    for i, path in enumerate(image_paths[:max_batch]):
        logger.info(f"Captioning frame {i+1}/{total}: {Path(path).name}")
        caption = generate_frame_caption(path, context_hint)
        captions.append(caption)

    # Pad remaining with empty strings if max_batch was hit
    while len(captions) < len(image_paths):
        captions.append("")

    successful = sum(1 for c in captions if c)
    logger.info(f"Generated {successful}/{len(image_paths)} frame captions")
    return captions
