from supabase import create_client, Client
import os
import uuid
import subprocess
from config.config import Config
import logging

logger = logging.getLogger(__name__)

SUPABASE_BUCKET_NAME = Config.SUPABASE_BUCKET_NAME
supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

def download_to_disk(storage_path, bucket_name=None):
    """Download file from Supabase to disk"""
    try:
        # Create a unique local filename to prevent collisions
        ext = os.path.splitext(storage_path)[1]
        local_filename = f"temp_{uuid.uuid4()}{ext}"
        local_path = os.path.join("/tmp", local_filename)

        logger.info(f"Downloading from Supabase")

        # Download from Supabase
        response = supabase.storage.from_(SUPABASE_BUCKET_NAME).download(storage_path)
        with open(local_path, 'wb') as f:
            f.write(response)
        
        return local_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        # Clean up if a partial file was created
        if 'local_path' in locals() and os.path.exists(local_path):
            os.remove(local_path)
        raise e
    
def get_presigned_stream_url(storage_path, bucket_name=None):
    """
    Generates a temporary (1 hour) URL that allows external tools 
    to read the file securely.
    """
    try:
        signed_url = supabase.storage.from_(SUPABASE_BUCKET_NAME).create_signed_url(
            storage_path,
            3600  # URL valid for 1 hour
        )
        return signed_url["signedURL"]
    except Exception as e:
        logger.error(f"Failed to generate URL: {e}")
        raise e
    
def stream_audio_from_supabase(storage_path, bucket_name=None):
    """
    Streams video from Supabase, extracts audio, and saves the audio locally.
    Returns the path to the local .mp3 file.
    """
    # Get the secure Stream URL
    stream_url = get_presigned_stream_url(storage_path, bucket_name)
    
    # Define Output Path
    output_audio_path = f"/tmp/audio_{uuid.uuid4()}.mp3"
    
    # FFmpeg Command
    command = [
        "ffmpeg",
        "-i", stream_url,
        "-vn", 
        "-acodec", "libmp3lame",
        "-ar", "16000",       # 16kHz for Whisper optimization
        "-ac", "1",           # Mono channel
        "-y",                 # Overwrite output if exists
        output_audio_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_audio_path

    except subprocess.CalledProcessError as e:
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        raise e
    
