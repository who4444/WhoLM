import boto3
import os
import uuid
import subprocess

s3_client = boto3.client('s3', region_name='us-east-1')

def download_to_disk(bucket, s3_key):
    try:
        # Create a unique local filename to prevent collisions
        ext = os.path.splitext(s3_key)[1]
        local_filename = f"temp_{uuid.uuid4()}{ext}"
        local_path = os.path.join("/tmp", local_filename)

        print(f"Downloading")

        # Download
        s3_client.download_file(bucket, s3_key, local_path)
        
        return local_path
    except Exception as e:
        print(f"Download failed: {e}")
        # Clean up if a partial file was created
        if 'local_path' in locals() and os.path.exists(local_path):
            os.remove(local_path)
        raise e
    
def get_presigned_stream_url(bucket_name, s3_key):
    """
    Generates a temporary (1 hour) URL that allows external tools 
    to read the file securely.
    """
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': s3_key},
            ExpiresIn=3600  # URL valid for 1 hour
        )
        return url
    except Exception as e:
        print(f"Failed to generate URL: {e}")
        raise e
    
def stream_audio_from_s3(bucket_name, s3_key):
    """
    Streams video from S3, extracts audio, and saves ONLY the audio locally.
    Returns the path to the local .mp3 file.
    """
    # Get the secure Stream URL
    stream_url = get_presigned_stream_url(bucket_name, s3_key)
    
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
    
