import boto3
import subprocess
import os
import json
import math
import re

s3_client = boto3.client('s3')

class IngestionRouter:
    def __init__(self, bucket_name):
        self.bucket = bucket_name

    def get_stream_url(self, key):
        return s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': key},
            ExpiresIn=300
        )
    
    def audio_check(self, stream_url, duration_sec=30, threshold_db=-50.0):
        command = [
            "ffmpeg",
            "-t", str(duration_sec),  # Only check first 30s
            "-i", stream_url,
            "-af", "volumedetect",    # Audio filter to calculate volume
            "-f", "null", "/dev/null" # Discard output
        ]

        try:
            # FFmpeg writes volume stats to stderr
            result = subprocess.run(command, capture_output=True, text=True)
            stderr = result.stderr

            # Regex to find "mean_volume: -25.5 dB"
            match = re.search(r"mean_volume:\s(-?[\d\.]+) dB", stderr)
            if match:
                mean_vol = float(match.group(1))
                # If volume is lower than threshold (e.g., -50dB), it's silence
                return (mean_vol < threshold_db), mean_vol
            
            # If no audio track is found, treat as silent
            return True, -99.9

        except Exception as e:
            print(f"Audio probe failed: {e}")
            return True, -99.9 # Default to silent on error
        
    def video_check(self, stream_url, duration_sec=30):
        """
        Checks if the video is just a static image or shows little changes.
        Technique: We use the 'mpdecimate' filter which drops duplicate frames.
        If we input 100 frames and output < 5, it's static.
        """
        command = [
            "ffmpeg",
            "-t", str(duration_sec),
            "-i", stream_url,
            # mpdecimate drops frames that don't change much
            # setpts recalculates timestamps so we can count frames easily
            "-vf", "mpdecimate,setpts=N/FRAME_RATE/TB", 
            "-f", "null", "-"
        ]
        
        try:
            # We count how many "unique" frames exist by parsing frame= lines
            result = subprocess.run(command, capture_output=True, text=True)
            stderr = result.stderr
            
            # Extract total frame count from the final log line like: "frame=  15 ..."
            # Note: This is an approximation. A more robust way counts lines in debug mode.
            # But usually, if it's static, ffmpeg processes it extremely fast and frame count is low.
            
            # BETTER METHOD: Check if the file has a video stream first
            probe_cmd = [
                "ffprobe", 
                "-v", "error", 
                "-select_streams", "v:0", 
                "-show_entries", "stream=codec_type", 
                "-of", "csv=p=0", 
                stream_url
            ]
            probe_res = subprocess.run(probe_cmd, capture_output=True, text=True)
            if not probe_res.stdout.strip():
                return True 
            return False 

        except Exception:
            return False
        
    def decide_modality(self, s3_key):
        url = self.get_stream_url(s3_key)

        # 1. Check Silence
        is_silent, volume = self.audio_check(url)

        if is_silent:
            return "VISUAL"

        # 2. Check Static (Only if there IS sound)
        is_static = self.video_check(url) 
        
        if is_static:
             return "AUDIO"

        return "AUDIO"