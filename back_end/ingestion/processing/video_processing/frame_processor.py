import imagehash
import subprocess
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class FrameExtractor:
    def __init__(self, output_dir = None):
        if output_dir is None:
            output_dir = "tmp"  # Default to tmp directory
        self.output_dir = Path(output_dir)

    def extract_frames(self, video_path, output_dir):
        """Extract 1 frame per second from the video using ffmpeg.
        """    
        video_path  = Path(video_path)
        video_id = video_path.stem
        output_dir = Path(output_dir) 

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting frames from {video_path} to {output_dir}")

        cmd = ['ffmpeg', '-i', str(video_path),
               '-vf', 'fps=1',
               str(output_dir / f'{video_id}_frame_%05d.jpg')
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"FFmpeg output: {result.stdout}")
            
            # Check if frames were actually created
            frame_files = list(output_dir.glob("*.jpg"))
            if not frame_files:
                raise RuntimeError(f"FFmpeg completed but no frames found in {output_dir}")
            
            logger.info(f"Successfully extracted {len(frame_files)} frames")
            return output_dir
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg stderr: {e.stderr}")
            raise RuntimeError(f"Frame extraction failed: {e.stderr}")
    
    def remove_dupes(self, output_dir, threshold=5, window=10, delete_files=True):
        """
        Remove near-duplicate frames using perceptual hashing with a sliding window.
        
        Args:
            output_dir (str | Path): Directory containing extracted frames (.jpg)
            threshold (int): Max Hamming distance between considered duplicates
            window (int): Number of recent hashes to compare against
            delete_files (bool): Whether to delete duplicate files
            
        Returns:
            list[str]: Filenames of kept frames
            
        Raises:
            RuntimeError: If image processing fails
        """
        output_dir = Path(output_dir)
        img_files = sorted([f for f in os.listdir(output_dir) if f.lower().endswith((".jpg", ".jpeg"))])
        last_hashes = []
        kept = []
        removed_count = 0
        errors = []

        for fname in tqdm(img_files, desc="Removing dupes"):
            path = output_dir / fname
            try:
                with Image.open(path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    h = imagehash.phash(img)
                    
                # Compare with recent hashes
                if not any(abs(h - prev) <= threshold for prev in last_hashes):
                    kept.append(fname)
                else:
                    removed_count += 1
                    if delete_files:
                        try:
                            os.remove(path)
                        except OSError as e:
                            errors.append(f"Failed to delete {fname}: {e}")
                            
                # Slide window
                last_hashes = (last_hashes + [h])[-window:]
                    
            except Exception as e:
                errors.append(f"Failed to process {fname}: {e}")
                continue  # Skip this file on error

        if errors:
            for error in errors[:5]:  # Show first 5 errors
                logger.error(f"- {error}")
            if len(errors) > 5:
                logger.error(f"...and {len(errors) - 5} more errors")
                
        return kept
    
    def process_video(self, video_path, dedup_thr, dedup_window):

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_name  = video_path.stem
        output_dir  = self.output_dir / video_name

        # Extract frames
        self.extract_frames(video_path, output_dir)

        # Check if frames were created
        frame_files = list(output_dir.glob("*.jpg"))
        if not frame_files:
            raise RuntimeError(f"No frames were extracted to {output_dir}")

        logger.info(f"Extracted {len(frame_files)} frames")

        # Remove duplicates
        kept_frames = self.remove_dupes(output_dir, dedup_thr, dedup_window)

        return kept_frames
    
