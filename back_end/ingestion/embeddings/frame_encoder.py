import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import re
from PIL import Image  

from config.config import Config
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import open_clip
import numpy as np

logger = logging.getLogger(__name__)


class FrameDataset(Dataset):
    def __init__(self, image_files, preprocess, max_image_size: Tuple[int, int] = (512, 512), fps: float = 1.0):
        self.image_files = image_files
        self.preprocess = preprocess
        self.max_image_size = max_image_size
        self.fps = fps # Added to calculate timestamp from frame number
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx):    
        img_path = self.image_files[idx]
        
        try:
            image = Image.open(img_path)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize logic (optional, CLIP preprocess usually handles this)
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            tensor = self.preprocess(image)
            
            # Extract metadata
            video_id = Path(img_path).parent.name
            
            # Parse video_id and frame info from filename
            video_id, frame_num, timestamp = self._parse_filename(Path(img_path).name, video_id)
            
            metadata = {
                "video_id": video_id,
                "frame_num": frame_num,
                "timestamp": timestamp,
                "file_path": str(img_path)
            }
            
            return tensor, metadata
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            return None, None
    
    def _parse_filename(self, filename: str, fallback_video_name: str) -> tuple:
        """
        Parses filenames like 'videoid_frame_frame_num.jpg'
        """
        try:
            base, _ = os.path.splitext(filename)
            
            # Match pattern: videoid_frame_frame_num
            match = re.search(r"^(.+)_frame_(\d+)$", base)
            
            if match:
                video_id = match.group(1)
                frame_num = int(match.group(2))
                # Calculate timestamp based on frame number and FPS
                timestamp = frame_num / self.fps 
                return video_id, frame_num, timestamp
            else:
                logger.warning(f"Filename {filename} did not match 'videoid_frame_frame_num' format.")
                return fallback_video_name, 0, 0.0
                
        except Exception as e:
            logger.warning(f"Could not parse filename {filename}: {e}")
            return fallback_video_name, 0, 0.0

    @staticmethod
    def collate_fn(batch):
        tensors = []
        metadata_list = []
        
        for tensor, metadata in batch:
            if tensor is not None and metadata is not None:
                tensors.append(tensor)
                metadata_list.append(metadata)
        
        if not tensors:
            return None, None
            
        return torch.stack(tensors), metadata_list

class FrameEncoder:
    def __init__(self, config: Config = None):
        self.config = config if config else Config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Loading CLIP model: {self.config.CLIP_MODEL_NAME}...")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=self.config.CLIP_MODEL_NAME,
            pretrained=self.config.CLIP_PRETRAINED,
            precision='fp16' if self.device == 'cuda' else 'fp32'
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
              
    def process_single_video(self, image_files: List[str]) -> np.ndarray:
        """Process video frames and return embeddings array aligned with input image files.
        
        Args:
            image_files: List of paths to frame images
            
        Returns:
            np.ndarray: Embeddings array with shape (n_frames, embedding_dim)
        """
        if not image_files:
            logger.warning("No image files provided")
            return np.array([])
            
        # Normalize paths and filter existing files
        image_files = [str(f) for f in image_files]
        existing_files = [f for f in image_files if Path(f).exists()]
        
        if not existing_files:
            logger.error(f"No valid image files found. Checked paths: {image_files[:3]}...")
            return np.array([])
        
        if len(existing_files) < len(image_files):
            logger.warning(f"Only {len(existing_files)}/{len(image_files)} image files exist")
        
        video_name = Path(existing_files[0]).parent.name
        embeddings_list = []
        
        try:
            dataset = FrameDataset(
                image_files=existing_files,
                preprocess=self.preprocess,
                fps=Config.VIDEO_KEYFRAME_FPS 
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                num_workers=self.config.WORKER_NUM,
                shuffle=False,
                pin_memory=True if self.device == 'cuda' else False,
                collate_fn=FrameDataset.collate_fn
            )
            
            for batch_tensors, batch_metadata in dataloader:
                if batch_tensors is None: 
                    continue
                
                # Move to device
                if self.device == 'cuda':
                    batch_tensors = batch_tensors.half().to(self.device)
                else:
                    batch_tensors = batch_tensors.to(self.device)
                
                with torch.no_grad():
                    clip_features = self.clip_model.encode_image(batch_tensors)
                    
                    clip_features = clip_features.cpu().numpy().astype(np.float32)
                    # Normalize features
                    clip_features /= np.linalg.norm(clip_features, axis=1, keepdims=True)
                
                # Collect embeddings
                for clip_emb in clip_features:
                    embeddings_list.append(clip_emb)
            
            if embeddings_list:
                return np.array(embeddings_list, dtype=np.float32)
            else:
                logger.error(f"No embeddings generated for video {video_name}")
                return np.array([])
            
        except Exception as e:
            logger.error(f"Error processing video {video_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.array([])
        