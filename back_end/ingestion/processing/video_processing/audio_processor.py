from pathlib import Path
import torch
import stable_whisper
import json
import re
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, output_dir='',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 model_size='large-v3'):
        self.output_dir = Path(output_dir)
        self.device = device
        if output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        try:
            self.model = stable_whisper.load_faster_whisper(model_size, device=self.device, compute_type="float16")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")

    def transcribe_audio(self, audio_path, language=None): 
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            result = self.model.transcribe(
                str(audio_path),
                beam_size=5,
                language=language 
            )
            return result
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    def group_into_paragraphs(self, result, min_length=600):
        """
        Merges segments into paragraph-sized chunks suitable for RAG.
        
        Args:
            result: The stable_whisper result object
            min_length: Minimum character count before considering a paragraph 'done'
        """
        chunks = []
        current_chunk = {"text": "", "start": None, "end": None}
        
        for segment in result.segments:
            if current_chunk["start"] is None:
                current_chunk["start"] = segment.start
            
            # Add text (strip whitespace to avoid double spaces)
            text = segment.text.strip()
            current_chunk["text"] += text + " "
            current_chunk["end"] = segment.end
            

            is_long_enough = len(current_chunk["text"]) > min_length
            has_punctuation = text.endswith(('.', '!', '?'))
            
            if is_long_enough and has_punctuation:
                current_chunk["text"] = current_chunk["text"].strip()
                chunks.append(current_chunk)
                current_chunk = {"text": "", "start": None, "end": None}
        
        # Capture any remaining text
        if current_chunk["text"]:
            current_chunk["text"] = current_chunk["text"].strip()
            chunks.append(current_chunk)
            
        return chunks

    def process_audio(self, audio_path, output_dir, language=None):
        audio_path = Path(audio_path)
        output_file = audio_path.stem
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Transcribe
            result = self.transcribe_audio(audio_path, language=language)
            
            # Export RAG JSON 
            rag_chunks = self.group_into_paragraphs(result)
            json_path = self.output_dir / f"{output_file}_transcript.json"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(rag_chunks, f, indent=2, ensure_ascii=False)

            return rag_chunks
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return None
        