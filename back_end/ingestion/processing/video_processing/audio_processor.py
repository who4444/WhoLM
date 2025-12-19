from pathlib import Path
from faster_whisper import WhisperModel
import torch
import stable_whisper

class AudioProcessor:
    def __init__(self, output_dir='',
                 device = 'cuda' if torch.cuda.is_available() else 'cpu',
                 model_size='large-v3'):
        self.output_dir = Path(output_dir)
        self.device = device
        if output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Load model once during init
        try:
            self.model = stable_whisper.load_faster_whisper(model_size, device=self.device, compute_type="float16")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")

    def transcribe_audio(self, audio_path, return_segments=True, language=None): 
        """Transcribe audio using Whisper model with timestamp-aligned segments
        
        Args:
            audio_path: Path to the audio file
            return_segments: If True, return full result with timestamps. If False, return only text.
            language: Optional language code to use. If None, Whisper will detect.

        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                beam_size = 5  
            )
            results = []
            for segment in segments:
                results.append({
                    "text": segment.text.strip(),
                    "start": segment.start,
                    "end": segment.end
        })
            return results
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
        
    def process_audio(self, audio_path, 
                      language=None):
        audio_path = Path(audio_path)
        
        #Get the base filename without extension
        audio_stem = audio_path.stem
        
        try:

            result = self.transcribe_audio(audio_path, return_segments=True, language=language)


            return result
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
