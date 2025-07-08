import os
import tempfile
import logging
import numpy as np
import soundfile as sf
import torchaudio
import torch
from pathlib import Path
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioService:
    """Service for audio processing and validation"""
    
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        self.target_sample_rate = 24000
        self.max_duration = 30  # seconds
        self.min_duration = 1   # seconds
    
    def preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio file for TTS synthesis
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to preprocessed audio file
        """
        try:
            logger.info(f"Preprocessing audio: {audio_path}")
            
            # Validate file
            self._validate_audio_file(audio_path)
            
            # Load audio
            audio_data, sample_rate = self._load_audio(audio_path)
            
            # Process audio
            processed_audio = self._process_audio(audio_data, sample_rate)
            
            # Save processed audio
            output_path = self._save_processed_audio(processed_audio)
            
            logger.info(f"Audio preprocessing completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise
    
    def _validate_audio_file(self, audio_path: str):
        """Validate audio file format and properties"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        file_ext = Path(audio_path).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {file_ext}")
        
        # Check file size (max 50MB)
        file_size = os.path.getsize(audio_path)
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            raise ValueError(f"Audio file too large: {file_size / 1024 / 1024:.1f}MB (max: 50MB)")
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file using torchaudio or soundfile"""
        try:
            # Try torchaudio first
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            audio_data = audio_tensor.numpy()
            
            # Convert to mono if stereo
            if audio_data.shape[0] > 1:
                audio_data = np.mean(audio_data, axis=0)
            else:
                audio_data = audio_data[0]
                
        except Exception:
            try:
                # Fallback to soundfile
                audio_data, sample_rate = sf.read(audio_path)
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                    
            except Exception as e:
                raise ValueError(f"Failed to load audio file: {e}")
        
        return audio_data, sample_rate
    
    def _process_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio data (resample, normalize, trim)"""
        
        # Validate duration
        duration = len(audio_data) / sample_rate
        if duration < self.min_duration:
            raise ValueError(f"Audio too short: {duration:.1f}s (min: {self.min_duration}s)")
        if duration > self.max_duration:
            logger.warning(f"Audio duration {duration:.1f}s exceeds recommended {self.max_duration}s")
            # Trim to max duration
            max_samples = int(self.max_duration * sample_rate)
            audio_data = audio_data[:max_samples]
        
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            logger.info(f"Resampling from {sample_rate}Hz to {self.target_sample_rate}Hz")
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            resampled = torchaudio.functional.resample(
                audio_tensor, 
                orig_freq=sample_rate, 
                new_freq=self.target_sample_rate
            )
            audio_data = resampled.squeeze(0).numpy()
        
        # Normalize audio
        audio_data = self._normalize_audio(audio_data)
        
        # Remove silence from beginning and end
        audio_data = self._trim_silence(audio_data)
        
        return audio_data
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to appropriate level"""
        # Remove DC offset
        audio_data = audio_data - np.mean(audio_data)
        
        # Normalize to 90% of max to prevent clipping
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data * (0.9 / max_val)
        
        return audio_data
    
    def _trim_silence(self, audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from beginning and end of audio"""
        # Find non-silent regions
        non_silent = np.abs(audio_data) > threshold
        
        if not np.any(non_silent):
            # If all audio is below threshold, return original
            return audio_data
        
        # Find first and last non-silent samples
        non_silent_indices = np.where(non_silent)[0]
        start_idx = max(0, non_silent_indices[0] - int(0.1 * self.target_sample_rate))  # Keep 0.1s before
        end_idx = min(len(audio_data), non_silent_indices[-1] + int(0.1 * self.target_sample_rate))  # Keep 0.1s after
        
        return audio_data[start_idx:end_idx]
    
    def _save_processed_audio(self, audio_data: np.ndarray) -> str:
        """Save processed audio to temporary file"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                output_path = temp_file.name
            
            # Save using soundfile
            sf.write(output_path, audio_data, self.target_sample_rate, format='WAV')
            
            return output_path
            
        except Exception as e:
            raise ValueError(f"Failed to save processed audio: {e}")
    
    def get_audio_info(self, audio_path: str) -> dict:
        """Get information about audio file"""
        try:
            audio_data, sample_rate = self._load_audio(audio_path)
            duration = len(audio_data) / sample_rate
            
            return {
                "duration": f"{duration:.2f}s",
                "sample_rate": f"{sample_rate}Hz",
                "channels": "Mono",
                "format": Path(audio_path).suffix.upper().lstrip('.'),
                "file_size": f"{os.path.getsize(audio_path) / 1024:.1f}KB"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def validate_reference_audio(self, audio_path: str) -> Tuple[bool, str]:
        """
        Validate if audio is suitable for TTS reference
        
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            self._validate_audio_file(audio_path)
            audio_data, sample_rate = self._load_audio(audio_path)
            
            duration = len(audio_data) / sample_rate
            
            # Check duration
            if duration < 3:
                return False, f"Audio too short for good reference: {duration:.1f}s (recommended: 3-15s)"
            
            if duration > 20:
                return False, f"Audio too long: {duration:.1f}s (recommended: 3-15s)"
            
            # Check audio level
            max_level = np.max(np.abs(audio_data))
            if max_level < 0.1:
                return False, "Audio level too low - please use a clearer recording"
            
            # Check for clipping
            clipping_ratio = np.sum(np.abs(audio_data) > 0.95) / len(audio_data)
            if clipping_ratio > 0.01:
                return False, f"Audio has clipping ({clipping_ratio*100:.1f}% of samples)"
            
            return True, "Audio is suitable for TTS reference"
            
        except Exception as e:
            return False, f"Audio validation failed: {str(e)}"
    
    def adjust_audio_speed(self, audio_path: str, speed_factor: float) -> str:
        """
        Adjust the speed of audio file
        
        Args:
            audio_path: Path to input audio file
            speed_factor: Speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
            
        Returns:
            Path to speed-adjusted audio file
        """
        try:
            # If no speed change needed, return original
            if abs(speed_factor - 1.0) < 0.01:
                return audio_path
                
            # Load audio
            audio_data, sample_rate = self._load_audio(audio_path)
            
            # Adjust speed using time-stretching (change speed without changing pitch)
            import librosa
            
            # Apply time stretching
            stretched_audio = librosa.effects.time_stretch(audio_data, rate=speed_factor)
            
            # Save the speed-adjusted audio
            output_path = self._save_processed_audio(stretched_audio)
            
            logger.info(f"Speed adjusted audio saved: {output_path} (factor: {speed_factor}x)")
            return output_path
                
        except Exception as e:
            logger.warning(f"Failed to adjust audio speed, using original: {e}")
            return audio_path  # Return original if speed adjustment fails
    
    def cleanup_temp_files(self, file_paths: list):
        """Clean up temporary audio files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup file {file_path}: {e}")
