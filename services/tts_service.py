import torch
import torchaudio
import numpy as np
import tempfile
import os
from pathlib import Path
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSService:
    """Service for Text-to-Speech synthesis using F5-TTS and E2-TTS models"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.vocoder = None
        self.sample_rate = 24000
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize TTS models"""
        try:
            # Import F5-TTS modules
            from f5_tts.infer.utils_infer import (
                load_vocoder, 
                load_model, 
                preprocess_ref_audio_text, 
                infer_process
            )
            
            self.load_vocoder = load_vocoder
            self.load_model = load_model
            self.preprocess_ref_audio_text = preprocess_ref_audio_text
            self.infer_process = infer_process
            
            # Load vocoder
            logger.info("Loading vocoder...")
            self.vocoder = self.load_vocoder()
            
            logger.info("TTS Service initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import F5-TTS modules: {e}")
            logger.error("Please install F5-TTS: pip install f5-tts")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            raise
    
    def _load_model(self, model_name: str):
        """Load a specific TTS model"""
        if model_name not in self.models:
            try:
                logger.info(f"Loading model: {model_name}")
                
                # Map model names to F5-TTS model identifiers
                model_mapping = {
                    "F5-TTS_v1": "F5TTS_v1_Base",
                    "E2-TTS": "E2TTS_Base"
                }
                
                model_id = model_mapping.get(model_name, "F5TTS_v1_Base")
                self.models[model_name] = self.load_model(model_id, device=self.device)
                
                logger.info(f"Model {model_name} loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
        
        return self.models[model_name]
    
    def synthesize(
        self,
        model_name: str,
        ref_audio_path: str,
        ref_text: str,
        gen_text: str,
        nfe_steps: int = 16,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        mode: str = "Basic-TTS"
    ) -> str:
        """
        Synthesize speech using the specified model
        
        Args:
            model_name: Name of the TTS model to use
            ref_audio_path: Path to reference audio file
            ref_text: Transcription of reference audio
            gen_text: Text to synthesize
            nfe_steps: Number of NFE steps for synthesis
            cfg_strength: CFG strength parameter
            sway_sampling_coef: Sway sampling coefficient
            mode: Synthesis mode (Basic-TTS or Multi-Speech)
        
        Returns:
            Path to generated audio file
        """
        try:
            # Load the specified model
            model = self._load_model(model_name)
            
            # Preprocess reference audio and text
            logger.info("Preprocessing reference audio...")
            ref_audio, ref_text_processed = self.preprocess_ref_audio_text(
                ref_audio_path, 
                ref_text,
                device=self.device
            )
            
            # Handle multi-speech mode
            if mode == "Multi-Speech (Podcast)":
                gen_text = self._format_for_multispeech(gen_text)
            
            # Perform inference
            logger.info(f"Generating speech with {model_name}...")
            generated_audio = self.infer_process(
                model=model,
                ref_audio=ref_audio,
                ref_text=ref_text_processed,
                gen_text=gen_text,
                vocoder=self.vocoder,
                nfe_step=nfe_steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef
            )
            
            # Save output audio
            output_path = self._save_audio(generated_audio)
            
            logger.info(f"Speech synthesis completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise
    
    def _format_for_multispeech(self, text: str) -> str:
        """Format text for multi-speech mode"""
        # Add speaker tags for podcast-style generation
        # This is a simplified implementation
        sentences = text.split('. ')
        formatted_text = ""
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                speaker = "Speaker A" if i % 2 == 0 else "Speaker B"
                formatted_text += f"[{speaker}] {sentence.strip()}. "
        
        return formatted_text.strip()
    
    def _save_audio(self, audio_tensor: torch.Tensor) -> str:
        """Save generated audio tensor to file"""
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                output_path = temp_file.name
            
            # Ensure audio is in correct format
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Normalize audio
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
            
            # Save audio file
            torchaudio.save(
                output_path,
                audio_tensor.cpu(),
                self.sample_rate,
                format="WAV"
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about available models"""
        return {
            "F5-TTS_v1": {
                "description": "Latest F5-TTS model with improved performance",
                "parameters": "335.8M",
                "recommended_use": "High-quality synthesis"
            },
            "E2-TTS": {
                "description": "Efficient E2-TTS model for faster inference",
                "parameters": "333.2M", 
                "recommended_use": "Fast synthesis"
            }
        }
    
    def clear_cache(self):
        """Clear model cache to free GPU memory"""
        self.models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache cleared")
