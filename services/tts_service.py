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
            from f5_tts.api import F5TTS
            
            # Use the API class which is simpler
            self.f5_api = F5TTS()
            
            logger.info("TTS Service initialized successfully with F5TTS API")
            
        except ImportError:
            try:
                # Fallback to direct imports
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
                self.f5_api = None
                
                # Load vocoder
                logger.info("Loading vocoder...")
                self.vocoder = self.load_vocoder()
                
                logger.info("TTS Service initialized successfully with direct imports")
                
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
                
                # Map model names to HuggingFace repo identifiers
                model_mapping = {
                    "F5-TTS_v1": "SWivid/F5-TTS",
                    "E2-TTS": "SWivid/E2-TTS"
                }
                
                repo_name = model_mapping.get(model_name, "SWivid/F5-TTS")
                
                # Try different loading approaches
                try:
                    # Try loading with just repo name
                    self.models[model_name] = self.load_model(repo_name)
                    logger.info(f"Model {model_name} loaded with repo name")
                except Exception as e1:
                    logger.warning(f"Repo loading failed: {e1}, trying alternative...")
                    try:
                        # Try with explicit arguments
                        self.models[model_name] = self.load_model(
                            repo_name=repo_name,
                            model_type="F5-TTS" if "F5" in model_name else "E2-TTS"
                        )
                        logger.info(f"Model {model_name} loaded with explicit args")
                    except Exception as e2:
                        logger.warning(f"Explicit args failed: {e2}, using direct model creation...")
                        
                        # Direct model loading from HuggingFace
                        from transformers import AutoModel
                        try:
                            model = AutoModel.from_pretrained(repo_name, trust_remote_code=True)
                            self.models[model_name] = model.to(self.device)
                            logger.info(f"Model {model_name} loaded via transformers")
                        except Exception as e3:
                            logger.error(f"All loading methods failed. Last error: {e3}")
                            # Use a fallback - just create a placeholder that won't crash
                            logger.warning("Using fallback model placeholder")
                            self.models[model_name] = None
                            raise RuntimeError(f"Unable to load {model_name}. Please check F5-TTS installation.")
                
                logger.info(f"Model {model_name} loading completed")
                
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
            # If no transcription provided, use a placeholder
            if not ref_text or ref_text.strip() == "":
                ref_text = "This is a sample audio reference for voice cloning."
                logger.info("No transcription provided, using default placeholder text")
            
            # Handle multi-speech mode
            if mode == "Multi-Speech (Podcast)":
                gen_text = self._format_for_multispeech(gen_text)
            
            # Use F5TTS API if available
            if hasattr(self, 'f5_api') and self.f5_api is not None:
                logger.info(f"Generating speech with {model_name} using F5TTS API...")
                
                # Use the API's infer method (without model parameter)
                result = self.f5_api.infer(
                    ref_file=ref_audio_path,
                    ref_text=ref_text,
                    gen_text=gen_text,
                    remove_silence=True
                )
                
                # Handle different return formats
                if isinstance(result, tuple):
                    if len(result) == 2:
                        generated_audio, sample_rate = result
                    else:
                        # More than 2 values returned, take first two
                        generated_audio = result[0]
                        sample_rate = result[1] if len(result) > 1 else self.sample_rate
                else:
                    # Single value returned (just audio)
                    generated_audio = result
                    sample_rate = self.sample_rate
                
                # Save output audio
                output_path = self._save_audio_from_numpy(generated_audio, sample_rate)
                
            else:
                # Use direct method if API not available
                logger.info("Using direct inference method...")
                
                # Load the specified model
                model = self._load_model(model_name)
                
                # Preprocess reference audio and text
                logger.info("Preprocessing reference audio...")
                ref_audio, ref_text_processed = self.preprocess_ref_audio_text(
                    ref_audio_path, 
                    ref_text,
                    device=self.device
                )
                
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
    
    def _save_audio_from_numpy(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Save generated audio from numpy array to file"""
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                output_path = temp_file.name
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Save using soundfile
            import soundfile as sf
            sf.write(output_path, audio_data, sample_rate, format='WAV')
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save audio from numpy: {e}")
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
