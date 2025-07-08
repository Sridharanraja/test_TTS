import os
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management for the TTS application"""
    
    def __init__(self):
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables with defaults"""
        config = {
            # API Configuration
            "groq_api_key": os.getenv("GROQ_API_KEY", ""),
            
            # Model Configuration
            "default_model": os.getenv("TTS_DEFAULT_MODEL", "F5-TTS_v1"),
            "model_cache_dir": os.getenv("MODEL_CACHE_DIR", str(Path.home() / ".cache" / "f5-tts")),
            
            # Audio Configuration
            "max_audio_duration": int(os.getenv("MAX_AUDIO_DURATION", "30")),
            "target_sample_rate": int(os.getenv("TARGET_SAMPLE_RATE", "24000")),
            "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "50")),
            
            # Generation Configuration
            "default_nfe_steps": int(os.getenv("DEFAULT_NFE_STEPS", "16")),
            "default_cfg_strength": float(os.getenv("DEFAULT_CFG_STRENGTH", "2.0")),
            "default_temperature": float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
            "max_generation_tokens": int(os.getenv("MAX_GENERATION_TOKENS", "500")),
            
            # Performance Configuration
            "enable_gpu": os.getenv("ENABLE_GPU", "true").lower() == "true",
            "gpu_memory_limit": int(os.getenv("GPU_MEMORY_LIMIT", "0")),  # 0 = no limit
            "batch_size": int(os.getenv("BATCH_SIZE", "1")),
            
            # Application Configuration
            "debug_mode": os.getenv("DEBUG_MODE", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "temp_dir": os.getenv("TEMP_DIR", "/tmp"),
            
            # UI Configuration
            "max_text_length": int(os.getenv("MAX_TEXT_LENGTH", "2000")),
            "enable_advanced_settings": os.getenv("ENABLE_ADVANCED_SETTINGS", "true").lower() == "true",
            
            # Safety Configuration
            "content_filter": os.getenv("CONTENT_FILTER", "false").lower() == "true",
            "rate_limit_requests": int(os.getenv("RATE_LIMIT_REQUESTS", "0")),  # 0 = no limit
        }
        
        return config
    
    def _validate_config(self):
        """Validate configuration values"""
        try:
            # Validate paths
            model_cache_dir = Path(self.config["model_cache_dir"])
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            
            temp_dir = Path(self.config["temp_dir"])
            if not temp_dir.exists():
                logger.warning(f"Temp directory does not exist: {temp_dir}")
            
            # Validate numeric ranges
            if not 1 <= self.config["default_nfe_steps"] <= 100:
                logger.warning("NFE steps out of recommended range (1-100)")
            
            if not 0.1 <= self.config["default_cfg_strength"] <= 5.0:
                logger.warning("CFG strength out of recommended range (0.1-5.0)")
            
            if not 0.0 <= self.config["default_temperature"] <= 2.0:
                logger.warning("Temperature out of recommended range (0.0-2.0)")
            
            # Validate API key presence
            if not self.config["groq_api_key"]:
                logger.warning("GROQ_API_KEY not set - AI text generation will not be available")
            
            # Set logging level
            numeric_level = getattr(logging, self.config["log_level"].upper(), logging.INFO)
            logging.getLogger().setLevel(numeric_level)
            
            logger.info("Configuration validated successfully")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            "default_model": self.get("default_model"),
            "cache_dir": self.get("model_cache_dir"),
            "nfe_steps": self.get("default_nfe_steps"),
            "cfg_strength": self.get("default_cfg_strength"),
            "enable_gpu": self.get("enable_gpu"),
            "gpu_memory_limit": self.get("gpu_memory_limit")
        }
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration"""
        return {
            "max_duration": self.get("max_audio_duration"),
            "sample_rate": self.get("target_sample_rate"),
            "max_file_size_mb": self.get("max_file_size_mb")
        }
    
    def get_groq_config(self) -> Dict[str, Any]:
        """Get Groq API configuration"""
        return {
            "api_key": self.get("groq_api_key"),
            "temperature": self.get("default_temperature"),
            "max_tokens": self.get("max_generation_tokens")
        }
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration"""
        return {
            "max_text_length": self.get("max_text_length"),
            "enable_advanced_settings": self.get("enable_advanced_settings"),
            "debug_mode": self.get("debug_mode")
        }
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.get("debug_mode", False)
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance-related settings"""
        return {
            "enable_gpu": self.get("enable_gpu"),
            "gpu_memory_limit": self.get("gpu_memory_limit"),
            "batch_size": self.get("batch_size"),
            "rate_limit": self.get("rate_limit_requests")
        }
    
    def print_config(self):
        """Print current configuration (excluding sensitive data)"""
        safe_config = self.config.copy()
        
        # Mask sensitive information
        if safe_config.get("groq_api_key"):
            safe_config["groq_api_key"] = "***" + safe_config["groq_api_key"][-4:]
        
        logger.info("Current Configuration:")
        for key, value in safe_config.items():
            logger.info(f"  {key}: {value}")
    
    def export_config(self) -> Dict[str, Any]:
        """Export configuration for external use"""
        return self.config.copy()
    
    def validate_requirements(self) -> Dict[str, bool]:
        """Validate system requirements"""
        requirements = {
            "groq_api_available": bool(self.get("groq_api_key")),
            "model_cache_writable": True,
            "temp_dir_writable": True,
            "gpu_available": False
        }
        
        try:
            # Check model cache directory
            cache_dir = Path(self.get("model_cache_dir"))
            test_file = cache_dir / "test_write"
            test_file.touch()
            test_file.unlink()
        except Exception:
            requirements["model_cache_writable"] = False
        
        try:
            # Check temp directory
            temp_dir = Path(self.get("temp_dir"))
            test_file = temp_dir / "test_write"
            test_file.touch()
            test_file.unlink()
        except Exception:
            requirements["temp_dir_writable"] = False
        
        try:
            # Check GPU availability
            import torch
            requirements["gpu_available"] = torch.cuda.is_available()
        except Exception:
            pass
        
        return requirements
