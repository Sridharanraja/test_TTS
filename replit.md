# E2-F5-TTS Speech Synthesis Application

## Overview

This is a Streamlit-based text-to-speech (TTS) application that leverages the F5-TTS and E2-TTS models for high-quality speech synthesis. The application provides a web interface for generating synthetic speech from text input, with support for reference audio to clone voices. It integrates with Groq's LLM API for intelligent text generation and preprocessing.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid UI development
- **Layout**: Wide layout with expandable sidebar for configuration
- **Components**: Modular UI components for reusability and maintainability
- **Real-time Processing**: Interactive widgets for parameter adjustment

### Backend Architecture
- **Service-Oriented Design**: Separated concerns into distinct service classes
- **GPU Acceleration**: CUDA support with fallback to CPU processing
- **Audio Processing**: Built on PyTorch and torchaudio for efficient audio manipulation
- **Model Management**: Dynamic loading and caching of TTS models

### Key Design Patterns
- **Service Layer Pattern**: Each major functionality (TTS, Audio, Groq) is encapsulated in its own service class
- **Configuration Management**: Centralized configuration with environment variable support
- **Component-Based UI**: Reusable UI components for consistent interface

## Key Components

### Core Services

1. **TTSService** (`services/tts_service.py`)
   - Manages F5-TTS and E2-TTS model loading and inference
   - Handles audio preprocessing and synthesis
   - GPU optimization for faster processing

2. **GroqService** (`services/groq_service.py`)
   - Integration with Groq's LLM API for text generation
   - Supports various language models (default: llama-3.3-70b-versatile)
   - Configurable generation parameters

3. **AudioService** (`services/audio_service.py`)
   - Audio file validation and preprocessing
   - Format conversion and quality optimization
   - Duration and size constraints enforcement

### Utilities

1. **GPUUtils** (`utils/gpu_utils.py`)
   - GPU detection and monitoring
   - Memory usage tracking
   - Performance recommendations

2. **Config** (`utils/config.py`)
   - Environment-based configuration management
   - Default value fallbacks
   - Validation and type conversion

### UI Components

1. **UIComponents** (`components/ui_components.py`)
   - Reusable Streamlit interface elements
   - GPU status displays
   - Model information panels

## Data Flow

1. **Input Processing**:
   - User provides text input via Streamlit interface
   - Optional reference audio upload for voice cloning
   - Configuration parameters set through UI controls

2. **Text Processing**:
   - Raw text validated and preprocessed
   - Optional LLM enhancement via Groq service
   - Text length and format validation

3. **Audio Processing**:
   - Reference audio (if provided) preprocessed by AudioService
   - Format conversion and quality optimization
   - Duration and sample rate normalization

4. **TTS Synthesis**:
   - Preprocessed inputs fed to TTS models
   - GPU-accelerated inference when available
   - Audio generation with configurable parameters

5. **Output Delivery**:
   - Generated audio saved to temporary files
   - Streamlit audio player for immediate playback
   - Download options for generated content

## External Dependencies

### Core ML Libraries
- **PyTorch**: Deep learning framework for model execution
- **torchaudio**: Audio processing and I/O operations
- **f5-tts**: Specialized TTS model implementation

### API Integrations
- **Groq API**: LLM services for text generation and enhancement
- **Streamlit**: Web application framework

### Audio Processing
- **soundfile**: Audio file I/O operations
- **numpy**: Numerical computations for audio processing

### System Utilities
- **psutil**: System resource monitoring
- **pathlib**: File system operations

## Deployment Strategy

### Environment Setup
- Python 3.8+ required
- CUDA-compatible GPU recommended for optimal performance
- Environment variables for API keys and configuration

### Configuration Management
- Environment-based configuration with sensible defaults
- Configurable model cache directories
- Adjustable performance parameters

### Resource Management
- Automatic GPU detection and fallback
- Memory usage monitoring and optimization
- Temporary file cleanup

### Scalability Considerations
- Service-oriented architecture for easy horizontal scaling
- Configurable batch processing
- Model caching for improved response times

## Changelog

```
Changelog:
- July 08, 2025. Initial setup
- July 08, 2025. Fixed text input session state management
- July 08, 2025. Fixed transcription language control issue - transcription field now properly controls output language
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```