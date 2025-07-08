import streamlit as st
import torch
import os
import tempfile
from pathlib import Path
import numpy as np

# Import custom services and utilities
from services.tts_service import TTSService
from services.groq_service import GroqService
from services.audio_service import AudioService
from utils.gpu_utils import GPUUtils
from utils.config import Config
from components.ui_components import UIComponents

# Configure page
st.set_page_config(
    page_title="E2-F5-TTS Speech Synthesis",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_services():
    """Initialize all services and check system requirements"""
    try:
        # Initialize configuration
        config = Config()
        
        # Check GPU availability
        gpu_info = GPUUtils.get_gpu_info()
        
        # Initialize services
        tts_service = TTSService()
        groq_service = GroqService()
        audio_service = AudioService()
        ui_components = UIComponents()
        
        return {
            'config': config,
            'gpu_info': gpu_info,
            'tts_service': tts_service,
            'groq_service': groq_service,
            'audio_service': audio_service,
            'ui_components': ui_components
        }
    except Exception as e:
        st.error(f"Failed to initialize services: {str(e)}")
        return None

def main():
    """Main application function"""
    
    # Initialize session state
    if 'services' not in st.session_state:
        with st.spinner("Initializing services..."):
            st.session_state.services = initialize_services()
    
    if not st.session_state.services:
        st.error("Failed to initialize application. Please refresh the page.")
        return
    
    services = st.session_state.services
    
    # Header
    st.title("🎤 E2-F5-TTS Speech Synthesis")
    st.markdown("Advanced text-to-speech synthesis with AI-powered text generation")
    
    # GPU Status Display
    services['ui_components'].display_gpu_status(services['gpu_info'])
    
    # Sidebar - Settings and Configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model Selection
        st.subheader("Model Configuration")
        selected_model = st.selectbox(
            "Choose TTS Model",
            options=["F5-TTS_v1", "E2-TTS"],
            help="F5-TTS_v1 offers better performance, E2-TTS provides faster inference"
        )
        
        # Mode Selection
        mode = st.radio(
            "Synthesis Mode",
            options=["Basic-TTS", "Multi-Speech (Podcast)"],
            help="Basic-TTS for single voice, Multi-Speech for conversation-style content"
        )
        
        # Advanced Settings
        with st.expander("Advanced Settings"):
            nfe_steps = st.slider("NFE Steps", 8, 32, 16, help="Higher values improve quality but increase processing time")
            cfg_strength = st.slider("CFG Strength", 0.5, 3.0, 2.0, step=0.1, help="Controls adherence to input text")
            sway_sampling_coef = st.slider("Sway Sampling", -1.0, 1.0, -1.0, step=0.1, help="Sampling coefficient for generation")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        # Text input tabs
        text_tab1, text_tab2 = st.tabs(["Manual Input", "AI Generate"])
        
        with text_tab1:
            manual_text = st.text_area(
                "Enter text to synthesize",
                height=150,
                placeholder="Type your text here...",
                key="manual_text_input"
            )
            # Store in session state
            if manual_text:
                st.session_state.current_input_text = manual_text
        
        with text_tab2:
            # Groq AI text generation
            if services['groq_service'].is_available():
                prompt = st.text_area(
                    "Describe what you want to generate",
                    height=100,
                    placeholder="E.g., 'Write a professional product announcement for a new smartphone'"
                )
                
                if st.button("Generate Text with AI", type="secondary"):
                    if prompt:
                        with st.spinner("Generating text with Groq AI..."):
                            try:
                                generated_text = services['groq_service'].generate_text(prompt)
                                st.session_state.current_input_text = generated_text
                                st.text_area("Generated Text", value=generated_text, height=150, key="generated_text_display")
                            except Exception as e:
                                st.error(f"Text generation failed: {str(e)}")
                    else:
                        st.warning("Please enter a prompt for AI text generation")
                    
            else:
                st.warning("⚠️ AI Text Generation Unavailable")
                st.info("Groq API key not configured. You can still use manual text input or add your API key to enable AI features.")
                
                # Show sample prompts as static suggestions
                st.subheader("💡 Sample Text Ideas")
                sample_texts = [
                    "Welcome to our revolutionary new product that will change the way you work and live.",
                    "Today, we're announcing exciting updates to our platform that will enhance your experience.",
                    "Join us as we explore the fascinating world of artificial intelligence and its applications.",
                    "This is a story about innovation, creativity, and the power of human imagination."
                ]
                
                for i, sample in enumerate(sample_texts):
                    if st.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                        st.session_state.current_input_text = sample
    
    with col2:
        st.header("🎵 Audio Reference")
        
        # Initialize ref_text
        ref_text = ""
        
        # Audio input tabs
        audio_tab1, audio_tab2 = st.tabs(["Upload Audio", "Record Audio"])
        
        with audio_tab1:
            uploaded_file = st.file_uploader(
                "Upload reference audio",
                type=['wav', 'mp3', 'flac', 'm4a'],
                help="Upload a clear audio sample (3-30 seconds recommended)"
            )
            
            if uploaded_file:
                st.audio(uploaded_file)
                
                # Reference text input
                ref_text = st.text_area(
                    "Transcription of reference audio (optional)",
                    height=100,
                    placeholder="Enter what is said in the reference audio (leave empty for auto-transcription)...",
                    key="ref_text_input"
                )
        
        with audio_tab2:
            # Audio recording component
            st.markdown("**Record Reference Audio**")
            
            # Simple recording interface
            if st.button("🎙️ Start Recording", type="secondary"):
                st.info("Recording functionality requires microphone access. Please use the upload option or implement with streamlit-audio-recorder package.")
            
            # Placeholder for recorded audio
            if 'recorded_audio' in st.session_state:
                st.audio(st.session_state.recorded_audio)
    
    # Get current input text from session state
    input_text = st.session_state.get('current_input_text', '')
    
    # Synthesis Section
    st.header("🚀 Speech Synthesis")
    
    # Show what text will be used for synthesis
    if input_text:
        st.info(f"**Text to synthesize:** {input_text[:100]}{'...' if len(input_text) > 100 else ''}")
    
    # Validation and synthesis
    synthesis_col1, synthesis_col2, synthesis_col3 = st.columns([2, 1, 2])
    
    with synthesis_col2:
        start_button = st.button("🎬 Start Synthesis", type="primary", use_container_width=True)
    
    if start_button:
        # Validate inputs
        validation_errors = []
        
        if not input_text:
            validation_errors.append("Please enter text to synthesize")
        
        if not uploaded_file and 'recorded_audio' not in st.session_state:
            validation_errors.append("Please upload or record reference audio")
        
        # Reference text is now optional
        
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        else:
            # Perform synthesis
            with st.spinner("Synthesizing speech... This may take a few minutes."):
                try:
                    # Prepare audio file
                    if uploaded_file:
                        # Save uploaded file to temporary location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                            temp_file.write(uploaded_file.read())
                            audio_path = temp_file.name
                    else:
                        audio_path = st.session_state.recorded_audio
                    
                    # Preprocess audio
                    processed_audio_path = services['audio_service'].preprocess_audio(audio_path)
                    
                    # Debug logging
                    st.write(f"DEBUG: About to synthesize with text: '{input_text}'")
                    st.write(f"DEBUG: Text length: {len(input_text)}")
                    
                    # Synthesize speech
                    output_audio = services['tts_service'].synthesize(
                        model_name=selected_model,
                        ref_audio_path=processed_audio_path,
                        ref_text=ref_text,
                        gen_text=input_text,
                        nfe_steps=nfe_steps,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                        mode=mode
                    )
                    
                    # Display results
                    st.success("Speech synthesis completed!")
                    
                    # Audio playback
                    st.audio(output_audio, format='audio/wav')
                    
                    # Download button
                    with open(output_audio, 'rb') as audio_file:
                        st.download_button(
                            label="📥 Download Audio",
                            data=audio_file.read(),
                            file_name="synthesized_speech.wav",
                            mime="audio/wav",
                            type="secondary"
                        )
                    
                    # Cleanup temporary files
                    if os.path.exists(audio_path) and uploaded_file:
                        os.unlink(audio_path)
                    if os.path.exists(processed_audio_path):
                        os.unlink(processed_audio_path)
                        
                except Exception as e:
                    st.error(f"Synthesis failed: {str(e)}")
                    st.error("Please check your inputs and try again. Ensure you have sufficient GPU memory.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Built with Streamlit • Powered by F5-TTS/E2-TTS • Enhanced with Groq AI
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
