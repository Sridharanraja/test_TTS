import streamlit as st
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UIComponents:
    """Reusable UI components for the Streamlit application"""
    
    def __init__(self):
        pass
    
    def display_gpu_status(self, gpu_info: Dict):
        """Display GPU status and recommendations"""
        if gpu_info.get("cuda_available"):
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 2])
                
                with col1:
                    st.success("🚀 GPU Detected")
                    st.write(f"**{gpu_info['device_name']}**")
                
                with col2:
                    memory_percent = (gpu_info['memory_allocated'] / gpu_info['memory_total']) * 100
                    st.metric(
                        "GPU Memory",
                        f"{gpu_info['memory_total']}MB",
                        f"{memory_percent:.1f}% used"
                    )
                
                with col3:
                    st.info("✅ Ready for synthesis")
                
                # Show recommendations in expander
                if gpu_info.get("recommendations"):
                    with st.expander("GPU Recommendations"):
                        for rec in gpu_info["recommendations"]:
                            st.write(f"• {rec}")
        else:
            st.warning("⚠️ No GPU detected - using CPU (slower performance)")
            if gpu_info.get("recommendations"):
                for rec in gpu_info["recommendations"]:
                    st.write(f"• {rec}")
    
    def display_model_info(self, model_info: Dict):
        """Display model information"""
        with st.expander("Model Information"):
            for model_name, info in model_info.items():
                st.subheader(model_name)
                st.write(f"**Description**: {info['description']}")
                st.write(f"**Parameters**: {info['parameters']}")
                st.write(f"**Recommended Use**: {info['recommended_use']}")
                st.write("---")
    
    def display_audio_info(self, audio_info: Dict, title: str = "Audio Information"):
        """Display audio file information"""
        if audio_info.get("error"):
            st.error(f"Error: {audio_info['error']}")
        else:
            with st.expander(title):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Duration**: {audio_info.get('duration', 'Unknown')}")
                    st.write(f"**Sample Rate**: {audio_info.get('sample_rate', 'Unknown')}")
                    st.write(f"**Channels**: {audio_info.get('channels', 'Unknown')}")
                
                with col2:
                    st.write(f"**Format**: {audio_info.get('format', 'Unknown')}")
                    st.write(f"**File Size**: {audio_info.get('file_size', 'Unknown')}")
    
    def display_synthesis_progress(self, progress_text: str, progress_value: Optional[float] = None):
        """Display synthesis progress"""
        if progress_value is not None:
            progress_bar = st.progress(progress_value)
        st.info(progress_text)
    
    def display_error_with_suggestions(self, error_msg: str, suggestions: List[str]):
        """Display error with helpful suggestions"""
        st.error(f"Error: {error_msg}")
        
        if suggestions:
            st.subheader("💡 Suggestions:")
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
    
    def display_text_generation_prompt_suggestions(self, category: str = "general") -> Optional[str]:
        """Display prompt suggestions for text generation"""
        suggestions = {
            "general": [
                "Write a professional product announcement",
                "Create an engaging story introduction",
                "Generate educational content explanation",
                "Write a business presentation opening"
            ],
            "announcements": [
                "Create a new product launch announcement",
                "Write a company milestone celebration",
                "Generate a service update notification",
                "Create an event invitation message"
            ],
            "educational": [
                "Explain artificial intelligence in simple terms",
                "Create a tutorial introduction for beginners",
                "Write a summary of recent scientific discoveries",
                "Generate learning objectives for a course"
            ],
            "creative": [
                "Write an opening for a mystery novel",
                "Create a character's emotional monologue",
                "Generate a description of a fantastical place",
                "Write dialogue for a dramatic scene"
            ]
        }
        
        category_suggestions = suggestions.get(category, suggestions["general"])
        
        st.subheader("💡 Prompt Suggestions")
        selected_suggestion = st.selectbox(
            "Choose a suggestion or write your own:",
            [""] + category_suggestions,
            key=f"suggestion_{category}"
        )
        
        return selected_suggestion if selected_suggestion else None
    
    def display_synthesis_results(self, audio_path: str, generation_info: Dict):
        """Display synthesis results with playback and download"""
        st.success("🎉 Speech synthesis completed!")
        
        # Audio playback
        st.audio(audio_path, format='audio/wav')
        
        # Generation information
        with st.expander("Generation Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Model**: {generation_info.get('model', 'Unknown')}")
                st.write(f"**Mode**: {generation_info.get('mode', 'Unknown')}")
                st.write(f"**NFE Steps**: {generation_info.get('nfe_steps', 'Unknown')}")
            
            with col2:
                st.write(f"**CFG Strength**: {generation_info.get('cfg_strength', 'Unknown')}")
                st.write(f"**Processing Time**: {generation_info.get('processing_time', 'Unknown')}")
                st.write(f"**Audio Length**: {generation_info.get('audio_length', 'Unknown')}")
        
        # Download button
        with open(audio_path, 'rb') as audio_file:
            st.download_button(
                label="📥 Download Audio",
                data=audio_file.read(),
                file_name="synthesized_speech.wav",
                mime="audio/wav",
                type="secondary",
                use_container_width=True
            )
    
    def display_validation_results(self, is_valid: bool, message: str):
        """Display validation results"""
        if is_valid:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")
    
    def create_settings_panel(self, current_settings: Dict) -> Dict:
        """Create advanced settings panel"""
        st.subheader("🔧 Advanced Settings")
        
        settings = {}
        
        # Audio processing settings
        with st.expander("Audio Processing"):
            settings["nfe_steps"] = st.slider(
                "NFE Steps",
                min_value=8,
                max_value=32,
                value=current_settings.get("nfe_steps", 16),
                help="Number of Neural Flow Estimation steps. Higher values improve quality but increase processing time."
            )
            
            settings["cfg_strength"] = st.slider(
                "CFG Strength",
                min_value=0.5,
                max_value=3.0,
                value=current_settings.get("cfg_strength", 2.0),
                step=0.1,
                help="Classifier-Free Guidance strength. Controls adherence to input text."
            )
            
            settings["sway_sampling_coef"] = st.slider(
                "Sway Sampling Coefficient",
                min_value=-1.0,
                max_value=1.0,
                value=current_settings.get("sway_sampling_coef", -1.0),
                step=0.1,
                help="Sampling coefficient for generation quality."
            )
        
        # Text generation settings
        with st.expander("Text Generation"):
            settings["temperature"] = st.slider(
                "Creativity (Temperature)",
                min_value=0.1,
                max_value=2.0,
                value=current_settings.get("temperature", 0.7),
                step=0.1,
                help="Controls randomness in text generation. Higher values are more creative."
            )
            
            settings["max_tokens"] = st.slider(
                "Maximum Tokens",
                min_value=100,
                max_value=1000,
                value=current_settings.get("max_tokens", 500),
                step=50,
                help="Maximum number of tokens to generate."
            )
        
        # Performance settings
        with st.expander("Performance"):
            settings["use_gpu"] = st.checkbox(
                "Use GPU (if available)",
                value=current_settings.get("use_gpu", True),
                help="Enable GPU acceleration for faster processing."
            )
            
            settings["clear_cache"] = st.button(
                "Clear Model Cache",
                help="Free up GPU memory by clearing cached models."
            )
        
        return settings
    
    def display_welcome_message(self):
        """Display welcome message and instructions"""
        st.markdown("""
        ### Welcome to E2-F5-TTS Speech Synthesis! 🎤
        
        This application allows you to create high-quality synthetic speech using state-of-the-art models:
        
        **Quick Start:**
        1. 📝 Enter or generate text in the Text Input section
        2. 🎵 Upload a reference audio file (3-30 seconds recommended)
        3. ✍️ Provide transcription of the reference audio
        4. 🚀 Click "Start Synthesis" to generate speech
        
        **Features:**
        - **Model Selection**: Choose between F5-TTS_v1 (high quality) and E2-TTS (fast)
        - **AI Text Generation**: Use Groq AI to generate content
        - **Multiple Modes**: Basic-TTS for single voice, Multi-Speech for conversations
        - **Audio Processing**: Automatic optimization of reference audio
        
        **Tips for Best Results:**
        - Use clear, noise-free reference audio
        - Keep reference audio between 3-15 seconds
        - Ensure accurate transcription of reference audio
        - Use natural, conversational text for better synthesis
        """)
    
    def display_footer(self):
        """Display application footer"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(
                """
                <div style='text-align: center; color: #666; font-size: 0.9em;'>
                    Built with ❤️ using Streamlit<br>
                    Powered by F5-TTS/E2-TTS • Enhanced with Groq AI<br>
                    <small>For educational and research purposes</small>
                </div>
                """, 
                unsafe_allow_html=True
            )
