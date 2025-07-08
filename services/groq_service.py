import os
import logging
from typing import Optional, List, Dict
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroqService:
    """Service for LLM text generation using Groq API"""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.client = None
        self.model = "llama-3.3-70b-versatile"
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Groq client"""
        try:
            if not self.api_key:
                logger.warning("GROQ_API_KEY not found in environment variables")
                self.client = None
                return
            
            self.client = Groq(api_key=self.api_key)
            logger.info("Groq client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.client = None
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text using Groq LLM
        
        Args:
            prompt: User prompt for text generation
            max_tokens: Maximum tokens to generate
            temperature: Creativity level (0-2)
            system_prompt: Optional system prompt
        
        Returns:
            Generated text
        """
        try:
            if not self.client:
                raise ValueError("Groq client not initialized")
            
            # Prepare messages
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({
                    "role": "system", 
                    "content": "You are a helpful assistant that generates high-quality text for text-to-speech synthesis. Create clear, natural-sounding content that will work well when spoken aloud."
                })
            
            messages.append({"role": "user", "content": prompt})
            
            # Generate text
            logger.info("Generating text with Groq...")
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            
            generated_text = response.choices[0].message.content
            logger.info("Text generation completed")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    def generate_tts_optimized_text(self, prompt: str) -> str:
        """Generate text optimized for TTS synthesis"""
        system_prompt = """You are an expert at creating text optimized for text-to-speech synthesis. 

Guidelines:
1. Use clear, natural language that sounds good when spoken
2. Avoid complex punctuation and formatting
3. Use short to medium sentences
4. Include natural pauses with commas and periods
5. Avoid abbreviations and acronyms
6. Write in a conversational tone
7. Keep the content engaging and easy to listen to

Generate content that will sound natural and professional when converted to speech."""
        
        return self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=400
        )
    
    def generate_podcast_content(self, topic: str, num_speakers: int = 2) -> str:
        """Generate podcast-style conversation content"""
        system_prompt = f"""You are creating content for a {num_speakers}-person podcast conversation. 

Guidelines:
1. Format as natural dialogue between speakers
2. Use [Speaker A], [Speaker B] format for speaker identification
3. Create engaging, informative conversation
4. Include natural transitions and responses
5. Keep individual responses conversational length
6. Make it sound like a real discussion
7. Include questions, agreements, and natural conversational flow

Topic: {topic}"""
        
        prompt = f"Create a {num_speakers}-person podcast conversation about: {topic}"
        
        return self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.8,
            max_tokens=600
        )
    
    def enhance_text_for_speech(self, original_text: str) -> str:
        """Enhance existing text for better TTS output"""
        prompt = f"""Please improve the following text to make it more suitable for text-to-speech synthesis:

Original text: {original_text}

Improvements needed:
1. Make it more conversational and natural-sounding
2. Improve sentence flow and rhythm
3. Add appropriate pauses
4. Ensure clarity when spoken
5. Maintain the original meaning

Provide only the improved version:"""
        
        return self.generate_text(
            prompt=prompt,
            temperature=0.5,
            max_tokens=len(original_text.split()) * 2
        )
    
    def get_text_suggestions(self, category: str) -> List[str]:
        """Get text suggestions for different categories"""
        suggestions_map = {
            "announcements": [
                "Create a professional product launch announcement",
                "Write a company news update",
                "Generate a service announcement",
                "Create an event invitation"
            ],
            "educational": [
                "Explain a complex concept in simple terms",
                "Create a tutorial introduction",
                "Write an educational summary",
                "Generate learning objectives"
            ],
            "creative": [
                "Write a short story opening",
                "Create a character monologue",
                "Generate a creative description",
                "Write a dramatic scene"
            ],
            "business": [
                "Create a professional presentation intro",
                "Write a business proposal summary",
                "Generate a company overview",
                "Create a professional greeting"
            ]
        }
        
        return suggestions_map.get(category, [
            "Generate engaging content on any topic",
            "Create professional announcements",
            "Write educational explanations",
            "Generate creative narratives"
        ])
    
    def is_available(self) -> bool:
        """Check if Groq service is available"""
        return self.client is not None and bool(self.api_key)
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model"""
        return {
            "model": self.model,
            "provider": "Groq",
            "description": "Fast LLM inference for text generation",
            "capabilities": "Text generation, conversation, creative writing"
        }
