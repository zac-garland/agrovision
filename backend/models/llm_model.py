"""
LLM model wrapper for Mistral 7B via Ollama.
Provides reasoning and synthesis capabilities for plant diagnosis.
"""

import os
from typing import Optional, Dict, List
from config import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Try to import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  ollama package not installed. LLM synthesis will be unavailable.")
    print("   Install with: pip install ollama")


class LLMModel:
    """Wrapper for Mistral 7B LLM via Ollama."""
    
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize LLM model.
        
        Args:
            model_name: Name of the Ollama model (default: from config)
            base_url: Ollama API base URL (default: http://localhost:11434)
        """
        self.available = False
        self.model_name = model_name or LLM_MODEL_NAME
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        if not OLLAMA_AVAILABLE:
            print("⚠️  Ollama package not available. Install with: pip install ollama")
            return
        
        # Initialize - availability will be checked on first use
        self.available = True
        self.client = None
        print(f"✅ LLM client initialized (model: {self.model_name})")
        print("   Note: Connection will be tested on first generation")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Generate text using the LLM.
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature (default: from config)
            max_tokens: Maximum tokens to generate (default: from config)
            system_prompt: Optional system prompt for context
            
        Returns:
            Dictionary with:
            - 'text': Generated text
            - 'success': Boolean
            - 'error': Error message if failed
        """
        if not self.available:
            return {
                'text': '',
                'success': False,
                'error': 'LLM not available'
            }
        
        try:
            temp = temperature if temperature is not None else LLM_TEMPERATURE
            max_toks = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
            
            # Build messages
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            # Try different Ollama API approaches
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Try chat API first (preferred)
            try:
                if hasattr(ollama, 'chat'):
                    response = ollama.chat(
                        model=self.model_name,
                        messages=messages if messages else [{'role': 'user', 'content': prompt}],
                        options={
                            'temperature': temp,
                            'num_predict': max_toks
                        }
                    )
                    text = response.get('message', {}).get('content', '')
                else:
                    raise AttributeError("chat method not available")
            except:
                # Fallback to generate API
                try:
                    response = ollama.generate(
                        model=self.model_name,
                        prompt=full_prompt,
                        options={
                            'temperature': temp,
                            'num_predict': max_toks
                        }
                    )
                    text = response.get('response', response.get('text', ''))
                except Exception as gen_error:
                    raise Exception(f"Both chat and generate APIs failed: {gen_error}")
            
            return {
                'text': text.strip(),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error generating LLM response: {error_msg}")
            
            # Check if it's a connection error
            if 'connection' in error_msg.lower() or 'refused' in error_msg.lower():
                error_msg = "Ollama service not running. Start with: ollama serve"
            
            return {
                'text': '',
                'success': False,
                'error': error_msg
            }
    
    def check_availability(self) -> bool:
        """Check if LLM is available and working."""
        if not self.available:
            return False
        
        try:
            # Try a simple test generation
            result = self.generate("Say 'OK' if you can hear me.", max_tokens=10)
            return result['success']
        except:
            return False


# Global instance
_llm_model = None

def get_llm_model(model_name: Optional[str] = None, base_url: Optional[str] = None) -> LLMModel:
    """Get or create global LLM model instance."""
    global _llm_model
    if _llm_model is None:
        _llm_model = LLMModel(model_name=model_name, base_url=base_url)
    return _llm_model

