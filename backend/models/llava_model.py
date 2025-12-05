"""
LLaVA model wrapper for vision-language plant analysis.
Uses Ollama's LLaVA model to identify plants and detect lesions through natural language.
"""

import os
from typing import Dict, Optional, List
from PIL import Image
import io
import base64

# Try to import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  ollama package not installed. LLaVA analysis will be unavailable.")
    print("   Install with: pip install ollama")


class LLaVAModel:
    """Wrapper for LLaVA vision-language model via Ollama."""
    
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize LLaVA model.
        
        Args:
            model_name: Name of the Ollama LLaVA model (default: 'llava' or 'llava:latest')
            base_url: Ollama API base URL (default: http://localhost:11434)
        """
        self.available = False
        self.model_name = model_name or os.getenv("LLAVA_MODEL_NAME", "llava:latest")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        if not OLLAMA_AVAILABLE:
            print("⚠️  Ollama package not available. Install with: pip install ollama")
            return
        
        # Check if model is available
        self.available = self._check_model_available()
        if self.available:
            print(f"✅ LLaVA model initialized (model: {self.model_name})")
        else:
            print(f"⚠️  LLaVA model '{self.model_name}' not available. Pull with: ollama pull {self.model_name}")
    
    def _check_model_available(self) -> bool:
        """Check if LLaVA model is available in Ollama."""
        try:
            models = ollama.list()
            if hasattr(models, 'models'):
                model_names = [m.model for m in models.models]
                # Check for exact match or variants
                return any(
                    self.model_name in name or name in self.model_name 
                    for name in model_names
                )
            return False
        except:
            return False
    
    def analyze_plant_image(
        self, 
        image_pil: Image.Image,
        include_lesion_analysis: bool = True
    ) -> Dict:
        """
        Analyze plant image for identification and disease detection.
        
        Args:
            image_pil: PIL Image of the plant
            include_lesion_analysis: Whether to include lesion detection in prompt
            
        Returns:
            Dictionary with:
            - 'plant_identification': Dict with species, confidence, reasoning
            - 'lesion_analysis': Dict with lesion detection, severity, locations
            - 'raw_response': Full LLaVA response text
            - 'success': Boolean
            - 'error': Error message if failed
        """
        if not self.available:
            return {
                'plant_identification': None,
                'lesion_analysis': None,
                'raw_response': '',
                'success': False,
                'error': 'LLaVA model not available'
            }
        
        try:
            # Build comprehensive prompt
            prompt = self._build_analysis_prompt(include_lesion_analysis)
            
            # Convert PIL image to base64 for Ollama
            image_base64 = self._image_to_base64(image_pil)
            
            # Call LLaVA via Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_base64]
                    }
                ],
                options={
                    'temperature': 0.3,  # Lower temperature for more consistent analysis
                    'num_predict': 512
                }
            )
            
            text = response.get('message', {}).get('content', '')
            
            # Parse response
            parsed = self._parse_response(text)
            
            return {
                'plant_identification': parsed.get('plant_identification'),
                'lesion_analysis': parsed.get('lesion_analysis'),
                'raw_response': text,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error in LLaVA analysis: {error_msg}")
            
            return {
                'plant_identification': None,
                'lesion_analysis': None,
                'raw_response': '',
                'success': False,
                'error': error_msg
            }
    
    def _build_analysis_prompt(self, include_lesion_analysis: bool) -> str:
        """Build prompt for LLaVA analysis."""
        prompt = """Analyze this plant image and provide a detailed assessment in the following structured format:

PLANT IDENTIFICATION:
- Species name (scientific name if known, or common name)
- Common name (if different from species name)
- Confidence level (high/medium/low)
- Reasoning: Why you identified it as this species (describe key visual features)

"""
        
        if include_lesion_analysis:
            prompt += """DISEASE/LESION ANALYSIS:
- Are there any visible lesions, spots, discoloration, or damage on the leaves?
- If yes, describe the type (brown spots, yellowing, black spots, etc.)
- Estimate the percentage of affected leaf area (0-100%)
- Severity assessment (none/low/moderate/high)
- Specific locations: Describe where on the leaves the issues appear (edges, center, scattered, etc.)
- Reasoning: What might be causing these issues

"""
        
        prompt += """Please be specific and detailed in your analysis. Format your response clearly with the sections above."""
        
        return prompt
    
    def _image_to_base64(self, image_pil: Image.Image) -> str:
        """Convert PIL Image to base64 string for Ollama."""
        # Convert to RGB if needed
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # Save to bytes
        buffer = io.BytesIO()
        image_pil.save(buffer, format='JPEG', quality=85)
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return image_base64
    
    def _parse_response(self, text: str) -> Dict:
        """Parse LLaVA response text into structured format."""
        plant_id = {
            'species_name': 'Unknown',
            'common_name': 'Unknown',
            'confidence': 'medium',
            'reasoning': ''
        }
        
        lesion_analysis = {
            'has_lesions': False,
            'lesion_type': None,
            'affected_percentage': 0.0,
            'severity': 'none',
            'locations': '',
            'reasoning': ''
        }
        
        # Simple parsing - look for key sections
        lines = text.split('\n')
        current_section = None
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Detect sections
            if 'plant identification' in line_lower:
                current_section = 'plant'
            elif 'disease' in line_lower or 'lesion' in line_lower:
                current_section = 'lesion'
            
            # Parse plant identification
            if current_section == 'plant':
                if 'species' in line_lower and ':' in line:
                    species = line.split(':', 1)[1].strip()
                    if species and species != 'Unknown':
                        plant_id['species_name'] = species
                        plant_id['common_name'] = species  # Default to same
                elif 'common name' in line_lower and ':' in line:
                    common = line.split(':', 1)[1].strip()
                    if common:
                        plant_id['common_name'] = common
                elif 'confidence' in line_lower and ':' in line:
                    conf = line.split(':', 1)[1].strip().lower()
                    if conf in ['high', 'medium', 'low']:
                        plant_id['confidence'] = conf
                elif 'reasoning' in line_lower or line.startswith('-'):
                    if 'reasoning' in line_lower:
                        # Get reasoning from next lines
                        reasoning_lines = []
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if lines[j].strip() and not lines[j].strip().startswith('DISEASE'):
                                reasoning_lines.append(lines[j].strip())
                            else:
                                break
                        plant_id['reasoning'] = ' '.join(reasoning_lines)
            
            # Parse lesion analysis
            if current_section == 'lesion':
                if 'yes' in line_lower or 'visible' in line_lower or 'lesion' in line_lower:
                    if any(word in line_lower for word in ['yes', 'visible', 'spot', 'damage', 'discoloration']):
                        lesion_analysis['has_lesions'] = True
                elif 'type' in line_lower and ':' in line:
                    lesion_type = line.split(':', 1)[1].strip()
                    if lesion_type:
                        lesion_analysis['lesion_type'] = lesion_type
                elif 'percentage' in line_lower or '%' in line:
                    # Try to extract percentage
                    import re
                    percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', line)
                    if percentages:
                        lesion_analysis['affected_percentage'] = float(percentages[0])
                elif 'severity' in line_lower and ':' in line:
                    severity = line.split(':', 1)[1].strip().lower()
                    if severity in ['none', 'low', 'moderate', 'high']:
                        lesion_analysis['severity'] = severity
                elif 'location' in line_lower and ':' in line:
                    locations = line.split(':', 1)[1].strip()
                    if locations:
                        lesion_analysis['locations'] = locations
        
        # Convert confidence to numeric
        conf_map = {'high': 0.8, 'medium': 0.6, 'low': 0.4}
        plant_id['confidence_numeric'] = conf_map.get(plant_id['confidence'], 0.6)
        
        return {
            'plant_identification': plant_id,
            'lesion_analysis': lesion_analysis
        }


# Global instance
_llava_model = None

def get_llava_model(model_name: Optional[str] = None, base_url: Optional[str] = None) -> LLaVAModel:
    """Get or create global LLaVA model instance."""
    global _llava_model
    if _llava_model is None:
        _llava_model = LLaVAModel(model_name=model_name, base_url=base_url)
    return _llava_model

