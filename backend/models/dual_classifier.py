"""
Dual classifier that uses both houseplant fine-tuned model and general PlantNet model.
Selects the prediction with the highest confidence from either model.
"""

from typing import Dict, Optional
from PIL import Image
from models.plantnet_model import get_plantnet_model
from models.houseplant_model import get_houseplant_model


class DualClassifier:
    """Classifier that uses both houseplant and PlantNet models."""
    
    def __init__(self):
        """Initialize dual classifier."""
        self.plantnet_model = get_plantnet_model()
        self.houseplant_model = get_houseplant_model()
    
    def predict(self, image_pil: Image.Image, top_k: int = 5) -> Dict:
        """
        Predict plant species using both models and select best result.
        
        Args:
            image_pil: PIL Image object
            top_k: Return top K predictions
            
        Returns:
            dict with best predictions and metadata about which model was used
        """
        # Run both models
        plantnet_results = self.plantnet_model.predict(image_pil, top_k=top_k)
        houseplant_results = self.houseplant_model.predict(image_pil, top_k=top_k)
        
        # Get primary predictions
        plantnet_primary = plantnet_results.get("primary")
        houseplant_primary = houseplant_results.get("primary")
        
        # Determine which model to use based on highest confidence
        plantnet_conf = plantnet_primary.get("confidence", 0.0) if plantnet_primary else 0.0
        houseplant_available = houseplant_results.get("available", False)
        houseplant_conf = houseplant_primary.get("confidence", 0.0) if (houseplant_primary and houseplant_available) else 0.0
        
        # Select best model (highest confidence)
        # Only use houseplant if it's available and has higher confidence
        if houseplant_available and houseplant_conf > plantnet_conf:
            selected_results = houseplant_results
            selected_model = "houseplant"
            confidence_diff = houseplant_conf - plantnet_conf
        else:
            # Use PlantNet (always available as fallback)
            selected_results = plantnet_results
            selected_model = "plantnet"
            confidence_diff = plantnet_conf - houseplant_conf
        
        # Format response
        primary = selected_results.get("primary")
        top_k_results = selected_results.get("top_k", [])
        
        # Add metadata about dual classifier decision
        result = {
            "primary": primary,
            "top_k": top_k_results,
            "all_results": selected_results.get("all_results", []),
            "dual_classifier": {
                "selected_model": selected_model,
                "houseplant_confidence": houseplant_conf,
                "plantnet_confidence": plantnet_conf,
                "confidence_difference": confidence_diff,
                "houseplant_available": houseplant_results.get("available", False)
            }
        }
        
        return result


# Global instance
_dual_classifier = None

def get_dual_classifier() -> DualClassifier:
    """Get or create global dual classifier instance."""
    global _dual_classifier
    if _dual_classifier is None:
        _dual_classifier = DualClassifier()
    return _dual_classifier

