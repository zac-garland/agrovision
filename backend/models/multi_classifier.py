"""
Multi-classifier that uses houseplant model, PlantNet model, and LLaVA vision model.
Collects results from all models for comparison and ensemble analysis.
"""

from typing import Dict, Optional, List
from PIL import Image
from models.plantnet_model import get_plantnet_model
from models.houseplant_model import get_houseplant_model
from models.llava_model import get_llava_model


class MultiClassifier:
    """Classifier that uses houseplant, PlantNet, and LLaVA models."""
    
    def __init__(self):
        """Initialize multi-classifier."""
        self.plantnet_model = get_plantnet_model()
        self.houseplant_model = get_houseplant_model()
        self.llava_model = get_llava_model()
    
    def predict(self, image_pil: Image.Image, top_k: int = 5, include_llava: bool = True) -> Dict:
        """
        Predict plant species using all available models.
        
        Args:
            image_pil: PIL Image object
            top_k: Return top K predictions for traditional models
            include_llava: Whether to include LLaVA analysis (slower but more comprehensive)
            
        Returns:
            dict with results from all models and ensemble analysis
        """
        results = {
            'plantnet': None,
            'houseplant': None,
            'llava': None,
            'ensemble': None,  # Best consensus result
            'all_models': {}
        }
        
        # Run PlantNet model
        try:
            print("   Running PlantNet model...")
            plantnet_results = self.plantnet_model.predict(image_pil, top_k=top_k)
            results['plantnet'] = plantnet_results
            results['all_models']['plantnet'] = {
                'primary': plantnet_results.get("primary"),
                'top_k': plantnet_results.get("top_k", []),
                'available': True
            }
        except Exception as e:
            print(f"   ⚠️  PlantNet error: {e}")
            results['all_models']['plantnet'] = {'available': False, 'error': str(e)}
        
        # Run houseplant model
        try:
            print("   Running houseplant model...")
            houseplant_results = self.houseplant_model.predict(image_pil, top_k=top_k)
            results['houseplant'] = houseplant_results
            results['all_models']['houseplant'] = {
                'primary': houseplant_results.get("primary"),
                'top_k': houseplant_results.get("top_k", []),
                'available': houseplant_results.get("available", False)
            }
        except Exception as e:
            print(f"   ⚠️  Houseplant model error: {e}")
            results['all_models']['houseplant'] = {'available': False, 'error': str(e)}
        
        # Run LLaVA model (if requested and available)
        if include_llava and self.llava_model.available:
            try:
                print("   Running LLaVA vision model...")
                llava_results = self.llava_model.analyze_plant_image(image_pil, include_lesion_analysis=True)
                results['llava'] = llava_results
                results['all_models']['llava'] = {
                    'plant_identification': llava_results.get('plant_identification'),
                    'lesion_analysis': llava_results.get('lesion_analysis'),
                    'raw_response': llava_results.get('raw_response'),
                    'available': llava_results.get('success', False)
                }
            except Exception as e:
                print(f"   ⚠️  LLaVA error: {e}")
                results['all_models']['llava'] = {'available': False, 'error': str(e)}
        else:
            results['all_models']['llava'] = {'available': False}
        
        # Determine ensemble result (best consensus)
        results['ensemble'] = self._determine_ensemble_result(results)
        
        # Add metadata
        results['metadata'] = {
            'models_run': [k for k, v in results['all_models'].items() if v.get('available', False)],
            'llava_available': self.llava_model.available if include_llava else False
        }
        
        return results
    
    def _determine_ensemble_result(self, results: Dict) -> Dict:
        """
        Determine best consensus result from all models.
        Uses highest confidence from traditional models, with LLaVA as validation.
        """
        plantnet_primary = results.get('plantnet', {}).get('primary')
        houseplant_primary = results.get('houseplant', {}).get('primary')
        houseplant_available = results.get('houseplant', {}).get('available', False)
        llava_plant = results.get('llava', {}).get('plant_identification')
        
        # Get confidences
        plantnet_conf = plantnet_primary.get("confidence", 0.0) if plantnet_primary else 0.0
        houseplant_conf = houseplant_primary.get("confidence", 0.0) if (houseplant_primary and houseplant_available) else 0.0
        
        # Select best traditional model (highest confidence)
        if houseplant_available and houseplant_conf > plantnet_conf:
            selected = houseplant_primary
            selected_model = "houseplant"
        else:
            selected = plantnet_primary
            selected_model = "plantnet"
        
        # Check if LLaVA agrees (optional validation)
        consensus_score = 1.0
        if llava_plant and selected:
            llava_species = llava_plant.get('species_name', '').lower()
            selected_species = selected.get('species_name', '').lower()
            
            # Simple name matching (could be improved)
            if llava_species and selected_species:
                # Check if names overlap
                if any(word in selected_species for word in llava_species.split() if len(word) > 3) or \
                   any(word in llava_species for word in selected_species.split() if len(word) > 3):
                    consensus_score = 1.2  # Boost confidence if LLaVA agrees
                else:
                    consensus_score = 0.9  # Slight penalty if disagreement
        
        return {
            'primary': selected,
            'selected_model': selected_model,
            'consensus_score': consensus_score,
            'llava_validation': llava_plant is not None
        }


# Global instance
_multi_classifier = None

def get_multi_classifier() -> MultiClassifier:
    """Get or create global multi-classifier instance."""
    global _multi_classifier
    if _multi_classifier is None:
        _multi_classifier = MultiClassifier()
    return _multi_classifier

