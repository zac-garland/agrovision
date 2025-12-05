"""
Species Classifier Model Wrapper

Loads and uses the trained species classifier model that combines:
- Houseplant species
- PlantNet species (with hierarchical naming)
- PlantVillage healthy classes
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json
from typing import Dict, List, Optional
from config import DEVICE

# Try to import EfficientNet
try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_PYTORCH_AVAILABLE = True
except ImportError:
    EFFICIENTNET_PYTORCH_AVAILABLE = False

# Model paths
BASE_DIR = Path(__file__).parent.parent.parent
SPECIES_MODEL_PATH = BASE_DIR / "models" / "efficientnet_b4_species_classifier.tar"
SPECIES_MODEL_DEV_PATH = BASE_DIR / "models" / "efficientnet_b4_species_classifier_dev.tar"


class SpeciesClassifier:
    """Wrapper for the unified species classifier model."""
    
    def __init__(self, use_dev_model=False):
        """
        Initialize species classifier.
        
        Args:
            use_dev_model: If True, use dev model (for testing)
        """
        self.device = torch.device(DEVICE)
        self.model = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.num_classes = 0
        self.available = False
        self.use_dev_model = use_dev_model
        
        self.load_model()
    
    def load_model(self):
        """Load species classifier model."""
        model_path = SPECIES_MODEL_DEV_PATH if self.use_dev_model else SPECIES_MODEL_PATH
        
        if not model_path.exists():
            print(f"⚠️  Species classifier model not found at {model_path}")
            if self.use_dev_model:
                print(f"   Trying production model at {SPECIES_MODEL_PATH}...")
                model_path = SPECIES_MODEL_PATH
                if not model_path.exists():
                    self.available = False
                    return
            else:
                self.available = False
                return
        
        try:
            print(f"🔄 Loading species classifier from {model_path}...")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get class mappings from checkpoint
            if isinstance(checkpoint, dict):
                self.class_to_idx = checkpoint.get('class_to_idx', {})
                self.idx_to_class = checkpoint.get('idx_to_class', {})
                self.num_classes = checkpoint.get('num_classes', len(self.class_to_idx))
            else:
                print("⚠️  Unexpected checkpoint format")
                self.available = False
                return
            
            # Create model architecture
            if EFFICIENTNET_PYTORCH_AVAILABLE:
                self.model = EfficientNet.from_name('efficientnet-b4', num_classes=self.num_classes)
            else:
                import torchvision.models as models
                self.model = models.efficientnet_b4(weights=None)
                in_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(in_features, self.num_classes)
            
            # Load weights
            if isinstance(checkpoint, dict):
                state_dict = (checkpoint.get('model') or 
                            checkpoint.get('model_state_dict') or 
                            checkpoint.get('state_dict') or
                            checkpoint)
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            self.available = True
            print(f"✅ Species classifier loaded: {self.num_classes} classes")
            
        except Exception as e:
            print(f"❌ Error loading species classifier: {e}")
            import traceback
            traceback.print_exc()
            self.available = False
    
    def preprocess_image(self, image_pil: Image.Image) -> torch.Tensor:
        """Preprocess image for model input."""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(image_pil).unsqueeze(0)
    
    def predict(self, image_pil: Image.Image, top_k: int = 5) -> Dict:
        """
        Predict plant species.
        
        Args:
            image_pil: PIL Image object
            top_k: Return top K predictions
            
        Returns:
            dict with predictions and metadata
        """
        if not self.available:
            return {
                'available': False,
                'error': 'Species classifier model not available'
            }
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_pil).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
            
            # Build results
            top_k_results = []
            for prob, idx in zip(top_probs, top_indices):
                class_name = self.idx_to_class.get(idx.item(), f"Class_{idx.item()}")
                top_k_results.append({
                    'species': class_name,
                    'confidence': prob.item()
                })
            
            primary = top_k_results[0] if top_k_results else None
            
            return {
                'available': True,
                'primary': primary,
                'top_k': top_k_results,
                'num_classes': self.num_classes
            }
            
        except Exception as e:
            print(f"❌ Error in species classifier prediction: {e}")
            import traceback
            traceback.print_exc()
            return {
                'available': False,
                'error': str(e)
            }


# Singleton instances
_species_classifier = None
_species_classifier_dev = None


def get_species_classifier(use_dev=False) -> SpeciesClassifier:
    """Get singleton instance of species classifier."""
    global _species_classifier, _species_classifier_dev
    
    if use_dev:
        if _species_classifier_dev is None:
            _species_classifier_dev = SpeciesClassifier(use_dev_model=True)
        return _species_classifier_dev
    else:
        if _species_classifier is None:
            _species_classifier = SpeciesClassifier(use_dev_model=False)
        return _species_classifier

