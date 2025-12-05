"""
Disease Classifier Model Wrapper

Loads and uses the trained disease classifier model from PlantVillage dataset.
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
DISEASE_MODEL_PATH = BASE_DIR / "models" / "efficientnet_b4_disease_classifier.tar"
DISEASE_MODEL_DEV_PATH = BASE_DIR / "models" / "efficientnet_b4_disease_classifier_dev.tar"
DISEASE_MAPPING_PATH = BASE_DIR / "backend" / "models" / "disease_mapping.json"


class DiseaseClassifier:
    """Wrapper for the disease classifier model."""
    
    def __init__(self, use_dev_model=False):
        """
        Initialize disease classifier.
        
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
        self.disease_mapping = {}  # Maps plant-specific diseases to general diseases
        self.reverse_mapping = {}  # Maps general diseases to plant-specific diseases
        
        self.load_model()
        self.load_disease_mapping()
    
    def load_model(self):
        """Load disease classifier model."""
        model_path = DISEASE_MODEL_DEV_PATH if self.use_dev_model else DISEASE_MODEL_PATH
        
        if not model_path.exists():
            print(f"⚠️  Disease classifier model not found at {model_path}")
            if self.use_dev_model:
                print(f"   Trying production model at {DISEASE_MODEL_PATH}...")
                model_path = DISEASE_MODEL_PATH
                if not model_path.exists():
                    print(f"   ❌ Neither dev nor production model found")
                    self.available = False
                    return
            else:
                # Try dev model as fallback
                print(f"   Trying dev model at {DISEASE_MODEL_DEV_PATH}...")
                if DISEASE_MODEL_DEV_PATH.exists():
                    model_path = DISEASE_MODEL_DEV_PATH
                    print(f"   ✅ Using dev model as fallback")
                else:
                    print(f"   ❌ Production model not found and dev model also missing")
                    self.available = False
                    return
        
        try:
            print(f"🔄 Loading disease classifier from {model_path}...")
            
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
            
            # Try to load state dict
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"   ⚠️  Missing keys (expected for classifier layer): {len(missing_keys)}")
            if unexpected_keys:
                print(f"   ⚠️  Unexpected keys: {len(unexpected_keys)}")
            
            self.model.to(self.device)
            self.model.eval()
            
            self.available = True
            print(f"✅ Disease classifier loaded: {self.num_classes} classes")
            
        except Exception as e:
            print(f"❌ Error loading disease classifier: {e}")
            import traceback
            traceback.print_exc()
            self.available = False
            print(f"   Model path attempted: {model_path}")
            print(f"   Model exists: {model_path.exists() if model_path else 'N/A'}")
    
    def load_disease_mapping(self):
        """Load disease mapping from plant-specific to general disease types."""
        try:
            if DISEASE_MAPPING_PATH.exists():
                with open(DISEASE_MAPPING_PATH, 'r') as f:
                    mapping_data = json.load(f)
                    self.disease_mapping = mapping_data.get('mapping', {})
                    self.reverse_mapping = mapping_data.get('reverse_mapping', {})
                print(f"✅ Loaded disease mapping: {len(self.disease_mapping)} plant-specific → {len(self.reverse_mapping)} general diseases")
            else:
                print(f"⚠️  Disease mapping not found at {DISEASE_MAPPING_PATH}")
                print("   Will use plant-specific disease names")
                self.disease_mapping = {}
                self.reverse_mapping = {}
        except Exception as e:
            print(f"❌ Error loading disease mapping: {e}")
            self.disease_mapping = {}
            self.reverse_mapping = {}
    
    def map_to_general_disease(self, plant_specific_disease: str) -> str:
        """
        Map plant-specific disease (e.g., "Tomato___Early_blight") to general disease (e.g., "Early Blight").
        
        Args:
            plant_specific_disease: Disease class name in format "Plant___Disease"
            
        Returns:
            General disease name, or original if no mapping found
        """
        if not self.disease_mapping:
            # No mapping loaded, extract disease part manually
            if '___' in plant_specific_disease:
                disease_part = plant_specific_disease.split('___', 1)[1]
                # Convert underscores to spaces and title case
                return disease_part.replace('_', ' ').title()
            return plant_specific_disease
        
        # Use mapping if available
        general_disease = self.disease_mapping.get(plant_specific_disease, None)
        if general_disease:
            return general_disease
        
        # Fallback: extract disease part if no mapping
        if '___' in plant_specific_disease:
            disease_part = plant_specific_disease.split('___', 1)[1]
            return disease_part.replace('_', ' ').title()
        
        return plant_specific_disease
    
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
        Predict plant disease.
        
        Args:
            image_pil: PIL Image object
            top_k: Return top K predictions
            
        Returns:
            dict with predictions and metadata
        """
        if not self.available:
            return {
                'available': False,
                'error': f'Disease classifier model not available. Model path checked: {DISEASE_MODEL_PATH} (exists: {DISEASE_MODEL_PATH.exists()}), {DISEASE_MODEL_DEV_PATH} (exists: {DISEASE_MODEL_DEV_PATH.exists()})',
                'primary': {},
                'top_k': []
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
            
            # Build results - map to general diseases (NO PLANT NAMES)
            top_k_results = []
            for prob, idx in zip(top_probs, top_indices):
                class_name = self.idx_to_class.get(idx.item(), f"Class_{idx.item()}")
                
                # Map plant-specific disease to general disease
                general_disease = self.map_to_general_disease(class_name)
                
                top_k_results.append({
                    'disease': general_disease,  # General disease name only (no plant)
                    'condition': general_disease,  # Same as disease (no plant)
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
            print(f"❌ Error in disease classifier prediction: {e}")
            import traceback
            traceback.print_exc()
            return {
                'available': False,
                'error': str(e)
            }


# Singleton instances
_disease_classifier = None
_disease_classifier_dev = None


def get_disease_classifier(use_dev=False) -> DiseaseClassifier:
    """Get singleton instance of disease classifier."""
    global _disease_classifier, _disease_classifier_dev
    
    if use_dev:
        if _disease_classifier_dev is None:
            _disease_classifier_dev = DiseaseClassifier(use_dev_model=True)
        return _disease_classifier_dev
    else:
        if _disease_classifier is None:
            _disease_classifier = DiseaseClassifier(use_dev_model=False)
        return _disease_classifier

