"""
Houseplant model wrapper for fine-tuned EfficientNet B4 on houseplant species.
"""

import torch
import torchvision.models as models
import torch.nn as nn
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional
from config import HOUSEPLANT_MODEL_PATH, DEVICE

BASE_DIR = Path(__file__).parent.parent.parent


class HouseplantModel:
    """Wrapper for fine-tuned houseplant species model."""
    
    def __init__(self):
        """Initialize houseplant model."""
        # Set device (supports MPS for Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.model = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.num_classes = 0
        self.available = False
        
        self.load_model()
    
    def load_model(self):
        """Load fine-tuned houseplant model."""
        if not HOUSEPLANT_MODEL_PATH.exists():
            print(f"⚠️  Houseplant model not found at {HOUSEPLANT_MODEL_PATH}")
            print("   Dual-classifier will use PlantNet model only")
            self.available = False
            return
        
        try:
            print(f"🔄 Loading houseplant model from {HOUSEPLANT_MODEL_PATH}...")
            
            # Load checkpoint
            checkpoint = torch.load(HOUSEPLANT_MODEL_PATH, map_location=self.device)
            
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
            model = models.efficientnet_b4(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.num_classes)
            
            # Load weights
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            
            self.model = model
            self.available = True
            
            print(f"✅ Houseplant model loaded: {self.num_classes} classes")
            
        except Exception as e:
            print(f"❌ Error loading houseplant model: {e}")
            import traceback
            traceback.print_exc()
            self.available = False
    
    def preprocess_image(self, image_pil: Image.Image) -> torch.Tensor:
        """Preprocess PIL image for model input."""
        from torchvision import transforms
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        img_tensor = preprocess(image_pil)
        return img_tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image_pil: Image.Image, top_k: int = 5) -> Dict:
        """
        Predict houseplant species.
        
        Args:
            image_pil: PIL Image object
            top_k: Return top K predictions
            
        Returns:
            dict with predictions similar to PlantNetModel format
        """
        if not self.available or self.model is None:
            return {
                "primary": None,
                "top_k": [],
                "all_results": [],
                "model_source": "houseplant",
                "available": False
            }
        
        try:
            # Preprocess
            img_tensor = self.preprocess_image(image_pil)
            
            # Inference
            with torch.no_grad():
                logits = self.model(img_tensor)
                probabilities = torch.softmax(logits, dim=1)
            
            # Get top k
            top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes), dim=1)
            
            # Format results
            results = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                class_idx = idx.item()
                class_name = self.idx_to_class.get(class_idx, f"Class_{class_idx}")
                
                results.append({
                    "species_id": None,  # Houseplant model doesn't have species IDs
                    "species_name": class_name,
                    "common_name": class_name,  # Class names are already common names
                    "confidence": float(prob.item())
                })
            
            return {
                "primary": results[0] if results else None,
                "top_k": results,
                "all_results": results,
                "model_source": "houseplant",
                "available": True
            }
            
        except Exception as e:
            print(f"❌ Error in houseplant prediction: {e}")
            import traceback
            traceback.print_exc()
            return {
                "primary": None,
                "top_k": [],
                "all_results": [],
                "model_source": "houseplant",
                "available": False
            }


# Global instance
_houseplant_model = None

def get_houseplant_model() -> HouseplantModel:
    """Get or create global houseplant model instance."""
    global _houseplant_model
    if _houseplant_model is None:
        _houseplant_model = HouseplantModel()
    return _houseplant_model

