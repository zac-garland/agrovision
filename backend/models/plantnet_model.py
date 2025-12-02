import torch
import torchvision.models as models
from pathlib import Path
import json
from config import (
    PLANTNET_MODEL_PATH, 
    PLANTNET_18_MODEL_PATH,
    COMMON_NAMES_PATH,
    DEVICE
)

# Metadata paths (relative to project root)
BASE_DIR = Path(__file__).parent.parent.parent
META_DIR = BASE_DIR / "meta"
CLASS_IDX_TO_SPECIES_ID = META_DIR / "class_idx_to_species_id.json"
SPECIES_ID_TO_NAME = META_DIR / "plantnet300K_species_id_2_name.json"


class PlantNetModel:
    """Wrapper for PlantNet-300K model inference."""
    
    def __init__(self, use_resnet18=False):
        """
        Initialize PlantNet model.
        
        Args:
            use_resnet18: If True, use ResNet18; else use ResNet152 (more accurate)
        """
        self.use_resnet18 = use_resnet18
        self.device = torch.device(DEVICE)
        self.model = None
        self.class_idx_to_species_id = None
        self.species_id_to_name = None
        self.common_names_map = None
        
        self.load_model()
        self.load_species_mapping()
        self.load_common_names()
    
    def load_model(self):
        """Load pretrained PlantNet model."""
        try:
            if self.use_resnet18:
                print("🔄 Loading PlantNet ResNet18...")
                checkpoint_path = PLANTNET_18_MODEL_PATH
                self.model = models.resnet18(weights=None)
                num_classes = 1081  # PlantNet-300K has 1081 species
            else:
                print("🔄 Loading PlantNet ResNet152...")
                checkpoint_path = PLANTNET_MODEL_PATH
                self.model = models.resnet152(weights=None)
                num_classes = 1081
            
            # Modify final layer for number of classes
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
            
            # Load checkpoint
            if Path(checkpoint_path).exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        self.model.load_state_dict(checkpoint['model'])
                    elif 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Try loading directly
                        try:
                            self.model.load_state_dict(checkpoint)
                        except:
                            print(f"⚠️  Warning: Could not load checkpoint directly. Keys: {list(checkpoint.keys())[:5]}")
                else:
                    self.model.load_state_dict(checkpoint)
                
                print(f"✅ Model loaded from {checkpoint_path}")
            else:
                print(f"⚠️  Model file not found at {checkpoint_path}")
                print("   Using random initialization (will get poor results)")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_species_mapping(self):
        """Load species ID to name mapping."""
        try:
            # Load class index to species ID mapping
            if CLASS_IDX_TO_SPECIES_ID.exists():
                with open(CLASS_IDX_TO_SPECIES_ID, 'r') as f:
                    self.class_idx_to_species_id = json.load(f)
                print(f"✅ Loaded class index to species ID mapping ({len(self.class_idx_to_species_id)} classes)")
            else:
                print(f"⚠️  Class index mapping not found at {CLASS_IDX_TO_SPECIES_ID}")
                self.class_idx_to_species_id = {}
            
            # Load species ID to name mapping
            if SPECIES_ID_TO_NAME.exists():
                with open(SPECIES_ID_TO_NAME, 'r') as f:
                    self.species_id_to_name = json.load(f)
                print(f"✅ Loaded species ID to name mapping ({len(self.species_id_to_name)} species)")
            else:
                print(f"⚠️  Species ID to name mapping not found at {SPECIES_ID_TO_NAME}")
                self.species_id_to_name = {}
                
        except Exception as e:
            print(f"❌ Error loading species mapping: {e}")
            self.class_idx_to_species_id = {}
            self.species_id_to_name = {}
    
    def load_common_names(self):
        """Load scientific name to common name mapping."""
        try:
            if Path(COMMON_NAMES_PATH).exists():
                with open(COMMON_NAMES_PATH, 'r') as f:
                    self.common_names_map = json.load(f)
                print(f"✅ Loaded {len(self.common_names_map)} common names")
            else:
                print(f"⚠️  Common names file not found at {COMMON_NAMES_PATH}")
                self.common_names_map = {}
        except Exception as e:
            print(f"❌ Error loading common names: {e}")
            self.common_names_map = {}
    
    def preprocess_image(self, image_pil):
        """
        Preprocess PIL image for model input.
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            torch.Tensor of shape (1, 3, 224, 224)
        """
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
    
    def predict(self, image_pil, top_k=5):
        """
        Get species predictions for image.
        
        Args:
            image_pil: PIL Image object
            top_k: Return top K predictions
            
        Returns:
            dict with top_k predictions
        """
        try:
            # Preprocess
            img_tensor = self.preprocess_image(image_pil)
            
            # Inference
            with torch.no_grad():
                logits = self.model(img_tensor)
                probabilities = torch.softmax(logits, dim=1)
            
            # Get top k
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
            
            # Format results
            results = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                class_idx = idx.item()
                
                # Get species ID from class index
                species_id = self.class_idx_to_species_id.get(str(class_idx), None)
                
                # Get scientific name from species ID
                if species_id and self.species_id_to_name:
                    species_name = self.species_id_to_name.get(str(species_id), f"Unknown species {species_id}")
                else:
                    species_name = f"Class_{class_idx}"
                
                # Get common name from scientific name
                common_name = self.common_names_map.get(species_name, species_name)
                
                results.append({
                    "species_id": int(species_id) if species_id else None,
                    "species_name": species_name,
                    "common_name": common_name,
                    "confidence": float(prob.item())
                })
            
            return {
                "primary": results[0] if results else None,
                "top_k": results,
                "all_results": results
            }
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            raise

# Global model instance
_plantnet_model = None

def get_plantnet_model(use_resnet18=False):
    """Get or create global PlantNet model instance."""
    global _plantnet_model
    if _plantnet_model is None:
        _plantnet_model = PlantNetModel(use_resnet18=use_resnet18)
    return _plantnet_model
