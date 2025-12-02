"""
Minimal PlantNet-300K Test
===========================

Goals for tonight:
1. Load pre-trained PlantNet model
2. Run inference on your plant photo
3. Identify any blockers before full system build

This script is bare-bones - just enough to validate the approach.
"""

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json
import os
from pathlib import Path
import urllib.request

# ============================================================================
# STEP 1: Utilities & Helpers
# ============================================================================

def ensure_weights_exist():
    """
    Check if PlantNet weights exist. Looks for ResNet152 first, then ResNet18.
    Returns tuple: (weights_path, model_type) or (None, None) if not found.
    """
    # Try ResNet152 first (better accuracy)
    resnet152_path = "resnet152_weights_best_acc.tar"
    if os.path.exists(resnet152_path):
        print(f"✓ Found ResNet152 weights at {resnet152_path}")
        return resnet152_path, 'resnet152'
    
    # Fall back to ResNet18
    resnet18_path = "resnet18_weights_best_acc.tar"
    if os.path.exists(resnet18_path):
        print(f"✓ Found ResNet18 weights at {resnet18_path}")
        return resnet18_path, 'resnet18'
    
    # Neither found
    print(f"✗ No weights found")
    print("\nTo get weights:")
    print("1. Visit: https://github.com/plantnet/PlantNet-300K")
    print("2. Look for 'pre-trained models' link in README")
    print("3. Download either:")
    print("   - resnet152_weights_best_acc.tar (recommended, better accuracy)")
    print("   - resnet18_weights_best_acc.tar (smaller, faster)")
    print("4. Place in same directory as this script")
    return None, None

def ensure_metadata_exists():
    """
    Check if metadata files exist. If not, suggest where to get them.
    Returns dict with paths or None if not found.
    """
    metadata_files = {
        'species_names': 'plantnet300K_species_id_2_name.json',
        'class_mapping': 'class_idx_to_species_id.json'
    }
    
    all_exist = True
    for key, filename in metadata_files.items():
        if not os.path.exists(filename):
            print(f"✗ Missing {filename}")
            all_exist = False
        else:
            print(f"✓ Found {filename}")
    
    if not all_exist:
        print("\nTo get metadata files:")
        print("1. Visit: https://zenodo.org/record/4726653")
        print("2. Download the metadata JSON files")
        print("3. Place in same directory as this script")
        return None
    
    return metadata_files

# ============================================================================
# STEP 2: Model Loading
# ============================================================================

def load_plantnet_model(weights_path, device='cpu', model_type='resnet152'):
    """
    Load pre-trained PlantNet model (ResNet18 or ResNet152).
    
    Args:
        weights_path: Path to weights tar file
        device: 'cpu' or 'cuda' (use 'cuda' if GPU available)
        model_type: 'resnet18' or 'resnet152' (default: resnet152 for better accuracy)
    
    Returns:
        model: Loaded PyTorch model in eval mode
    """
    print(f"\n[Loading Model on {device}]")
    print(f"[Model Type: {model_type}]")
    
    # Create model with 1081 classes (PlantNet has 1081 plant species)
    if model_type == 'resnet152':
        model = models.resnet152(weights=None)
        model.fc = torch.nn.Linear(2048, 1081)
    else:  # resnet18
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(512, 1081)
    
    # Load weights
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        
        # The checkpoint might have different keys depending on how it was saved
        # Try different key patterns (ORDER MATTERS - check most specific first)
        if 'model' in checkpoint:
            # PlantNet format: checkpoint has 'model', 'epoch', 'optimizer'
            model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Try loading directly (might be just state dict)
            model.load_state_dict(checkpoint)
        
        print(f"✓ Loaded weights from {weights_path}")
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        print("\nDEBUGGING INFO:")
        print(f"  Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
        return None
    
    model.to(device)
    model.eval()
    
    return model

# ============================================================================
# STEP 3: Image Preprocessing
# ============================================================================

def preprocess_image(image_path, size=224):
    """
    Load and preprocess image for PlantNet model.
    
    Args:
        image_path: Path to image file
        size: Target image size (224x224 is standard)
    
    Returns:
        Preprocessed image tensor (1, 3, 224, 224)
    """
    print(f"\n[Processing Image: {image_path}]")
    
    # Open image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✓ Loaded image: {image.size}")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return None
    
    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    print(f"✓ Preprocessed to shape: {image_tensor.shape}")
    
    return image_tensor

# ============================================================================
# STEP 4: Species Name Mapping
# ============================================================================

def load_species_mapping(metadata_files):
    """
    Load mapping from class index to species name.
    
    Returns:
        Dict: {class_index: species_name}
    """
    print(f"\n[Loading Species Names]")
    
    try:
        with open(metadata_files['species_names']) as f:
            species_id_to_name = json.load(f)
        
        with open(metadata_files['class_mapping']) as f:
            class_idx_to_species_id = json.load(f)
        
        print(f"✓ Loaded {len(species_id_to_name)} species names")
        
        # Create class_idx -> species_name mapping
        idx_to_name = {}
        for class_idx_str, species_id_str in class_idx_to_species_id.items():
            # Keys might be strings, convert if needed
            species_name = species_id_to_name.get(species_id_str, f"Unknown species {species_id_str}")
            idx_to_name[int(class_idx_str)] = species_name
        
        return idx_to_name
    
    except Exception as e:
        print(f"✗ Error loading species names: {e}")
        return None

# ============================================================================
# STEP 5: Inference
# ============================================================================

def identify_plant(model, image_tensor, idx_to_name, device='cpu', top_k=5):
    """
    Run inference on image and return top-K predictions.
    
    Args:
        model: Loaded PyTorch model
        image_tensor: Preprocessed image tensor
        idx_to_name: Mapping from class index to species name
        device: 'cpu' or 'cuda'
        top_k: Number of top predictions to return
    
    Returns:
        List of (species_name, confidence) tuples
    """
    print(f"\n[Running Inference]")
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k, dim=1)
    
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        class_idx = idx.item()
        species_name = idx_to_name.get(class_idx, f"Unknown class {class_idx}")
        confidence = prob.item()
        predictions.append((species_name, confidence))
        print(f"  {len(predictions)}. {species_name}: {confidence:.2%}")
    
    return predictions

# ============================================================================
# STEP 6: Full Pipeline
# ============================================================================

def test_plantnet(image_path):
    """
    Full pipeline: check setup → load model → identify plant
    
    Args:
        image_path: Path to plant photo
    
    Returns:
        Predictions or None if failed
    """
    print("=" * 70)
    print("PlantNet-300K Minimal Test")
    print("=" * 70)
    
    # Step 1: Check weights (auto-detects ResNet152 or ResNet18)
    weights_path, model_type = ensure_weights_exist()
    if weights_path is None:
        print("\n⚠ BLOCKER: Weights file missing. Cannot continue.")
        print("ACTION: Download weights from GitHub and place in current directory.")
        return None
    
    # Step 2: Check metadata
    metadata_files = ensure_metadata_exists()
    if metadata_files is None:
        print("\n⚠ BLOCKER: Metadata files missing. Cannot continue.")
        print("ACTION: Download metadata from Zenodo and place in current directory.")
        return None
    
    # Step 3: Check image
    if not os.path.exists(image_path):
        print(f"\n✗ Image not found: {image_path}")
        return None
    
    # Step 4: Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Step 5: Load model (automatically detected: ResNet152 or ResNet18)
    model = load_plantnet_model(weights_path, device=device, model_type=model_type)
    if model is None:
        print("\n⚠ BLOCKER: Could not load model weights.")
        print("ACTION: Check that weights file is valid and in correct format.")
        return None
    
    # Step 6: Preprocess image
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        print("\n⚠ BLOCKER: Could not load/process image.")
        return None
    
    # Step 7: Load species mapping
    idx_to_name = load_species_mapping(metadata_files)
    if idx_to_name is None:
        print("\n⚠ BLOCKER: Could not load species names.")
        return None
    
    # Step 8: Run inference
    predictions = identify_plant(model, image_tensor, idx_to_name, device=device, top_k=5)
    
    print("\n" + "=" * 70)
    print("✓ SUCCESS: PlantNet model is working!")
    print("=" * 70)
    
    return predictions

# ============================================================================
# STEP 7: Main
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for any image files in current directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path('.').glob(f'*{ext}'))
        
        if image_files:
            image_path = str(image_files[0])
            print(f"Found image: {image_path}\n")
        else:
            print("Usage: python plantnet_minimal_test.py <image_path>")
            print("\nOR: Place a plant photo in the current directory and run script.")
            print("     Supported formats: .jpg, .jpeg, .png")
            sys.exit(1)
    
    # Run test
    predictions = test_plantnet(image_path)
    
    if predictions:
        print("\nYour plant is likely:")
        for i, (name, conf) in enumerate(predictions[:3], 1):
            print(f"  {i}. {name} ({conf:.1%} confidence)")