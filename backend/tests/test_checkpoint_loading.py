#!/usr/bin/env python3
"""Test script to verify checkpoint loading works correctly."""

import torch
import torchvision.models as models
from pathlib import Path

MODEL_PATH = Path("models/efficientnet_b4_weights_best_acc.tar")
DEVICE = torch.device("mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else "cpu")

print("="*60)
print("Testing Checkpoint Loading")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Checkpoint path: {MODEL_PATH}")
print()

# Check if file exists
if not MODEL_PATH.exists():
    print(f"❌ Checkpoint file not found: {MODEL_PATH}")
    exit(1)

print(f"✅ Checkpoint file exists: {MODEL_PATH.stat().st_size / (1024*1024):.1f} MB")
print()

# Load checkpoint
print("Loading checkpoint...")
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    print(f"✅ Checkpoint loaded successfully")
    print(f"   Type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"   Keys: {list(checkpoint.keys())}")
        
        # Find state dict
        state_dict = checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
        
        if state_dict:
            print(f"\n✅ Found state_dict with {len(state_dict)} keys")
            
            # Check first layer weights
            first_layer_key = None
            for key in sorted(state_dict.keys()):
                if 'features.0' in key and 'weight' in key:
                    first_layer_key = key
                    break
            
            if first_layer_key:
                weight = state_dict[first_layer_key]
                weight_sum = weight.sum().item()
                weight_mean = weight.mean().item()
                print(f"\nFirst layer weights ({first_layer_key}):")
                print(f"   Shape: {weight.shape}")
                print(f"   Sum: {weight_sum:.4f}")
                print(f"   Mean: {weight_mean:.6f}")
                print(f"   Min: {weight.min().item():.6f}")
                print(f"   Max: {weight.max().item():.6f}")
                
                if abs(weight_sum) < 1e-6:
                    print("\n⚠️  WARNING: Weights appear to be zero/uninitialized!")
                else:
                    print("\n✅ Weights look valid (non-zero)")
        else:
            print("\n❌ No state_dict found in checkpoint")
    else:
        print(f"\n⚠️  Checkpoint is not a dict, type: {type(checkpoint)}")
        
except Exception as e:
    print(f"\n❌ Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test model loading
print("\n" + "="*60)
print("Testing Model Loading")
print("="*60)

try:
    # Create model with ImageNet weights
    print("Creating model with ImageNet weights...")
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    print("✅ Model created")
    
    # Get ImageNet first layer weight
    imagenet_weight_sum = model.features[0][0].weight.data.sum().item()
    print(f"   ImageNet first layer weight sum: {imagenet_weight_sum:.4f}")
    
    # Try loading PlantNet weights
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
        
        if state_dict:
            print("\nLoading PlantNet weights on top...")
            # Remove classifier
            backbone_state_dict = {k: v for k, v in state_dict.items() 
                                  if not k.startswith('classifier') and not k.startswith('_fc')}
            
            missing, unexpected = model.load_state_dict(backbone_state_dict, strict=False)
            print(f"✅ Loaded PlantNet weights")
            print(f"   Missing keys: {len(missing)}")
            print(f"   Unexpected keys: {len(unexpected)}")
            
            # Check if weights changed
            new_weight_sum = model.features[0][0].weight.data.sum().item()
            print(f"\nAfter loading PlantNet weights:")
            print(f"   First layer weight sum: {new_weight_sum:.4f}")
            
            if abs(new_weight_sum - imagenet_weight_sum) < 1e-3:
                print("   ⚠️  Weight sum is very similar - might not have loaded")
            else:
                print("   ✅ Weight sum changed - PlantNet weights loaded!")
                
except Exception as e:
    print(f"\n❌ Error testing model: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test Complete")
print("="*60)

