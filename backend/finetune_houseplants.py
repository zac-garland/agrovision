#!/usr/bin/env python3
"""
Fine-tune EfficientNet B4 model on house plant species dataset.
Uses the existing EfficientNet B4 model as a starting point for transfer learning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pathlib import Path
import os
import sys
import json
from datetime import datetime
from PIL import Image
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import BASE_DIR, MODEL_DIR, EFFICIENTNET_MODEL_PATH

# ============ CONFIG ============
DATA_DIR = BASE_DIR / "house_plant_species"
MODEL_SAVE_PATH = MODEL_DIR / "efficientnet_b4_houseplant_finetuned.tar"
LOG_FILE = BASE_DIR / "backend" / "finetune_log.txt"

BATCH_SIZE = 32
LEARNING_RATE = 0.001  # Lower for fine-tuning
EPOCHS = 10
# On macOS, use 0 workers to avoid multiprocessing issues
# On Linux, can use 4-8 workers for faster data loading
NUM_WORKERS = 0 if sys.platform == 'darwin' else 4  # For DataLoader

# Force GPU usage if available (check CUDA first, then MPS for Apple Silicon, then CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS)")
    print("   Metal Performance Shaders enabled for GPU acceleration")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  WARNING: No GPU available. Training will be very slow on CPU.")
    print("   Consider using a GPU-enabled machine or reducing BATCH_SIZE.")

# ============ LOGGING SETUP ============
def log(message, also_print=True):
    """Log message to file and optionally print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    
    with open(LOG_FILE, 'a') as f:
        f.write(log_message)
    
    if also_print:
        print(message)

# Initialize log file
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
log("="*80)
log("Starting fine-tuning session")
log(f"Device: {DEVICE}")
log(f"Data directory: {DATA_DIR}")
log(f"Model save path: {MODEL_SAVE_PATH}")
log("="*80)

# ============ DATASET ============
class HouseplantDataset(Dataset):
    """Load houseplant images from folder structure"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.root_dir}")
        
        # Build class mapping - only include classes with images
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        idx = 0  # Sequential index starting from 0
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                
                # Add all images from this class
                images_found = 0
                class_images = []
                class_labels = []
                
                for ext in valid_extensions:
                    for img_path in class_dir.glob(f"*{ext}"):
                        if img_path.is_file():
                            try:
                                # Verify image is valid
                                Image.open(img_path).verify()
                                class_images.append(img_path)
                                class_labels.append(idx)  # Use sequential idx
                                images_found += 1
                            except Exception as e:
                                log(f"⚠️  Skipping invalid image {img_path}: {e}", also_print=False)
                
                # Only add class if it has images
                if images_found > 0:
                    self.class_to_idx[class_name] = idx
                    self.idx_to_class[idx] = class_name
                    self.images.extend(class_images)
                    self.labels.extend(class_labels)
                    log(f"   Class '{class_name}' (idx={idx}): {images_found} images")
                    idx += 1  # Only increment after adding a valid class
        
        log(f"✅ Found {len(self.images)} images in {len(self.class_to_idx)} classes")
        
        if len(self.images) == 0:
            raise ValueError("No images found in dataset directory!")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            log(f"⚠️  Error loading {img_path}: {e}", also_print=False)
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============ TRANSFORMS ============
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),  # Augmentation
    transforms.RandomRotation(15),           # Augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Augmentation
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============ MAIN EXECUTION ============
if __name__ == '__main__':
    # ============ LOAD DATA ============
    log("Loading dataset...")
    try:
        full_dataset = HouseplantDataset(DATA_DIR, transform=train_transform)
    except Exception as e:
        log(f"❌ Error loading dataset: {e}")
        sys.exit(1)

    # Create train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Update val dataset transform
    val_dataset.dataset.transform = val_transform

    # Pin memory only for CUDA, not for MPS or CPU
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory
    )

    log(f"Training samples: {train_size}, Validation samples: {val_size}")

    # ============ LOAD PRE-TRAINED MODEL ============
    log("Loading pre-trained EfficientNet B4...")

    num_classes = len(full_dataset.class_to_idx)
    log(f"Number of houseplant classes: {num_classes}")
    
    # Validate labels are in correct range (0 to num_classes-1)
    max_label = max(full_dataset.labels) if full_dataset.labels else -1
    min_label = min(full_dataset.labels) if full_dataset.labels else -1
    log(f"Label range: {min_label} to {max_label} (expected: 0 to {num_classes-1})")
    
    if max_label >= num_classes:
        raise ValueError(
            f"Label index {max_label} is out of bounds! "
            f"Model expects {num_classes} classes (indices 0-{num_classes-1}), "
            f"but found label {max_label}. Check dataset class indexing."
        )
    
    if min_label < 0:
        raise ValueError(f"Found negative label index: {min_label}")

    # ALWAYS start with ImageNet pretrained weights (this is critical!)
    log("   Starting with ImageNet pretrained weights...")
    try:
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        log("   ✅ Loaded ImageNet pretrained weights")
    except AttributeError:
        # Older torchvision versions
        model = models.efficientnet_b4(pretrained=True)
        log("   ✅ Loaded ImageNet pretrained weights (legacy)")
    
    # Replace final layer for our number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    log(f"   Modified classifier for {num_classes} classes")
    
    # Optionally load PlantNet weights on top of ImageNet weights
    if EFFICIENTNET_MODEL_PATH.exists():
        log(f"   Attempting to load PlantNet weights from: {EFFICIENTNET_MODEL_PATH}")
        try:
            checkpoint = torch.load(EFFICIENTNET_MODEL_PATH, map_location=DEVICE)
            
            # Extract state dict
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
                
                if state_dict:
                    # Detect checkpoint format (efficientnet-pytorch vs torchvision)
                    has_conv_stem = any('conv_stem' in k for k in state_dict.keys())
                    has_blocks = any('blocks.' in k for k in state_dict.keys())
                    has_features = any('features.' in k for k in state_dict.keys())
                    
                    if has_conv_stem and has_blocks:
                        log("   ✅ Detected: PlantNet uses efficientnet-pytorch format")
                        log("   ⚠️  WARNING: Layer names don't match torchvision format")
                        log("   💡 PlantNet weights use different architecture (conv_stem/blocks vs features)")
                        log("   ✅ Will use ImageNet weights (compatible) + PlantNet knowledge via fine-tuning")
                        log("   ℹ️  The model will still benefit from ImageNet pretraining for plant features")
                        # Note: Can't directly load efficientnet-pytorch weights into torchvision model
                        # Architecture differences prevent direct mapping
                    elif has_features:
                        log("   ✅ Detected: Compatible torchvision format")
                        # Verify weights look valid (check first layer)
                        first_layer_key = None
                        for key in state_dict.keys():
                            if 'features.0' in key and 'weight' in key:
                                first_layer_key = key
                                break
                        
                        if first_layer_key:
                            weight_sum = state_dict[first_layer_key].sum().item()
                            log(f"   First layer weight sum: {weight_sum:.4f} (should be non-zero)")
                            
                            if abs(weight_sum) < 1e-6:
                                log("   ⚠️  Weights appear to be zero/uninitialized, skipping PlantNet weights")
                            else:
                                # Remove classifier layer and load backbone only
                                backbone_state_dict = {k: v for k, v in state_dict.items() 
                                                      if not k.startswith('classifier') and not k.startswith('_fc')}
                                
                                missing_keys, unexpected_keys = model.load_state_dict(backbone_state_dict, strict=False)
                                log(f"   ✅ Loaded PlantNet backbone weights")
                                log(f"      Missing keys: {len(missing_keys)} (expected, classifier removed)")
                                log(f"      Unexpected keys: {len(unexpected_keys)}")
                                
                                # Verify weights were actually loaded
                                test_weight = model.features[0][0].weight.data.sum().item()
                                if abs(test_weight) > 1e-6:
                                    log(f"   ✅ Verified: Model weights loaded (first layer sum: {test_weight:.4f})")
                                else:
                                    log("   ⚠️  Warning: Model weights may not have loaded correctly")
                        else:
                            log("   ⚠️  Could not find first layer in torchvision format")
                    else:
                        log("   ⚠️  Unknown checkpoint format, using ImageNet weights only")
                else:
                    log("   ⚠️  No state_dict found in checkpoint, using ImageNet weights only")
            else:
                log("   ⚠️  Unexpected checkpoint format, using ImageNet weights only")
                
        except Exception as e:
            log(f"   ⚠️  Error loading PlantNet weights: {e}")
            log("   Continuing with ImageNet pretrained weights only")
    else:
        log(f"   ℹ️  PlantNet weights not found, using ImageNet pretrained weights")

    # ============ FINE-TUNE SETUP ============
    log("Setting up fine-tuning...")

    # Option 1: Freeze backbone, train only final layer (FASTER, SAFER)
    FREEZE_BACKBONE = True  # Set to False to unfreeze everything

    if FREEZE_BACKBONE:
        log("   Freezing backbone, training only classifier...")
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        log("   Unfreezing all layers for full fine-tuning...")
        for param in model.parameters():
            param.requires_grad = True

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"   Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    model = model.to(DEVICE)

    # ============ TRAINING SETUP ============
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # ============ TRAINING LOOP ============
    best_val_acc = 0
    best_epoch = 0

    log("\n" + "="*80)
    log("Starting training...")
    log("="*80)

    for epoch in range(EPOCHS):
        epoch_start_time = datetime.now()
        
        log(f"\n{'='*80}")
        log(f"Epoch {epoch+1}/{EPOCHS}")
        log(f"{'='*80}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Progress update every 10 batches
            if (batch_idx + 1) % 10 == 0:
                current_acc = 100 * train_correct / train_total
                log(f"   Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%", also_print=False)
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch results
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        log(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        log(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        log(f"Learning Rate: {current_lr:.6f}")
        log(f"Epoch Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            log(f"✅ New best model! Saving to {MODEL_SAVE_PATH}...")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_val_acc,
                'num_classes': num_classes,
                'class_to_idx': full_dataset.class_to_idx,
                'idx_to_class': full_dataset.idx_to_class,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, MODEL_SAVE_PATH)
            
            log(f"   ✅ Model saved! Best accuracy: {best_val_acc:.2f}%")

    log("\n" + "="*80)
    log("Training complete!")
    log(f"Best validation accuracy: {best_val_acc:.2f}% (at epoch {best_epoch})")
    log(f"Model saved to: {MODEL_SAVE_PATH}")
    log("="*80)

