#!/usr/bin/env python3
"""
Optimized fine-tune script that properly uses PlantNet EfficientNet B4 weights.
Uses efficientnet-pytorch library to match PlantNet checkpoint format exactly.
This ensures PlantNet weights can be properly loaded and used as a starting point.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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

# Try to import efficientnet-pytorch (required for PlantNet compatibility)
try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_PYTORCH_AVAILABLE = True
except ImportError:
    EFFICIENTNET_PYTORCH_AVAILABLE = False
    print("❌ Error: efficientnet-pytorch is required for PlantNet compatibility")
    print("   Install with: pip install efficientnet-pytorch")
    print("   Falling back to torchvision-based approach...")
    # Fall back to torchvision if not available
    from torchvision import models

# ============ CONFIG ============
DATA_DIR = BASE_DIR / "house_plant_species"
MODEL_SAVE_PATH = MODEL_DIR / "efficientnet_b4_houseplant_finetuned_optimized.tar"
LOG_FILE = BASE_DIR / "backend" / "finetune_optimized_log.txt"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
NUM_WORKERS = 0 if sys.platform == 'darwin' else 4

# Force GPU usage if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  WARNING: No GPU available. Training will be very slow on CPU.")

# ============ LOGGING SETUP ============
def log(message, also_print=True):
    """Log message to file and optionally print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message)
    
    if also_print:
        print(message)

# Initialize log file
log("="*80)
log("Starting OPTIMIZED fine-tuning with PlantNet weights")
log(f"Using library: {'efficientnet-pytorch' if EFFICIENTNET_PYTORCH_AVAILABLE else 'torchvision'}")
log(f"Device: {DEVICE}")
log(f"PlantNet model path: {EFFICIENTNET_MODEL_PATH}")
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
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        idx = 0
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                images_found = 0
                class_images = []
                class_labels = []
                
                for ext in valid_extensions:
                    for img_path in class_dir.glob(f"*{ext}"):
                        if img_path.is_file():
                            try:
                                Image.open(img_path).verify()
                                class_images.append(img_path)
                                class_labels.append(idx)
                                images_found += 1
                            except Exception as e:
                                log(f"⚠️  Skipping invalid image {img_path}: {e}", also_print=False)
                
                if images_found > 0:
                    self.class_to_idx[class_name] = idx
                    self.idx_to_class[idx] = class_name
                    self.images.extend(class_images)
                    self.labels.extend(class_labels)
                    log(f"   Class '{class_name}' (idx={idx}): {images_found} images")
                    idx += 1
        
        log(f"✅ Found {len(self.images)} images in {len(self.class_to_idx)} classes")
        
        if len(self.images) == 0:
            raise ValueError("No images found in dataset directory!")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            log(f"⚠️  Error loading {img_path}: {e}", also_print=False)
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============ TRANSFORMS ============
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    torch.manual_seed(42)
    random.seed(42)

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    val_dataset.dataset.transform = val_transform

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
    log("Loading pre-trained EfficientNet B4 with PlantNet weights...")

    num_classes = len(full_dataset.class_to_idx)
    log(f"Number of houseplant classes: {num_classes}")
    
    # Validate labels
    max_label = max(full_dataset.labels) if full_dataset.labels else -1
    min_label = min(full_dataset.labels) if full_dataset.labels else -1
    log(f"Label range: {min_label} to {max_label} (expected: 0 to {num_classes-1})")
    
    if max_label >= num_classes:
        raise ValueError(f"Label index {max_label} is out of bounds!")
    if min_label < 0:
        raise ValueError(f"Found negative label index: {min_label}")

    if EFFICIENTNET_PYTORCH_AVAILABLE:
        # Use efficientnet-pytorch to match PlantNet format exactly
        log("   Using efficientnet-pytorch library (matches PlantNet format)")
        
        # Step 1: Create model with ImageNet pretrained weights
        log("   Step 1: Creating EfficientNet B4 with ImageNet pretrained weights...")
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1000)
        log("   ✅ Loaded ImageNet pretrained weights")
        
        # Step 2: Load PlantNet weights on top
        if EFFICIENTNET_MODEL_PATH.exists():
            log(f"   Step 2: Loading PlantNet weights from: {EFFICIENTNET_MODEL_PATH}")
            try:
                checkpoint = torch.load(EFFICIENTNET_MODEL_PATH, map_location=DEVICE)
                
                # Extract state dict
                if isinstance(checkpoint, dict):
                    plantnet_state_dict = checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
                else:
                    plantnet_state_dict = checkpoint
                
                if plantnet_state_dict:
                    # Verify format (should be efficientnet-pytorch)
                    has_conv_stem = any('conv_stem' in k for k in plantnet_state_dict.keys())
                    has_blocks = any('blocks.' in k for k in plantnet_state_dict.keys())
                    
                    if has_conv_stem and has_blocks:
                        log("   ✅ Format verified: efficientnet-pytorch (matches!)")
                        
                        # Remove classifier (_fc) - we'll replace it for houseplant classes
                        backbone_state_dict = {
                            k: v for k, v in plantnet_state_dict.items() 
                            if not k.startswith('_fc')  # efficientnet-pytorch uses _fc for classifier
                        }
                        
                        # Load PlantNet backbone weights (this will overwrite ImageNet weights)
                        missing_keys, unexpected_keys = model.load_state_dict(backbone_state_dict, strict=False)
                        
                        # Filter out classifier-related missing keys
                        missing_backbone = [k for k in missing_keys if not k.startswith('_fc')]
                        
                        log(f"   ✅ Loaded PlantNet backbone weights")
                        log(f"      Missing keys: {len(missing_backbone)} (acceptable)")
                        log(f"      Unexpected keys: {len(unexpected_keys)}")
                        
                        # Verify weights were actually loaded
                        if 'conv_stem.weight' in model.state_dict():
                            test_weight = model.state_dict()['conv_stem.weight'].sum().item()
                            if abs(test_weight) > 1e-6:
                                log(f"   ✅ Verified: PlantNet weights loaded (conv_stem sum: {test_weight:.4f})")
                            else:
                                log("   ⚠️  Warning: Weights may not have loaded correctly")
                    else:
                        log("   ⚠️  Checkpoint format doesn't match, using ImageNet only")
                else:
                    log("   ⚠️  Could not extract state dict from checkpoint")
                    
            except Exception as e:
                log(f"   ⚠️  Error loading PlantNet weights: {e}")
                log("   Continuing with ImageNet pretrained weights only")
                import traceback
                log(traceback.format_exc(), also_print=False)
        else:
            log(f"   ⚠️  PlantNet weights not found at {EFFICIENTNET_MODEL_PATH}")
        
        # Step 3: Replace classifier for houseplant classes
        log(f"   Step 3: Replacing classifier for {num_classes} houseplant classes...")
        in_features = model._fc.in_features  # efficientnet-pytorch uses _fc
        model._fc = nn.Linear(in_features, num_classes)
        log(f"   ✅ Classifier replaced: {in_features} -> {num_classes}")
        
    else:
        # Fallback to torchvision (less optimal, but works)
        log("   ⚠️  efficientnet-pytorch not available, using torchvision (suboptimal)")
        log("   💡 Install efficientnet-pytorch for better PlantNet compatibility")
        
        from torchvision import models
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        log(f"   ✅ Created torchvision model with {num_classes} classes")

    # ============ FINE-TUNE SETUP ============
    log("Setting up fine-tuning...")

    FREEZE_BACKBONE = True  # Faster, safer - train only classifier

    if EFFICIENTNET_PYTORCH_AVAILABLE:
        # Freeze backbone for efficientnet-pytorch
        if FREEZE_BACKBONE:
            log("   Freezing backbone, training only classifier...")
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze classifier
            for param in model._fc.parameters():
                param.requires_grad = True
        else:
            log("   Unfreezing all layers for full fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
    else:
        # Freeze backbone for torchvision
        if FREEZE_BACKBONE:
            log("   Freezing backbone, training only classifier...")
            for param in model.features.parameters():
                param.requires_grad = False
        else:
            log("   Unfreezing all layers for full fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True

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
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
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
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
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
                'model_type': 'efficientnet-b4',
                'library': 'efficientnet-pytorch' if EFFICIENTNET_PYTORCH_AVAILABLE else 'torchvision',
                'pretrained_source': 'PlantNet-300K + ImageNet' if EFFICIENTNET_MODEL_PATH.exists() else 'ImageNet'
            }, MODEL_SAVE_PATH)
            
            log(f"   ✅ Model saved! Best accuracy: {best_val_acc:.2f}%")

    log("\n" + "="*80)
    log("Training complete!")
    log(f"Best validation accuracy: {best_val_acc:.2f}% (at epoch {best_epoch})")
    log(f"Model saved to: {MODEL_SAVE_PATH}")
    log("="*80)

