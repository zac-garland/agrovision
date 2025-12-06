#!/usr/bin/env python3
"""
Performance evaluation script for AgroVision+ classifiers.
Evaluates both species and disease classifiers and generates summary tables.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Don't import model wrappers here - we'll load models directly from checkpoints
# This avoids import issues with backend.config

# Try to import EfficientNet
try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_PYTORCH_AVAILABLE = True
except ImportError:
    EFFICIENTNET_PYTORCH_AVAILABLE = False
    print("❌ Error: efficientnet-pytorch is required")
    sys.exit(1)

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random


class ClassifierEvaluator:
    """Evaluates classifier performance on validation/test set."""
    
    def __init__(self, model_path, model_type='disease'):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to model checkpoint
            model_type: 'species' or 'disease'
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.num_classes = 0
        
        self.load_model()
    
    def load_model(self):
        """Load model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"🔄 Loading {self.model_type} classifier from {self.model_path}...")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract class mappings
        self.class_to_idx = checkpoint.get('class_to_idx', {})
        self.idx_to_class = checkpoint.get('idx_to_class', {})
        self.num_classes = checkpoint.get('num_classes', len(self.class_to_idx))
        
        # Debug: Print mapping info
        print(f"  Model class mappings: {len(self.class_to_idx)} classes in class_to_idx")
        print(f"  Model idx mappings: {len(self.idx_to_class)} classes in idx_to_class")
        if self.idx_to_class:
            sample_classes = list(self.idx_to_class.values())[:5]
            print(f"  Sample classes: {sample_classes}")
        
        # Create model
        if EFFICIENTNET_PYTORCH_AVAILABLE:
            self.model = EfficientNet.from_name('efficientnet-b4', num_classes=self.num_classes)
        else:
            import torchvision.models as models
            self.model = models.efficientnet_b4(weights=None)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, self.num_classes)
        
        # Load weights
        state_dict = (checkpoint.get('model') or 
                     checkpoint.get('model_state_dict') or 
                     checkpoint.get('state_dict') or
                     checkpoint)
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Loaded model: {self.num_classes} classes")
    
    def preprocess_image(self, image_pil):
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
    
    def predict(self, image_pil):
        """Get model predictions."""
        image_tensor = self.preprocess_image(image_pil).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
        
        return predicted_idx, probabilities.cpu().numpy(), confidence
    
    def evaluate_dataset(self, dataset, dataset_name="Validation"):
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: Dataset object with __getitem__ returning (image, label)
            dataset_name: Name for reporting
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n📊 Evaluating on {dataset_name} set...")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Run predictions
        for images, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            images = images.to(self.device)
            labels = labels.numpy()
            
            with torch.no_grad():
                outputs = self.model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Debug: Print label ranges
        print(f"  Label range: {all_labels.min()} to {all_labels.max()}")
        print(f"  Prediction range: {all_predictions.min()} to {all_predictions.max()}")
        print(f"  Model expects classes: 0 to {self.num_classes - 1}")
        print(f"  Unique labels in data: {len(np.unique(all_labels))}")
        print(f"  Unique predictions: {len(np.unique(all_predictions))}")
        
        # Check for label mismatches
        if all_labels.max() >= self.num_classes:
            print(f"  ⚠️  WARNING: Some labels ({all_labels.max()}) exceed model's class count ({self.num_classes})")
        if all_predictions.max() >= self.num_classes:
            print(f"  ⚠️  WARNING: Some predictions ({all_predictions.max()}) exceed model's class count ({self.num_classes})")
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate comprehensive metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics - only for classes present in the data
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        num_classes_in_data = len(unique_classes)
        
        # Calculate per-class metrics only for classes that appear in the data
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=unique_classes)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=unique_classes)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=unique_classes)
        
        # Create full arrays for all model classes (pad with zeros for missing classes)
        metrics['precision_per_class'] = np.zeros(self.num_classes)
        metrics['recall_per_class'] = np.zeros(self.num_classes)
        metrics['f1_per_class'] = np.zeros(self.num_classes)
        
        for i, class_idx in enumerate(unique_classes):
            if 0 <= class_idx < self.num_classes:
                metrics['precision_per_class'][class_idx] = precision_per_class[i]
                metrics['recall_per_class'][class_idx] = recall_per_class[i]
                metrics['f1_per_class'][class_idx] = f1_per_class[i]
        
        metrics['num_classes_in_data'] = num_classes_in_data
        metrics['unique_classes'] = unique_classes
        
        # ROC-AUC (multi-class)
        try:
            # Binarize labels for multi-class ROC-AUC
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            if y_true_bin.shape[1] == 1:
                # Binary case
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0])
            else:
                # Multi-class case
                metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
                metrics['roc_auc_weighted'] = roc_auc_score(y_true_bin, y_proba, average='weighted', multi_class='ovr')
        except Exception as e:
            print(f"⚠️  Could not calculate ROC-AUC: {e}")
            metrics['roc_auc_macro'] = None
            metrics['roc_auc_weighted'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Top-k accuracy (if applicable)
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            top3_correct = 0
            for i, true_label in enumerate(y_true):
                top3_preds = np.argsort(y_proba[i])[-3:][::-1]
                if true_label in top3_preds:
                    top3_correct += 1
            metrics['top3_accuracy'] = top3_correct / len(y_true)
        else:
            metrics['top3_accuracy'] = None
        
        return metrics
    
    def print_summary_table(self, metrics, dataset_name="Validation"):
        """Print a formatted summary table."""
        print(f"\n{'='*80}")
        print(f"{self.model_type.upper()} CLASSIFIER - {dataset_name.upper()} SET PERFORMANCE")
        print(f"{'='*80}")
        
        # Main metrics table
        data = {
            'Metric': [
                'Accuracy',
                'Precision (Macro)',
                'Precision (Weighted)',
                'Recall (Macro)',
                'Recall (Weighted)',
                'F1 Score (Macro)',
                'F1 Score (Weighted)',
                'Top-3 Accuracy',
                'ROC-AUC (Macro)',
                'ROC-AUC (Weighted)'
            ],
            'Value': [
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision_macro']:.4f}",
                f"{metrics['precision_weighted']:.4f}",
                f"{metrics['recall_macro']:.4f}",
                f"{metrics['recall_weighted']:.4f}",
                f"{metrics['f1_macro']:.4f}",
                f"{metrics['f1_weighted']:.4f}",
                f"{metrics['top3_accuracy']:.4f}" if metrics['top3_accuracy'] else "N/A",
                f"{metrics['roc_auc_macro']:.4f}" if metrics.get('roc_auc_macro') else "N/A",
                f"{metrics['roc_auc_weighted']:.4f}" if metrics.get('roc_auc_weighted') else "N/A"
            ]
        }
        
        df = pd.DataFrame(data)
        print("\n📊 Overall Performance Metrics:")
        print(df.to_string(index=False))
        
        # Per-class metrics (sample of classes if too many)
        # Only show classes that exist in the metrics (may be fewer than num_classes)
        num_metric_classes = len(metrics['f1_per_class'])
        
        if num_metric_classes <= 20:
            print(f"\n📋 Per-Class Performance (showing all {num_metric_classes} classes present in validation set):")
            class_data = []
            for idx in range(num_metric_classes):
                class_name = self.idx_to_class.get(idx, f"Class_{idx}")
                class_data.append({
                    'Class': class_name[:40],  # Truncate long names
                    'Precision': f"{metrics['precision_per_class'][idx]:.4f}",
                    'Recall': f"{metrics['recall_per_class'][idx]:.4f}",
                    'F1 Score': f"{metrics['f1_per_class'][idx]:.4f}"
                })
            
            class_df = pd.DataFrame(class_data)
            print(class_df.to_string(index=False))
        else:
            print(f"\n📋 Per-Class Performance (showing top 10 and bottom 10 by F1 score):")
            num_classes_in_data = metrics.get('num_classes_in_data', num_metric_classes)
            print(f"   Note: {num_classes_in_data} classes present in validation set (model has {self.num_classes} total classes)")
            
            # Only include classes that actually appear in the data
            unique_classes = metrics.get('unique_classes', np.arange(num_metric_classes))
            class_f1 = [(self.idx_to_class.get(int(i), f"Class_{i}"), metrics['f1_per_class'][int(i)]) 
                        for i in unique_classes if 0 <= int(i) < self.num_classes]
            class_f1.sort(key=lambda x: x[1], reverse=True)
            
            # Top 10
            print("\nTop 10 Classes:")
            top_data = [{'Class': name[:40], 'F1 Score': f"{f1:.4f}"} 
                       for name, f1 in class_f1[:10]]
            print(pd.DataFrame(top_data).to_string(index=False))
            
            # Bottom 10
            print("\nBottom 10 Classes:")
            bottom_data = [{'Class': name[:40], 'F1 Score': f"{f1:.4f}"} 
                          for name, f1 in class_f1[-10:]]
            print(pd.DataFrame(bottom_data).to_string(index=False))
        
        print(f"\n{'='*80}\n")


def load_validation_dataset(model_type, data_dir, generalization_mapping_path=None):
    """
    Load validation dataset for evaluation.
    This recreates the same split used during training (seed=42).
    """
    # Import from classifier data directory
    classifier_data_path = PROJECT_ROOT / "classifier data"
    sys.path.insert(0, str(classifier_data_path))
    
    # Import the dataset class
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_disease_classifier",
        classifier_data_path / "train_disease_classifier.py"
    )
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    PlantVillageDataset = train_module.PlantVillageDataset
    
    BASE_DIR = Path(__file__).parent.parent / "classifier data"
    
    if model_type == 'species':
        # Load species dataset (same as training)
        BASE_DIR = PROJECT_ROOT / "classifier data"
        
        # Import species dataset classes
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_species_classifier",
            BASE_DIR / "train_species_classifier.py"
        )
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        
        PlantNetDataset = train_module.PlantNetDataset
        FolderDataset = train_module.FolderDataset
        
        # Note: PlantVillageHealthyDataset is defined inline in training script
        # For evaluation, we'll focus on houseplant + PlantNet datasets
        # which are the primary data sources
        
        # Load datasets in the SAME ORDER as training to recreate unified mapping
        HOUSE_PLANT_DIR = BASE_DIR / "house_plant_species"
        PLANTNET_DIR = BASE_DIR / "plantnet"
        PLANTVILLAGE_DIR = BASE_DIR / "plantvillage" / "plantvillage dataset" / "color"
        
        # Step 1: Load houseplant dataset first (creates initial class mapping)
        if not HOUSE_PLANT_DIR.exists():
            print("⚠️  Houseplant directory not found")
            return None
        
        houseplant_dataset = FolderDataset(HOUSE_PLANT_DIR, transform=None)
        print(f"  ✅ Loaded houseplant dataset: {len(houseplant_dataset)} images, {len(houseplant_dataset.class_to_idx)} classes")
        
        # Step 2: Create unified mapping starting with houseplant
        unified_class_mapping = houseplant_dataset.class_to_idx.copy()
        
        # Step 3: Load PlantNet with unified mapping (extends it)
        plantnet_dataset = None
        if PLANTNET_DIR.exists():
            plantnet_dataset = PlantNetDataset(PLANTNET_DIR, transform=None, class_mapping=unified_class_mapping)
            # Update unified mapping with PlantNet's extended mapping
            unified_class_mapping = plantnet_dataset.class_to_idx.copy()
            print(f"  ✅ Loaded PlantNet dataset: {len(plantnet_dataset)} images")
            print(f"     Total classes after PlantNet: {len(unified_class_mapping)}")
        
        # Step 4: Load PlantVillage healthy (extends unified mapping further)
        # This is the same logic as training script
        plantvillage_images = []
        plantvillage_labels = []
        current_max_idx = max(unified_class_mapping.values()) if unified_class_mapping else -1
        
        if PLANTVILLAGE_DIR.exists():
            try:
                # Find all healthy class directories
                healthy_dirs = [d for d in PLANTVILLAGE_DIR.iterdir() 
                              if d.is_dir() and 'healthy' in d.name.lower()]
                
                valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.webp', '.WEBP'}
                
                for healthy_dir in healthy_dirs:
                    # Extract plant name from directory name (e.g., "Tomato___healthy" -> "Tomato")
                    plant_name = healthy_dir.name.split('___')[0] if '___' in healthy_dir.name else healthy_dir.name.replace('_healthy', '').replace('healthy', '')
                    
                    # Add to unified mapping if not present
                    if plant_name not in unified_class_mapping:
                        current_max_idx += 1
                        unified_class_mapping[plant_name] = current_max_idx
                    
                    # Collect images
                    class_images = []
                    for img_path in healthy_dir.rglob('*'):
                        if img_path.suffix.lower() in valid_extensions:
                            class_images.append(img_path)
                    
                    # Add images and labels
                    plantvillage_images.extend(class_images)
                    plantvillage_labels.extend([unified_class_mapping[plant_name]] * len(class_images))
                
                print(f"  ✅ Loaded PlantVillage healthy: {len(plantvillage_images)} images")
                print(f"     Total classes after PlantVillage: {len(unified_class_mapping)}")
            except Exception as e:
                print(f"  ⚠️  Error loading PlantVillage healthy: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 5: Combine all datasets
        datasets = [houseplant_dataset]
        if plantnet_dataset:
            datasets.append(plantnet_dataset)
        
        # Create PlantVillage dataset if we have images
        if plantvillage_images:
            class PlantVillageHealthyDataset(Dataset):
                def __init__(self, images, labels, transform):
                    self.images = images
                    self.labels = labels
                    self.transform = transform
                
                def __len__(self):
                    return len(self.images)
                
                def __getitem__(self, idx):
                    try:
                        image = Image.open(self.images[idx]).convert('RGB')
                        if self.transform:
                            image = self.transform(image)
                        return image, self.labels[idx]
                    except Exception as e:
                        print(f"Error loading PlantVillage image: {e}")
                        image = Image.new('RGB', (224, 224))
                        if self.transform:
                            image = self.transform(image)
                        return image, self.labels[idx]
            
            plantvillage_dataset = PlantVillageHealthyDataset(plantvillage_images, plantvillage_labels, transform=None)
            datasets.append(plantvillage_dataset)
        
        # Concatenate datasets
        from torch.utils.data import ConcatDataset
        full_dataset = ConcatDataset(datasets)
        print(f"  ✅ Total combined dataset: {len(full_dataset)} images, {len(unified_class_mapping)} classes")
        
        # Get all labels using the unified mapping
        # For ConcatDataset, we need to map through each sub-dataset
        all_labels = []
        label_map = {}
        current_idx = 0
        
        for dataset in datasets:
            dataset_len = len(dataset)
            if hasattr(dataset, 'labels'):
                # Direct label access (PlantNetDataset, PlantVillageHealthyDataset)
                for i in range(dataset_len):
                    label_map[current_idx + i] = dataset.labels[i]
            elif hasattr(dataset, 'images') and hasattr(dataset, 'class_to_idx'):
                # FolderDataset - map to unified indices
                for i in range(dataset_len):
                    if i < len(dataset.images):
                        img_path = dataset.images[i]
                        # Find class name from path
                        for class_name, local_idx in dataset.class_to_idx.items():
                            if class_name in str(img_path):
                                # Map to unified class index
                                unified_idx = unified_class_mapping.get(class_name, local_idx)
                                label_map[current_idx + i] = unified_idx
                                break
                        else:
                            # Default to first class if not found
                            label_map[current_idx + i] = 0
            current_idx += dataset_len
        
        all_labels = [label_map[i] for i in range(len(full_dataset))]
        
        # Recreate the same stratified split (seed=42)
        from sklearn.model_selection import train_test_split
        from collections import Counter
        
        all_indices = list(range(len(full_dataset)))
        
        # Check for classes with insufficient samples for stratification
        label_counts = Counter(all_labels)
        min_samples_per_class = min(label_counts.values()) if label_counts else 0
        
        if min_samples_per_class < 2:
            print(f"  ⚠️  Some classes have < 2 samples (min: {min_samples_per_class})")
            print(f"     Using non-stratified split for evaluation")
            # Use non-stratified split if some classes have too few samples
            _, val_indices = train_test_split(
                all_indices,
                test_size=0.2,
                stratify=None,
                random_state=42
            )
        else:
            # Use stratified split
            _, val_indices = train_test_split(
                all_indices,
                test_size=0.2,
                stratify=all_labels,
                random_state=42
            )
        
        # Create validation subset with transforms
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        class TransformDataset(Dataset):
            def __init__(self, dataset, indices, transform, label_map=None):
                self.dataset = dataset
                self.indices = indices
                self.transform = transform
                self.label_map = label_map  # Maps dataset index to unified label
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                actual_idx = self.indices[idx]
                
                # For ConcatDataset, need to find which sub-dataset contains this index
                if isinstance(self.dataset, ConcatDataset):
                    # Find which sub-dataset and local index
                    cumsum = 0
                    for sub_dataset in self.dataset.datasets:
                        if actual_idx < cumsum + len(sub_dataset):
                            local_idx = actual_idx - cumsum
                            image, _ = sub_dataset[local_idx]
                            
                            # Use unified label from label_map (already mapped)
                            label = self.label_map[actual_idx] if self.label_map and actual_idx in self.label_map else 0
                            break
                        cumsum += len(sub_dataset)
                else:
                    image, _ = self.dataset[actual_idx]
                    # Use unified label from label_map
                    label = self.label_map[actual_idx] if self.label_map and actual_idx in self.label_map else 0
                
                if self.transform:
                    image = self.transform(image)
                return image, label
        
        val_dataset = TransformDataset(full_dataset, val_indices, val_transform, label_map=label_map)
        
        print(f"  ✅ Created validation dataset: {len(val_indices)} samples")
        print(f"     Using unified class mapping with {len(unified_class_mapping)} classes")
        
        return val_dataset
    else:
        # Load disease dataset
        PLANTVILLAGE_DIR = BASE_DIR / "plantvillage" / "plantvillage dataset" / "color"
        
        # Create dataset (same as training)
        full_dataset = PlantVillageDataset(
            PLANTVILLAGE_DIR,
            transform=None,
            max_samples_per_class=None,
            generalization_mapping_path=generalization_mapping_path
        )
        
        # Recreate the same stratified split (seed=42)
        from sklearn.model_selection import train_test_split
        all_indices = list(range(len(full_dataset)))
        all_labels = [full_dataset.labels[i] for i in all_indices]
        
        _, val_indices = train_test_split(
            all_indices,
            test_size=0.2,
            stratify=all_labels,
            random_state=42
        )
        
        # Create validation subset
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        class TransformDataset(Dataset):
            def __init__(self, dataset, indices, transform):
                self.dataset = dataset
                self.indices = indices
                self.transform = transform
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                actual_idx = self.indices[idx]
                img_path = self.dataset.images[actual_idx]
                label = self.dataset.labels[actual_idx]
                
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label
        
        val_dataset = TransformDataset(full_dataset, val_indices, val_transform)
        
        return val_dataset


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate classifier performance")
    parser.add_argument(
        "--species-model",
        type=str,
        default=str(PROJECT_ROOT / "models" / "efficientnet_b4_species_classifier.tar"),
        help="Path to species classifier model"
    )
    parser.add_argument(
        "--disease-model",
        type=str,
        default=str(PROJECT_ROOT / "models" / "efficientnet_b4_disease_classifier.tar"),
        help="Path to disease classifier model"
    )
    parser.add_argument(
        "--evaluate-species",
        action="store_true",
        help="Evaluate species classifier"
    )
    parser.add_argument(
        "--evaluate-disease",
        action="store_true",
        help="Evaluate disease classifier"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate both classifiers"
    )
    
    args = parser.parse_args()
    
    if not args.evaluate_species and not args.evaluate_disease and not args.all:
        print("⚠️  No classifier specified. Use --evaluate-species, --evaluate-disease, or --all")
        return
    
    # Evaluate disease classifier
    if args.evaluate_disease or args.all:
        if Path(args.disease_model).exists():
            print("\n" + "="*80)
            print("EVALUATING DISEASE CLASSIFIER")
            print("="*80)
            
            evaluator = ClassifierEvaluator(args.disease_model, model_type='disease')
            
            # Load validation dataset
            generalization_mapping = PROJECT_ROOT / "classifier data" / "disease_generalization_mapping_7class.json"
            val_dataset = load_validation_dataset('disease', None, generalization_mapping)
            
            if val_dataset:
                metrics = evaluator.evaluate_dataset(val_dataset, "Validation")
                evaluator.print_summary_table(metrics, "Validation")
            else:
                print("⚠️  Could not load validation dataset")
        else:
            print(f"⚠️  Disease model not found: {args.disease_model}")
    
    # Evaluate species classifier
    if args.evaluate_species or args.all:
        if Path(args.species_model).exists():
            print("\n" + "="*80)
            print("EVALUATING SPECIES CLASSIFIER")
            print("="*80)
            
            evaluator = ClassifierEvaluator(args.species_model, model_type='species')
            
            # Load validation dataset
            val_dataset = load_validation_dataset('species', None, None)
            
            if val_dataset:
                # Map dataset labels to model's unified class indices
                # The model's class_to_idx is the unified mapping
                # We need to map dataset class names to these indices
                if hasattr(val_dataset, 'dataset') and evaluator.class_to_idx:
                    # Create label mapper: dataset class name -> model class index
                    # This is complex - for now, we'll evaluate as-is and note the issue
                    print(f"  ⚠️  Note: Dataset labels may not match model's unified class mapping")
                    print(f"     This could cause lower accuracy if labels don't align")
                    print(f"     Model expects unified mapping with {len(evaluator.class_to_idx)} classes")
                
                metrics = evaluator.evaluate_dataset(val_dataset, "Validation")
                evaluator.print_summary_table(metrics, "Validation")
            else:
                print("⚠️  Could not load validation dataset")
        else:
            print(f"⚠️  Species model not found: {args.species_model}")


if __name__ == "__main__":
    main()

