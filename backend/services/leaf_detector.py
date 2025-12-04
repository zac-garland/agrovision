"""
Leaf detection service using YOLO model.
Detects and isolates leaves from plant images.
"""

from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2
import tempfile
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not installed. YOLO leaf detection will be unavailable.")
    print("   Install with: pip install ultralytics")


class LeafDetector:
    """Wrapper for YOLO leaf detection model."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize YOLO leaf detector.
        
        Args:
            model_path: Path to YOLO model file. If None, uses default path.
        """
        if not YOLO_AVAILABLE:
            self.model = None
            self.available = False
            return
        
        if model_path is None:
            # Default path relative to project root
            base_dir = Path(__file__).parent.parent.parent
            model_path = base_dir / "models" / "yolo11x_leaf.pt"
        
        try:
            if model_path.exists():
                print(f"🔄 Loading YOLO leaf detection model from {model_path}...")
                self.model = YOLO(str(model_path))
                self.available = True
                print("✅ YOLO model loaded successfully")
            else:
                print(f"⚠️  YOLO model not found at {model_path}")
                print("   Leaf detection will be disabled")
                self.model = None
                self.available = False
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.model = None
            self.available = False
    
    def detect_leaves(
        self, 
        image: Image.Image, 
        confidence_threshold: float = 0.15,
        return_boxes: bool = False
    ) -> Dict:
        """
        Detect leaves in an image.
        
        Args:
            image: PIL Image object
            confidence_threshold: Minimum confidence for detections
            return_boxes: If True, also return bounding box coordinates
            
        Returns:
            Dictionary with:
            - 'leaves': List of PIL Images (cropped leaves)
            - 'num_leaves': Number of leaves detected
            - 'boxes': (optional) List of bounding boxes [(x1, y1, x2, y2), ...]
            - 'confidences': (optional) List of confidence scores
        """
        if not self.available or self.model is None:
            return {
                'leaves': [image],  # Return original image if detection fails
                'num_leaves': 1,
                'boxes': [],
                'confidences': []
            }
        
        try:
            # YOLO works best with file paths (avoids numpy/PyTorch compatibility issues)
            # Convert PIL Image to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to temporary file and pass path to YOLO (most reliable method)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                image.save(tmp_file.name, 'JPEG', quality=95)
                tmp_path = tmp_file.name
            
            try:
                # Run YOLO inference using file path (most reliable method)
                results = self.model.predict(
                    tmp_path,
                    task="detect",
                    save=False,
                    conf=confidence_threshold,
                    verbose=False
                )
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            # Get the original image array for cropping
            img_array = np.array(image)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                # No leaves detected, return original image
                return {
                    'leaves': [image],
                    'num_leaves': 1,
                    'boxes': [],
                    'confidences': []
                }
            
            # Extract bounding boxes and crop leaves
            leaves = []
            boxes = []
            confidences = []
            
            # Convert to RGB if needed
            img_rgb = img_array
            if img_rgb.shape[2] == 4:  # RGBA
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
            elif len(img_rgb.shape) == 2:  # Grayscale
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
            
            # Process each detection
            for box in results[0].boxes:
                conf = float(box.conf.item())
                
                # Get normalized coordinates
                x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()
                
                # Convert to pixel coordinates
                img_height, img_width = img_rgb.shape[:2]
                x1_px = int(x1 * img_width)
                y1_px = int(y1 * img_height)
                x2_px = int(x2 * img_width)
                y2_px = int(y2 * img_height)
                
                # Ensure coordinates are within image bounds
                x1_px = max(0, x1_px)
                y1_px = max(0, y1_px)
                x2_px = min(img_width, x2_px)
                y2_px = min(img_height, y2_px)
                
                # Crop leaf from image
                leaf_crop = img_rgb[y1_px:y2_px, x1_px:x2_px]
                
                if leaf_crop.size > 0:
                    leaf_image = Image.fromarray(leaf_crop)
                    leaves.append(leaf_image)
                    boxes.append((x1_px, y1_px, x2_px, y2_px))
                    confidences.append(conf)
            
            if not leaves:
                # No valid leaves cropped, return original
                leaves = [image]
                num_leaves = 1
            else:
                num_leaves = len(leaves)
            
            result = {
                'leaves': leaves,
                'num_leaves': num_leaves,
                'confidences': confidences
            }
            
            if return_boxes:
                result['boxes'] = boxes
            
            return result
            
        except Exception as e:
            print(f"❌ Error in leaf detection: {e}")
            import traceback
            traceback.print_exc()
            # Return original image on error
            return {
                'leaves': [image],
                'num_leaves': 1,
                'boxes': [],
                'confidences': []
            }


# Global instance
_leaf_detector = None

def get_leaf_detector(model_path: Optional[Path] = None) -> LeafDetector:
    """Get or create global leaf detector instance."""
    global _leaf_detector
    if _leaf_detector is None:
        _leaf_detector = LeafDetector(model_path=model_path)
    return _leaf_detector

