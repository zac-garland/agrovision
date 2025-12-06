"""
Lesion analysis service for detecting and highlighting potential disease areas.
Uses image processing techniques to identify abnormalities in leaf images.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class LesionAnalyzer:
    """Analyzes leaf images to detect and highlight potential lesions/disease areas."""
    
    def __init__(self):
        """Initialize lesion analyzer."""
        # HSV ranges for healthy green (can be adjusted)
        self.healthy_green_lower = np.array([25, 30, 30], dtype=np.uint8)
        self.healthy_green_upper = np.array([95, 255, 255], dtype=np.uint8)
        
        # HSV ranges for common lesion colors (yellowing, browning, spotting)
        self.lesion_yellow_lower = np.array([10, 50, 50], dtype=np.uint8)
        self.lesion_yellow_upper = np.array([25, 255, 255], dtype=np.uint8)
        
        self.lesion_brown_lower = np.array([5, 50, 30], dtype=np.uint8)
        self.lesion_brown_upper = np.array([25, 255, 200], dtype=np.uint8)
    
    def check_green_percentage(self, image: Image.Image) -> float:
        """
        Quick check to get green percentage without full analysis.
        Used for filtering out non-leaf detections (flowers, pots, soil, etc.).
        
        Args:
            image: PIL Image of a detected object
            
        Returns:
            Green percentage (0-100)
        """
        try:
            # Convert PIL to numpy array
            img_array = np.array(image.convert('RGB'))
            
            # Segment leaf from background
            segmented = self._segment_leaf(img_array)
            
            # Calculate green percentage
            green_pct, _ = self._calculate_green_percentage(segmented)
            
            return float(green_pct)
            
        except Exception as e:
            print(f"⚠️  Error checking green percentage: {e}")
            return 0.0
    
    def analyze_leaf(self, image: Image.Image) -> Dict:
        """
        Analyze a leaf image for potential lesions and health indicators.
        
        Args:
            image: PIL Image of a leaf
            
        Returns:
            Dictionary with analysis results:
            - 'health_score': 0-1 (1 = very healthy)
            - 'green_percentage': Percentage of healthy green pixels
            - 'lesion_percentage': Percentage of potential lesion pixels
            - 'lesion_areas': List of lesion region coordinates
            - 'anomaly_regions': Regions that deviate from healthy green
            - 'has_potential_issues': Boolean
        """
        try:
            # Convert PIL to numpy array
            img_array = np.array(image.convert('RGB'))
            
            # Get image dimensions
            height, width = img_array.shape[:2]
            
            # Segment leaf from background (optional, can improve accuracy)
            segmented = self._segment_leaf(img_array)
            
            # Analyze colors
            green_pct, green_mask = self._calculate_green_percentage(segmented)
            lesion_pct, lesion_mask = self._detect_lesions(segmented)
            
            # Find lesion regions
            lesion_areas = self._find_lesion_regions(lesion_mask)
            
            # Calculate health score with emphasis on lesion detection
            # Lesion percentage is the PRIMARY indicator - it should dominate the score
            # Green percentage is secondary and should NOT mask lesion detection
            # Formula: Lesions always reduce score significantly, green only helps if minimal/no lesions
            lesion_impact = (lesion_pct / 100.0) * 2.0  # Lesions weighted 2.0x (more aggressive)
            
            # Green only contributes if there are minimal lesions (< 1%)
            # This prevents green from masking lesion detection
            if lesion_pct < 1.0:
                green_contribution = (green_pct / 100.0) * 0.1  # Minimal green boost only when healthy
            else:
                green_contribution = 0.0  # No green boost when lesions are present
            
            health_score = max(0.0, min(1.0, 1.0 - lesion_impact + green_contribution))
            
            # Determine if there are potential issues - primarily based on lesions
            # Lesion detection is the main concern, not green color
            has_potential_issues = bool(
                lesion_pct > 3.0  # More than 3% lesion-like areas indicates issues
                # Removed green_pct check since many healthy plants aren't green
            )
            
            return {
                'health_score': float(health_score),
                'green_percentage': float(green_pct),
                'lesion_percentage': float(lesion_pct),
                'lesion_areas': lesion_areas,
                'has_potential_issues': bool(has_potential_issues),  # Ensure Python bool
                'num_lesion_regions': len(lesion_areas)
            }
            
        except Exception as e:
            print(f"❌ Error in lesion analysis: {e}")
            import traceback
            traceback.print_exc()
            return {
                'health_score': 0.5,
                'green_percentage': 50.0,
                'lesion_percentage': 0.0,
                'lesion_areas': [],
                'has_potential_issues': False,
                'num_lesion_regions': 0
            }
    
    def highlight_lesions(self, image: Image.Image, threshold: float = 0.1) -> Image.Image:
        """
        Create a visualization highlighting potential lesion areas.
        
        Args:
            image: PIL Image of a leaf
            threshold: Minimum lesion confidence to highlight
            
        Returns:
            PIL Image with lesions highlighted (red overlay)
        """
        try:
            img_array = np.array(image.convert('RGB'))
            
            # Detect lesions
            _, lesion_mask = self._detect_lesions(img_array)
            
            # Create overlay
            overlay = img_array.copy()
            red_overlay = np.zeros_like(img_array)
            red_overlay[:, :, 0] = 255  # Red channel
            
            # Apply red overlay where lesions are detected
            lesion_regions = lesion_mask > (threshold * 255)
            overlay[lesion_regions] = cv2.addWeighted(
                overlay[lesion_regions], 0.7,
                red_overlay[lesion_regions], 0.3,
                0
            )
            
            return Image.fromarray(overlay)
            
        except Exception as e:
            print(f"❌ Error highlighting lesions: {e}")
            return image
    
    def _segment_leaf(self, img_array: np.ndarray) -> np.ndarray:
        """
        Segment leaf from background using GrabCut.
        
        Args:
            img_array: RGB image array
            
        Returns:
            Segmented image with background removed
        """
        try:
            height, width = img_array.shape[:2]
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Create initial rectangle (with 5% margin)
            margin = 0.05
            x = int(width * margin)
            y = int(height * margin)
            rect = (x, y, width - 2*x, height - 2*y)
            
            # Initialize mask and models
            mask = np.zeros((height, width), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask (foreground = leaf)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Apply mask to image
            segmented = img_array * mask2[:, :, np.newaxis]
            
            return segmented
            
        except Exception as e:
            print(f"⚠️  Error in leaf segmentation: {e}")
            # Return original if segmentation fails
            return img_array
    
    def _calculate_green_percentage(self, img_array: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculate percentage of healthy green pixels.
        
        Args:
            img_array: RGB image array
            
        Returns:
            Tuple of (green_percentage, green_mask)
        """
        try:
            # Remove black pixels (background)
            non_black = ~np.all(img_array == [0, 0, 0], axis=2)
            
            if non_black.sum() == 0:
                return 0.0, np.zeros(img_array.shape[:2], dtype=np.uint8)
            
            # Convert to HSV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            # Create mask for green pixels
            green_mask = cv2.inRange(hsv, self.healthy_green_lower, self.healthy_green_upper)
            
            # Only count green pixels in non-background areas
            green_in_leaf = green_mask & (non_black.astype(np.uint8) * 255)
            
            # Calculate percentage
            total_pixels = non_black.sum()
            green_pixels = np.count_nonzero(green_in_leaf)
            green_percentage = (green_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
            
            return green_percentage, green_mask
            
        except Exception as e:
            print(f"⚠️  Error calculating green percentage: {e}")
            return 0.0, np.zeros(img_array.shape[:2], dtype=np.uint8)
    
    def _detect_lesions(self, img_array: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Detect potential lesion areas (yellowing, browning, spots).
        
        Args:
            img_array: RGB image array
            
        Returns:
            Tuple of (lesion_percentage, lesion_mask)
        """
        try:
            # Remove black pixels (background)
            non_black = ~np.all(img_array == [0, 0, 0], axis=2)
            
            if non_black.sum() == 0:
                return 0.0, np.zeros(img_array.shape[:2], dtype=np.uint8)
            
            # Convert to HSV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            # Detect yellowing/browning (common in diseased leaves)
            yellow_mask = cv2.inRange(hsv, self.lesion_yellow_lower, self.lesion_yellow_upper)
            brown_mask = cv2.inRange(hsv, self.lesion_brown_lower, self.lesion_brown_upper)
            
            # Combine masks
            lesion_mask = cv2.bitwise_or(yellow_mask, brown_mask)
            
            # Only count lesions in non-background areas
            lesion_in_leaf = lesion_mask & (non_black.astype(np.uint8) * 255)
            
            # Calculate percentage
            total_pixels = non_black.sum()
            lesion_pixels = np.count_nonzero(lesion_in_leaf)
            lesion_percentage = (lesion_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
            
            return lesion_percentage, lesion_mask
            
        except Exception as e:
            print(f"⚠️  Error detecting lesions: {e}")
            return 0.0, np.zeros(img_array.shape[:2], dtype=np.uint8)
    
    def _find_lesion_regions(self, lesion_mask: np.ndarray, min_area: int = 50) -> List[Dict]:
        """
        Find connected regions of lesions.
        
        Args:
            lesion_mask: Binary mask of lesion areas
            min_area: Minimum area (in pixels) for a region to be considered
            
        Returns:
            List of dictionaries with region info:
            - 'bbox': (x1, y1, x2, y2) bounding box
            - 'area': Area in pixels
            - 'centroid': (x, y) center point
        """
        try:
            # Find contours
            contours, _ = cv2.findContours(
                lesion_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area >= min_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Get centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w // 2, y + h // 2
                    
                    regions.append({
                        'bbox': (x, y, x + w, y + h),
                        'area': int(area),
                        'centroid': (cx, cy)
                    })
            
            return regions
            
        except Exception as e:
            print(f"⚠️  Error finding lesion regions: {e}")
            return []


# Global instance
_lesion_analyzer = None

def get_lesion_analyzer() -> LesionAnalyzer:
    """Get or create global lesion analyzer instance."""
    global _lesion_analyzer
    if _lesion_analyzer is None:
        _lesion_analyzer = LesionAnalyzer()
    return _lesion_analyzer

