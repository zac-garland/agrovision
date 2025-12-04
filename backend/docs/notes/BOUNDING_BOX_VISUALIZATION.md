# Bounding Box Visualization Feature

## ✅ Implementation Complete

The frontend now displays bounding boxes around detected leaves and lesions on the uploaded image.

## What Was Added

### Backend Changes

1. **Added bounding box data to response** (`backend/routes/diagnose.py`):
   - Leaf bounding boxes: `disease_detection.leaf_analysis.leaf_boxes`
   - Individual leaf data with bounding boxes: `individual_leaves[].leaf_box`
   - Lesion bounding boxes: `individual_leaves[].lesion_areas[]` (each with `bbox`)

### Frontend Changes

1. **Added visualization function** (`frontend/app.py`):
   - `draw_bounding_boxes()` - Draws boxes on the image
   - Green boxes for detected leaves
   - Red boxes for detected lesions
   - Handles coordinate transformation for lesions (from leaf space to original image space)

2. **Display annotated image**:
   - Shows the original image with bounding boxes overlaid
   - Displayed in the results section after diagnosis
   - Caption explains: "green boxes = leaves, red boxes = lesions"

## How It Works

### Leaf Detection
- YOLO model detects leaves in the image
- Returns bounding boxes: `(x1, y1, x2, y2)` in pixel coordinates
- Green boxes drawn around each detected leaf

### Lesion Detection
- Each detected leaf is analyzed for lesions
- Lesion bounding boxes are relative to the leaf (cropped leaf coordinates)
- Coordinates are transformed to original image space for display
- Red boxes drawn around detected lesion regions

## Visual Output

```
┌─────────────────────────────┐
│  Original Image             │
│                             │
│  ┌──────┐                   │
│  │ Leaf │ ← Green box       │
│  │  ┌─┐ │                   │
│  │  │L│ │ ← Red box (lesion)│
│  │  └─┘ │                   │
│  └──────┘                   │
│                             │
└─────────────────────────────┘
```

## Usage

1. Upload an image through the frontend
2. Click "run diagnosis"
3. After processing, an annotated image appears showing:
   - **Green boxes**: Detected leaves
   - **Red boxes**: Detected lesions within leaves

## Technical Details

### Coordinate Transformation

Lesion coordinates are detected on cropped leaf images, so they need to be transformed back to the original image coordinates:

```python
# Lesion box in leaf space: (lx1, ly1, lx2, ly2)
# Leaf box in original image: (leaf_x1, leaf_y1, leaf_x2, leaf_y2)

# Transform to original image space:
abs_x1 = leaf_x1 + lx1
abs_y1 = leaf_y1 + ly1
abs_x2 = leaf_x1 + lx2
abs_y2 = leaf_y1 + ly2
```

### Colors

- **Green**: Healthy leaves (detected by YOLO)
- **Red**: Lesions (disease areas detected within leaves)

## Future Enhancements

- [ ] Add toggle to show/hide boxes
- [ ] Different colors for different severity levels
- [ ] Confidence scores on boxes
- [ ] Click boxes to see detailed analysis
- [ ] Export annotated image

