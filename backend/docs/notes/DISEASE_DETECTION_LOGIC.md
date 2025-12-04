# Current Disease Detection Logic

## Overview

Our disease detection system uses **image processing and color analysis** rather than a trained disease classification model. It's designed to work with **any plant species** without requiring a specific disease dataset.

## Architecture

```
Image → Leaf Detection (YOLO) → Lesion Analysis (Color/Image Processing) → Health Scoring → Diagnosis Synthesis
```

## Step-by-Step Process

### 1. **Leaf Detection** (`services/leaf_detector.py`)
- **Model**: YOLO11x (custom-trained for leaf detection)
- **Purpose**: Isolate individual leaves from the plant image
- **Output**: 
  - Cropped leaf images
  - Bounding boxes for each leaf
  - Detection confidence scores

**Key Logic:**
- Uses YOLO object detection with confidence threshold: `0.15`
- Detects multiple leaves in a single image
- Crops each detected leaf for individual analysis

### 2. **Lesion Analysis** (`services/lesion_analyzer.py`)

#### A. **Leaf Segmentation**
- Uses **GrabCut algorithm** to segment leaf from background
- Improves accuracy by focusing analysis on leaf area only

#### B. **Color Analysis - Healthy Green Detection**
- **Method**: HSV color space analysis
- **Healthy green range**: 
  - Hue: 25-95
  - Saturation: 30-255
  - Value: 30-255
- **Output**: Percentage of healthy green pixels in the leaf

#### C. **Lesion Detection**
- **Method**: HSV color space analysis for unhealthy colors
- **Detects**:
  - **Yellowing** (common in diseased leaves)
    - HSV Range: [10-25, 50-255, 50-255]
  - **Browning** (dead/damaged tissue)
    - HSV Range: [5-25, 50-255, 30-200]
- **Output**: Percentage of lesion-like pixels

#### D. **Lesion Region Identification**
- Uses **contour detection** (OpenCV) to find connected lesion regions
- Minimum area filter: 50 pixels (removes noise)
- Returns bounding boxes for each lesion region

### 3. **Health Score Calculation**

```python
health_score = max(0.0, min(1.0, green_percentage / 100.0 - lesion_percentage / 100.0))
```

- **Range**: 0.0 (very unhealthy) to 1.0 (perfectly healthy)
- Formula balances:
  - High green percentage = healthy
  - Low lesion percentage = healthy

### 4. **Issue Detection Logic**

A leaf is flagged as having **potential issues** if:
```python
has_potential_issues = (
    lesion_percentage > 5.0 OR  # More than 5% lesion-like areas
    green_percentage < 50.0      # Less than 50% healthy green
)
```

### 5. **Severity Assessment** (`services/diagnosis_engine.py`)

Based on overall health score (average of all leaves):

| Health Score | Severity | Condition |
|--------------|----------|-----------|
| ≥ 0.8 | `none` | Plant appears healthy |
| 0.6 - 0.8 | `low` | Minor health concerns detected |
| 0.4 - 0.6 | `moderate` | Moderate health issues detected |
| < 0.4 | `high` | Significant health problems detected |

### 6. **Diagnosis Synthesis**

#### Option A: **LLM-Powered** (if Ollama/Mistral available)
- Uses Mistral 7B to generate intelligent reasoning
- Considers:
  - Plant species identification
  - Health scores
  - Lesion percentages
  - Number of affected leaves
- Generates natural language diagnosis and treatment recommendations

#### Option B: **Rule-Based** (fallback)
- Deterministic logic based on health scores
- Pre-defined treatment plans by severity level
- Simple, reliable, always available

## Current Limitations

### ❌ **No Specific Disease Names**
- System detects **"unhealthy" vs "healthy"**
- Does NOT identify specific diseases (e.g., "powdery mildew", "rust", "blight")
- General health assessment only

### ❌ **Color-Based Detection Only**
- Relies on color changes (yellowing, browning)
- May miss:
  - Texture-based diseases
  - Diseases that don't change color significantly
  - Early-stage diseases

### ❌ **No Plant-Specific Knowledge**
- Same logic applies to all plants
- Doesn't account for:
  - Natural color variations between species
  - Species-specific disease patterns
  - Seasonal color changes

### ✅ **Advantages**
- Works with any plant (no dataset required)
- Fast and lightweight
- No training needed
- General health assessment is still useful

## Parameters & Thresholds

### Current Settings

| Parameter | Value | Location |
|-----------|-------|----------|
| Leaf detection confidence | 0.15 | `routes/diagnose.py` |
| Lesion threshold | 5.0% | `lesion_analyzer.py` |
| Green threshold | 50.0% | `lesion_analyzer.py` |
| Minimum lesion area | 50 pixels | `lesion_analyzer.py` |
| Health score thresholds | 0.4, 0.6, 0.8 | `diagnosis_engine.py` |

### HSV Color Ranges

**Healthy Green:**
- Lower: `[25, 30, 30]`
- Upper: `[95, 255, 255]`

**Yellowing (Lesions):**
- Lower: `[10, 50, 50]`
- Upper: `[25, 255, 255]`

**Browning (Lesions):**
- Lower: `[5, 50, 30]`
- Upper: `[25, 255, 200]`

## Output Structure

### Leaf Analysis Data
```python
{
    'health_score': 0.96,           # 0-1 scale
    'green_percentage': 96.6,       # % of healthy green
    'lesion_percentage': 0.6,       # % of lesion areas
    'num_lesion_regions': 3,        # Number of distinct lesions
    'has_potential_issues': False,  # Boolean flag
    'lesion_areas': [               # Bounding boxes
        {
            'bbox': (x1, y1, x2, y2),
            'area': 1388,
            'centroid': (cx, cy)
        },
        ...
    ]
}
```

### Final Diagnosis
```python
{
    'condition': "Plant appears healthy",
    'severity': "none",  # none/low/moderate/high
    'confidence': 0.04,
    'reasoning': "...",
    'source': "llm"  # or "rule_based"
}
```

## Future Improvements

### Potential Enhancements

1. **Specific Disease Classification**
   - Train/use a disease classification model
   - Identify specific diseases by name
   - Plant-specific disease knowledge base

2. **Machine Learning Approach**
   - Train CNN on disease images
   - Better accuracy than color-based detection
   - Can learn texture and pattern features

3. **Plant-Specific Tuning**
   - Different thresholds per plant species
   - Species-specific healthy color ranges
   - Knowledge of natural variations

4. **Multi-Modal Analysis**
   - Combine color + texture + shape analysis
   - Time-series analysis (multiple images over time)
   - Environmental context (temperature, humidity, etc.)

5. **Advanced Lesion Detection**
   - Pattern recognition (spots, streaks, patches)
   - Edge detection for lesion boundaries
   - Deep learning segmentation models

## Summary

**Current Approach:**
- ✅ General health assessment
- ✅ Color-based lesion detection
- ✅ Works with any plant
- ✅ Fast and lightweight

**What It Does:**
- Detects unhealthy areas by color analysis
- Quantifies health with numerical scores
- Provides severity assessment
- Generates treatment recommendations

**What It Doesn't Do:**
- ❌ Identify specific disease names
- ❌ Plant-specific disease knowledge
- ❌ Texture/pattern-based detection
- ❌ Early-stage disease detection

The system is designed as a **general health monitor** rather than a **specific disease classifier**. It's useful for identifying when a plant needs attention, but not for diagnosing the exact disease.

