# Supplementary Resources Analysis

Analysis of resources in the `supplementary/` folder for Phases 4-6.

## 📦 Available Resources

### 1. **YOLO Leaf Detection Model** (`yolo11x_leaf.pt`)
- **Type:** YOLO v11x model (PyTorch/Ultralytics format)
- **Purpose:** Leaf detection and isolation from images
- **Use Case:** Phase 4 - Pre-processing step to isolate leaves before disease detection
- **Status:** ✅ Ready to use

**Code Example from `agrovision_ 1.py`:**
```python
from ultralytics import YOLO
model = YOLO('yolo11x_leaf.pt')
result = model.predict(image_path, task="detect", save=False, conf=0.15)
# Crop detected leaves from bounding boxes
```

### 2. **ResNet Disease Detection Model** (`my_resnet_model.keras`)
- **Type:** TensorFlow/Keras ResNet-18 model
- **Purpose:** Disease classification on isolated leaf images
- **Classes:** 38 disease classes (see class list below)
- **Use Case:** Phase 4 - Primary disease detection model
- **Status:** ✅ Ready to use (may need TensorFlow conversion or wrapper)

**Model Architecture:**
- ResNet-18 with custom implementation
- 38 output classes
- Input size: 224x224
- Uses dropout (0.3) for regularization

**Code Example from `agrovision_ 1.py`:**
```python
import tensorflow as tf
model = ResNet(resnet18_arch, num_classes=38, dropout_rate=0.3)
model.load_weights('my_resnet_model.keras')
prediction = model.predict(preprocessed_image)
```

### 3. **AML Recommendation Engine** (`AML_Recommendation_Engine.ipynb`)
- **Type:** Rule-based recommendation system
- **Purpose:** Generate care recommendations from disease + weather data
- **Use Case:** Phase 5 - Could replace or complement LLM synthesis
- **Status:** ✅ Ready to adapt

**Features:**
- Maps 38 disease classes to 5 categories (fungal, bacterial, viral, pest, healthy)
- Weather-aware recommendations using OpenWeather API
- Simple, readable rule-based logic

### 4. **Reference Implementation** (`agrovision_ 1.py`)
- Complete pipeline example showing:
  - YOLO leaf detection
  - Leaf isolation/cropping
  - ResNet disease prediction
  - Image segmentation (GrabCut)
  - Green percentage calculation
  - Weather API integration

## 🎯 Disease Classes (38 Total)

From `AML_Recommendation_Engine.ipynb`:

1. Apple___Apple_scab
2. Apple___Black_rot
3. Apple___Cedar_apple_rust
4. Apple___healthy
5. Blueberry___healthy
6. Cherry_(including_sour)___Powdery_mildew
7. Cherry_(including_sour)___healthy
8. Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
9. Corn_(maize)___Common_rust_
10. Corn_(maize)___Northern_Leaf_Blight
11. Corn_(maize)___healthy
12. Grape___Black_rot
13. Grape___Esca_(Black_Measles)
14. Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
15. Grape___healthy
16. Orange___Haunglongbing_(Citrus_greening)
17. Peach___Bacterial_spot
18. Peach___healthy
19. Pepper,_bell___Bacterial_spot
20. Pepper,_bell___healthy
21. Potato___Early_blight
22. Potato___Late_blight
23. Potato___healthy
24. Raspberry___healthy
25. Soybean___healthy
26. Squash___Powdery_mildew
27. Strawberry___Leaf_scorch
28. Strawberry___healthy
29. Tomato___Bacterial_spot
30. Tomato___Early_blight
31. Tomato___Late_blight
32. Tomato___Leaf_Mold
33. Tomato___Septoria_leaf_spot
34. Tomato___Spider_mites Two-spotted_spider_mite
35. Tomato___Target_Spot
36. Tomato___Tomato_Yellow_Leaf_Curl_Virus
37. Tomato___Tomato_mosaic_virus
38. Tomato___healthy

**Disease Categories:**
- Fungal (most common)
- Bacterial
- Viral
- Pest
- Healthy

## 🔧 Integration Plan

### Phase 4: Disease Detection Integration

#### Option A: Use Existing ResNet Model (Recommended)
1. Create `backend/models/disease_model.py` wrapper for Keras model
2. Load `my_resnet_model.keras` 
3. Create class label mapping (38 classes)
4. Integrate into `/diagnose` endpoint

**Pros:**
- Model already trained and ready
- 38 classes covers common diseases
- Good accuracy likely

**Cons:**
- TensorFlow dependency (we're using PyTorch for PlantNet)
- May need to convert or run in separate process

#### Option B: Use YOLO for Leaf Isolation + ResNet for Disease
1. Add YOLO leaf detection step before disease detection
2. Crop isolated leaves
3. Run ResNet on each cropped leaf
4. Aggregate results

**Pros:**
- Better accuracy on complex images
- Handles multiple leaves
- More robust

**Cons:**
- More complex pipeline
- Slower processing

### Phase 5: Recommendation/LLM Synthesis

#### Option A: Rule-Based System (Faster, More Reliable)
- Use `AML_Recommendation_Engine.ipynb` logic
- Simple, deterministic recommendations
- No LLM dependency
- Weather-aware

#### Option B: LLM Synthesis (More Flexible)
- Use Mistral 7B as planned
- Can combine PlantNet + Disease + Weather data
- More flexible reasoning
- Requires LLM setup

#### Option C: Hybrid Approach (Best of Both)
- Use rule-based for standard cases
- Use LLM for complex/unusual cases
- Fallback to rules if LLM unavailable

## 📋 Implementation Steps

### Step 1: Disease Model Wrapper
```python
# backend/models/disease_model.py
class DiseaseModel:
    def __init__(self):
        self.model = tf.keras.models.load_model('my_resnet_model.keras')
        self.class_labels = [...]  # 38 classes
        self.disease_categories = {...}  # from AML notebook
    
    def predict(self, image_pil):
        # Preprocess PIL image
        # Run inference
        # Return predictions with categories
```

### Step 2: YOLO Leaf Detection (Optional)
```python
# backend/services/leaf_detector.py
class LeafDetector:
    def __init__(self):
        self.model = YOLO('yolo11x_leaf.pt')
    
    def isolate_leaves(self, image_pil):
        # Detect leaves
        # Crop bounding boxes
        # Return list of leaf images
```

### Step 3: Recommendation Engine
```python
# backend/services/recommendation_engine.py
# Port from AML_Recommendation_Engine.ipynb
def make_recommendation(disease_label, weather_description):
    # Map disease to category
    # Get base recommendations
    # Add weather adjustments
    # Return care tips
```

## 🔍 Files to Examine

1. **`supplementary/agrovision_ 1.py`**
   - Complete pipeline example
   - Image preprocessing code
   - Model loading examples

2. **`supplementary/AML_Recommendation_Engine.ipynb`**
   - Rule-based recommendation logic
   - Disease category mapping
   - Weather-aware adjustments

3. **Model Files:**
   - `supplementary/yolo11x_leaf.pt` - YOLO model
   - `supplementary/my_resnet_model.keras` - Disease model

## ✅ Next Actions

1. **Examine model files** - Verify compatibility
2. **Create disease model wrapper** - Phase 4 implementation
3. **Port recommendation engine** - Phase 5 implementation
4. **Test integration** - End-to-end testing
5. **Consider dependencies** - TensorFlow vs PyTorch

## 📝 Notes

- The ResNet model uses TensorFlow/Keras (we're using PyTorch for PlantNet)
- May need to decide: Keep both frameworks or convert one
- YOLO model requires `ultralytics` package
- Recommendation engine is lightweight and easy to port
- Weather API key in code (should be moved to config/env)

## 🚀 Recommendations

1. **Start with ResNet disease model** - It's ready to use
2. **Add YOLO later** - As optimization for complex images
3. **Use rule-based recommendations first** - Then add LLM for flexibility
4. **Create wrapper classes** - Similar to PlantNetModel structure

