# Testing Guide for AgroVision+ Backend

This guide will help you test all the currently implemented features.

## Prerequisites

1. **Activate your virtual environment:**
```bash
cd /Users/zacgarland/r_projects/agrovision
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

2. **Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

3. **Verify test image exists:**
```bash
ls backend/static/test-image.jpeg
```

## Testing Methods

### Method 1: Quick Test Script (Recommended)

I've created a comprehensive test script that tests everything automatically.

```bash
cd backend
python tests/test_all.py
```

This will test:
- ✅ Health endpoint
- ✅ PlantNet model loading
- ✅ Diagnosis endpoint

### Method 2: Manual Step-by-Step Testing

#### Step 1: Test Health Endpoint

**Terminal 1 - Start the server:**
```bash
cd backend
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

**Terminal 2 - Test health:**
```bash
curl http://localhost:5000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "service": "AgroVision+ Backend",
  "version": "1.0.0"
}
```

#### Step 2: Test PlantNet Model Directly

**Terminal 1 - Keep server running, or start new terminal:**
```bash
cd backend
python tests/test_plantnet.py
```

**Expected output:**
- Model loading messages
- Image loading confirmation
- Top 5 predictions with confidence scores

**Note:** First run will take longer as the model loads (10-30 seconds depending on your system).

#### Step 3: Test Diagnosis Endpoint

**Terminal 1 - Make sure server is running:**
```bash
cd backend
python app.py
```

**Terminal 2 - Test endpoint:**
```bash
cd backend
python tests/test_endpoint.py
```

**Expected response:**
```json
{
  "success": true,
  "diagnosis": {
    "plant_species": {
      "primary": {...},
      "top_5": [...]
    },
    "disease_detection": {
      "primary": null,
      "all_diseases": {}
    },
    "final_diagnosis": {...},
    "treatment_plan": {...},
    "metadata": {...}
  },
  "error": null
}
```

### Method 3: Using curl (Manual API Testing)

#### Test Health Endpoint:
```bash
curl http://localhost:5000/health
```

#### Test Diagnosis Endpoint:
```bash
curl -X POST \
  http://localhost:5000/diagnose \
  -F "image=@backend/static/test-image.jpeg"
```

### Method 4: Using Python Requests (Interactive)

```python
import requests
import json

# Test health
response = requests.get("http://localhost:5000/health")
print("Health:", response.json())

# Test diagnose
with open("backend/static/test-image.jpeg", "rb") as f:
    files = {"image": f}
    response = requests.post(
        "http://localhost:5000/diagnose",
        files=files,
        timeout=60
    )
    print("Diagnosis:", json.dumps(response.json(), indent=2))
```

## Testing Checklist

- [ ] Server starts without errors
- [ ] Health endpoint returns 200 OK
- [ ] PlantNet model loads successfully
- [ ] PlantNet test script runs without errors
- [ ] Diagnosis endpoint accepts image upload
- [ ] Diagnosis endpoint returns valid JSON
- [ ] Response contains plant_species predictions
- [ ] Processing time is recorded
- [ ] Error handling works (test with invalid image)

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Make sure you're in the backend directory and virtual environment is activated:
```bash
cd backend
source ../venv/bin/activate  # Adjust path as needed
```

### Issue: "Model file not found"
**Solution:** Verify model weights exist:
```bash
ls models/resnet152_weights_best_acc.tar
ls models/resnet18_weights_best_acc.tar
```

### Issue: "Connection refused" when testing endpoint
**Solution:** Make sure Flask server is running in another terminal:
```bash
cd backend
python app.py
```

### Issue: "Test image not found"
**Solution:** Verify the test image exists:
```bash
ls backend/static/test-image.jpeg
```

If it doesn't exist, you can use any plant image:
```bash
cp /path/to/your/plant/image.jpg backend/static/test-image.jpeg
```

### Issue: Model loading takes very long or fails
**Solution:** 
- First load will be slower (10-30 seconds)
- Check available RAM (model needs ~500MB)
- Verify you're using CPU (DEVICE="cpu" in config.py)
- If you have GPU, change to DEVICE="cuda" in config.py

## Expected Performance

- **Model Loading:** 10-30 seconds (first time)
- **Health Check:** < 1 second
- **PlantNet Inference:** 1-3 seconds per image
- **Diagnosis Endpoint:** 2-5 seconds (including model loading)

## Next Steps

Once all tests pass:
1. ✅ Phase 1-3 are complete
2. 🔄 Ready for Phase 4 (Disease Detection)
3. 🔄 Ready for Phase 5 (LLM Integration)

