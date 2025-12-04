# Quick Testing Guide

## Fastest Way to Test Everything

### Option 1: Automated Test (Recommended) ⚡

**Terminal 1 - Start the server:**
```bash
cd /Users/zacgarland/r_projects/agrovision/backend
source ../venv/bin/activate
python app.py
```

**Terminal 2 - Run all tests:**
```bash
cd /Users/zacgarland/r_projects/agrovision/backend
source ../venv/bin/activate
python tests/test_all.py
```

This will automatically test:
- ✅ Health endpoint
- ✅ Model loading
- ✅ Model inference
- ✅ Diagnosis endpoint

### Option 2: Manual Testing (Step by Step)

#### 1. Start the Server
```bash
cd /Users/zacgarland/r_projects/agrovision/backend
source ../venv/bin/activate
python app.py
```

#### 2. Test Health (New Terminal)
```bash
curl http://localhost:5000/health
```

**Expected:** `{"status": "healthy", "service": "AgroVision+ Backend", "version": "1.0.0"}`

#### 3. Test Model Loading (New Terminal)
```bash
cd /Users/zacgarland/r_projects/agrovision/backend
source ../venv/bin/activate
python tests/test_plantnet.py
```

#### 4. Test Diagnosis Endpoint (Same Terminal)
```bash
python tests/test_endpoint.py
```

## Quick Verification

Run these commands to verify everything is set up:

```bash
# Check if server is running
curl http://localhost:5000/health

# Check if test image exists
ls backend/static/test-image.jpeg

# Check if models exist
ls models/*.tar
```

## Troubleshooting

**"Connection refused"** → Start Flask server first  
**"Module not found"** → Activate virtual environment  
**"Image not found"** → Check `backend/static/test-image.jpeg` exists  

For detailed testing, see `TESTING_GUIDE.md`

