# Phase 5 Testing Guide

## ✅ Current Status

Based on quick tests:
- ✅ Ollama CLI installed (version 0.12.0)
- ✅ Ollama service running
- ✅ Mistral model available
- ❌ Ollama Python package needs installation

## 🚀 Quick Setup

### Step 1: Install Ollama Python Package

**Option A: If using virtual environment:**
```bash
cd /Users/zacgarland/r_projects/agrovision
source venv/bin/activate  # or your venv name
cd backend
pip install ollama
```

**Option B: If using system Python:**
```bash
pip install ollama
```

### Step 2: Verify Setup

Run the quick test:
```bash
cd backend
python tests/test_phase5_quick.py
```

Should show all ✅ checks passing!

### Step 3: Test LLM Integration

**Test with endpoint (requires Flask server):**

```bash
# Terminal 1: Start Flask server
cd backend
python app.py

# Terminal 2: Test endpoint
python test_image.py static/test-image2.jpeg
```

The endpoint will automatically:
- Use LLM if available ✅
- Fall back to rule-based if LLM unavailable ✅

## 🧪 Testing Options

### Option 1: Quick Setup Test
```bash
python tests/test_phase5_quick.py
```

### Option 2: Full Phase 5 Test (after installing ollama)
```bash
python tests/test_phase5.py
```

### Option 3: Test Endpoint Directly
```bash
# Make sure Flask server is running first
curl -X POST \
  -F "image=@static/test-image2.jpeg" \
  http://127.0.0.1:5000/diagnose | python -m json.tool
```

## 📝 What to Expect

### With LLM (Phase 5):
- Intelligent reasoning text
- Plant-specific recommendations
- Detailed treatment plans
- Source: "llm"

### Without LLM (Fallback):
- Rule-based recommendations
- Based on health scores
- Still comprehensive
- Source: "rule_based"

## 💡 Troubleshooting

**"Ollama package not installed"**
- Solution: `pip install ollama`

**"Cannot connect to Ollama"**
- Solution: `ollama serve` (in separate terminal)

**"Mistral model not found"**
- Solution: `ollama pull mistral`

**"Module not found" errors**
- Solution: Make sure you're in the backend directory
- Or activate virtual environment first

## ✅ Ready to Test!

After installing `ollama` Python package, everything should work!

