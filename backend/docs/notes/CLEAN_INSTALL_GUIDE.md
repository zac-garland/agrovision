# Clean Virtual Environment Setup Guide

## Problem
**NumPy 2.x compatibility issue** - PyTorch was compiled against NumPy 1.x but your environment has NumPy 2.2.6 installed.

Error message:
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
```

## ✅ Solution: Clean Install

### Step 1: Deactivate Current Environment
```bash
deactivate
```

### Step 2: Remove Old Virtual Environment (Optional but Recommended)
```bash
cd /Users/zacgarland/r_projects/agrovision
rm -rf venv
# OR if using datascience_env:
# rm -rf datascience_env
```

### Step 3: Create Fresh Virtual Environment
```bash
cd /Users/zacgarland/r_projects/agrovision
python3 -m venv venv
# OR for datascience_env:
# python3 -m venv datascience_env
```

### Step 4: Activate New Environment
```bash
source venv/bin/activate
# OR
# source datascience_env/bin/activate
```

### Step 5: Upgrade pip (Important!)
```bash
pip install --upgrade pip setuptools wheel
```

### Step 6: Install Requirements
```bash
cd backend
pip install -r requirements.txt
```

### Step 7: Verify Versions
```bash
python -c "import numpy; import torch; print(f'NumPy: {numpy.__version__}'); print(f'PyTorch: {torch.__version__}')"
```

**Expected output:**
- NumPy: 1.26.x (NOT 2.x)
- PyTorch: 2.2.0

### Step 8: Test Phase 4
```bash
python tests/test_phase4.py
```

## 🔧 Quick Fix (If You Don't Want to Recreate Venv)

If you want to try fixing the current environment first:

```bash
pip install "numpy<2.0,>=1.24.0" --force-reinstall
pip install "torch==2.2.0" --force-reinstall
```

**However**, a clean venv is **highly recommended** for best results and to avoid other hidden conflicts.

## 📝 Requirements.txt Updated

I've updated `requirements.txt` to enforce NumPy < 2.0:
```
numpy<2.0,>=1.24.0  # Must be < 2.0 for PyTorch compatibility
```

This prevents NumPy 2.x from being installed in the future.

## 💡 Why Clean Install?

1. **Avoids dependency conflicts** - Fresh start ensures compatibility
2. **Cleaner environment** - No leftover packages causing issues
3. **Guaranteed versions** - Exactly what's in requirements.txt
4. **Faster than debugging** - Less time troubleshooting conflicts

## 🎯 Recommended Approach

**Yes, create a clean venv!** It's the most reliable solution for this type of compatibility issue.
