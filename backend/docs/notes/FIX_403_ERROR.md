# Fix 403 Error - CORS Configuration

## Problem
Frontend getting `403 Forbidden` when calling backend API.

## Solution
Added CORS (Cross-Origin Resource Sharing) support to allow frontend (port 8501) to call backend (port 5000).

## Steps to Fix

### 1. Install flask-cors
```bash
cd backend
pip install flask-cors
```

### 2. Restart Backend Server
```bash
python app.py
```

The backend now includes CORS support and will accept requests from any origin (including the Streamlit frontend).

## What Changed

1. **Added `flask-cors` to requirements.txt**
2. **Added CORS to app.py**:
   ```python
   from flask_cors import CORS
   CORS(app, resources={r"/*": {"origins": "*"}})
   ```

## Test

After restarting the backend:
1. Restart backend server
2. Try the frontend again
3. The 403 error should be gone!

## Alternative: Check Browser Console

If 403 persists, check browser console (F12) for actual error message. It might be:
- CORS preflight failure
- Network error
- Authentication issue

