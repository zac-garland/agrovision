# 🚀 Step-by-Step Deployment Guide: Railway + HF Spaces

**Do this on Day 3 morning, after your code is working locally.**

**Time: ~1.5 hours | Cost: $5-10/month**

---

## Prerequisites

You'll need:
- ✅ Working Flask + Streamlit app (locally functional)
- ✅ Git installed
- ✅ GitHub account (free)
- ✅ Internet connection

---

## Architecture

```
Your GitHub Repo
    │
    ├──→ Railway (Backend)
    │    ├── Flask app
    │    ├── Ollama + Mistral
    │    └── Models (5.5GB)
    │    Result: https://agrovision-backend.railway.app/identify
    │
    └──→ Hugging Face Spaces (Frontend)
         ├── Streamlit app
         └── Calls Railway backend
         Result: https://huggingface.co/spaces/YOUR_USERNAME/agrovision
```

---

## PHASE 1: GitHub Setup (15 minutes)

### Step 1.1: Make GitHub Repo

1. Go to https://github.com/new
2. Name: `agrovision-pivot`
3. Description: "Agentic plant diagnosis system"
4. Visibility: **Public** (needed for Railway and HF Spaces)
5. Click "Create repository"

### Step 1.2: Push Your Code

```bash
cd /Users/zacgarland/r_projects/agrovision-pivot

# Initialize git (if not already)
git init

# Add all files
git add .

# Initial commit
git commit -m "init: agentic plant diagnosis system"

# Connect to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/agrovision-pivot.git

# Push
git branch -M main
git push -u origin main
```

**Verify:** Go to GitHub and see your files uploaded

---

## PHASE 2: Deploy Backend to Railway (30 minutes)

### Step 2.1: Create Railway Account

1. Go to https://railway.app
2. Click "Start New Project"
3. Sign up with GitHub (easiest)
4. Authorize Railway to access your GitHub

### Step 2.2: Deploy from GitHub

1. Click "Deploy from GitHub"
2. Select your repository: `agrovision-pivot`
3. Click "Deploy Now"

### Step 2.3: Configure Deployment

Once deployed, Railway will show you your project.

**Add environment variables:**

1. Click "Variables" (or Environment tab)
2. Add these:
   ```
   FLASK_ENV=production
   PYTHONUNBUFFERED=1
   PORT=5000
   ```

**Check startup logs:**
1. Click "Logs"
2. Should see Flask starting
3. Might see errors (expected if Ollama isn't set up yet)

### Step 2.4: Get Your Backend URL

1. In Railway dashboard, look for "Domains"
2. You'll see something like: `agrovision-prod-xyz.railway.app`
3. Test it: `https://agrovision-prod-xyz.railway.app/identify`
4. **Copy this URL** - you'll need it for frontend

**Note:** First load will be slow (5-10 min) as it downloads and builds. This is normal.

---

## PHASE 3: Deploy Frontend to Hugging Face (30 minutes)

### Step 3.1: Create HF Spaces Repo

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Name: `agrovision` (or similar)
4. Visibility: **Public**
5. License: MIT
6. Click "Create"

### Step 3.2: Create Separate GitHub Repo for Frontend

We'll use a separate repo so HF Spaces auto-deploys:

```bash
# Create new directory
mkdir ~/agrovision-frontend-hf
cd ~/agrovision-frontend-hf

# Copy frontend files
cp /Users/zacgarland/r_projects/agrovision-pivot/frontend/app.py .
cp /Users/zacgarland/r_projects/agrovision-pivot/frontend/requirements.txt .

# Create README
cat > README.md << 'EOF'
# AgroVision+ Plant Diagnosis Frontend

Streamlit UI for plant diagnosis system.

**Backend API:** https://agrovision-prod-xyz.railway.app

See full project: https://github.com/YOUR_USERNAME/agrovision-pivot
EOF

# Initialize git
git init
git add .
git commit -m "init: frontend for HF Spaces"

# Create new GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/agrovision-frontend.git
git branch -M main
git push -u origin main
```

### Step 3.3: Update API URL in Streamlit

Before pushing to GitHub, update your Streamlit app to call the Railway backend:

**In `app.py`:**
```python
# Change this line:
# API_BASE = "http://localhost:5000"

# To this (use YOUR Railway URL):
import os
API_BASE = os.getenv('API_BASE', 'https://agrovision-prod-xyz.railway.app')
# Replace agrovision-prod-xyz with your actual Railway URL
```

**Then commit and push:**
```bash
git add app.py
git commit -m "chore: update backend API URL for production"
git push
```

### Step 3.4: Connect HF Spaces to GitHub

Back in HF Spaces:

1. In your Space, click "Settings"
2. Scroll to "Repository"
3. Click "Link a repository"
4. Select: `YOUR_USERNAME/agrovision-frontend`
5. HF will auto-deploy from GitHub

**Wait 2-5 minutes for Streamlit to build and start**

---

## PHASE 4: Testing (15 minutes)

### Step 4.1: Test Backend API

```bash
# Test the /identify endpoint
curl -X POST https://agrovision-prod-xyz.railway.app/identify \
  -F "image=@/Users/zacgarland/r_projects/agrovision-pivot/test-image.jpeg"

# Should return JSON with plant identification
```

### Step 4.2: Test Frontend UI

1. Go to: `https://huggingface.co/spaces/YOUR_USERNAME/agrovision`
2. Upload a test image
3. Click "Identify Plant"
4. Should show plant species and confidence
5. Go through full flow (questions → diagnosis)

### Step 4.3: Check Logs for Errors

**Railway logs:**
1. Go to Railway dashboard
2. Click "Logs"
3. Look for errors or issues

**HF Spaces logs:**
1. Go to HF Space
2. Click "... → View logs"
3. Check for Python errors

---

## COMMON ISSUES & FIXES

### Issue 1: Railway Build Takes Too Long

**Symptom:** Deployment stuck for 10+ minutes

**Cause:** Installing PyTorch for first time (takes a while)

**Fix:** This is normal. Wait up to 20 minutes for first deployment.

---

### Issue 2: Ollama Not Found

**Symptom:** Error: "ollama: command not found"

**Cause:** Ollama not pre-installed in Railway

**Fix:** Add startup script to install Ollama

Create `railway.sh` in root:
```bash
#!/bin/bash
set -e

echo "Installing Ollama..."
curl https://ollama.ai/install.sh | sh

echo "Pulling Mistral model..."
ollama pull mistral &

echo "Starting Flask..."
cd backend
python main.py
```

Then update `Procfile`:
```
web: bash railway.sh
```

---

### Issue 3: API Returns 404

**Symptom:** Frontend gets 404 from backend

**Causes:**
- Railway URL is wrong
- Backend not started yet
- Flask routes not defined

**Fix:**
1. Check Railway URL is correct in `app.py`
2. Check Railway logs for startup errors
3. Verify Flask endpoints are implemented

---

### Issue 4: Frontend Can't Connect to Backend

**Symptom:** "Connection refused" or timeout

**Causes:**
- CORS error
- Backend URL wrong
- Backend not running

**Fix:**
1. Add CORS to Flask:
```python
from flask_cors import CORS
CORS(app)
```

2. Update `backend/requirements.txt`:
```
flask-cors==3.0.10
```

3. Verify backend URL is accessible in browser

---

### Issue 5: Models Download Timeout

**Symptom:** Deployment times out downloading models

**Cause:** Models are 5.5GB, takes time

**Fix:**
- Pre-download models before deployment
- Or use smaller models
- Or increase Railway timeout

---

## AFTER DEPLOYMENT

### Share Your Links

Now you have two public URLs:

1. **Backend API:**
   ```
   https://agrovision-prod-xyz.railway.app
   ```

2. **Frontend (Streamlit):**
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/agrovision
   ```

### Share with Professor/Class

Send the Streamlit link:
```
Check out my plant diagnosis AI!
https://huggingface.co/spaces/YOUR_USERNAME/agrovision

Try uploading a plant photo and let me know what it diagnoses.
```

### Update Your README

In your GitHub repo, add:

```markdown
## 🚀 Live Demo

Try it here: https://huggingface.co/spaces/YOUR_USERNAME/agrovision

Backend API: https://agrovision-prod-xyz.railway.app
```

---

## CONTINUOUS DEPLOYMENT

After initial deployment, updates are automatic:

```bash
# Make changes locally
cd /Users/zacgarland/r_projects/agrovision-pivot
# ... edit code ...

# Commit and push
git add .
git commit -m "feat: improve plant identification accuracy"
git push

# Railway automatically redeploys from main
# HF Spaces auto-syncs 30 seconds later
# Your live app is updated!
```

---

## SCALING UP (Optional)

If you need faster inference later:

### Option A: Railway GPU

In Railway dashboard:
1. Go to your project
2. Click on service
3. Click "Compute"
4. Select GPU instance (T4 or better)
5. Cost: $50-100/month

### Option B: Reduce Model Size

Use ResNet18 instead of ResNet152:
- Faster inference (2-5 seconds vs 10-30)
- Uses less memory
- Trade-off: Slightly lower accuracy

Update `plant_identifier.py`:
```python
# Use ResNet18 instead of ResNet152
model = resnet18(pretrained=False)
```

---

## MONITORING & MAINTENANCE

### Check Status

Visit your Railway and HF dashboards regularly:

**Railway:** https://railway.app/dashboard
**HF Spaces:** https://huggingface.co/spaces

### Monitor Costs

**Railway:**
- Free tier: $5/month credits
- After that: ~$0.12/hour (CPU only) = ~$90/month 24/7
- Don't worry, it's very cheap

### Keep Models Fresh

Models don't auto-update. If you want newer versions:
1. Pull latest on local
2. Push to railway
3. Railway rebuilds

---

## Troubleshooting Checklist

Before asking for help, check:

- [ ] Code works locally? (Flask + Streamlit)
- [ ] GitHub repo is public?
- [ ] Both Railway and HF have access to GitHub?
- [ ] Flask routes match API_CONTRACT.md?
- [ ] Backend URL in Streamlit is correct?
- [ ] All dependencies in requirements.txt?
- [ ] Checked logs for errors?
- [ ] Tried refreshing the browser?

---

## Success Criteria

✅ You're done when:

- [ ] Railway deployment shows green "Build successful"
- [ ] Backend API returns 200 on test request
- [ ] HF Spaces shows "Running" status
- [ ] Frontend loads in browser
- [ ] Full flow works (upload → diagnosis → results)
- [ ] Public URLs are shareable

---

## Timeline

```
Day 3, 9:00 AM:
├─ 9:00-9:15   Phase 1: GitHub setup
├─ 9:15-9:45   Phase 2: Railway deployment
├─ 9:45-10:15  Phase 3: HF Spaces deployment
├─ 10:15-10:30 Phase 4: Testing & fixes
└─ 10:30       Live demo ready! 🎉
```

---

## Questions During Deployment?

1. **Railway not deploying?**
   - Check GitHub repo is public
   - Check logs for build errors
   - Verify Procfile exists

2. **HF Spaces won't start?**
   - Check requirements.txt is valid
   - Check app.py has no syntax errors
   - Check API_BASE URL is correct

3. **API calls failing?**
   - Check backend URL in code
   - Test URL in browser directly
   - Check CORS configuration
   - Check Flask routes are implemented

4. **Models not loading?**
   - Check Railway logs
   - Models might still be downloading (wait 10 min)
   - Verify model paths are correct

---

## You've Got This!

Once you've done this once, it becomes second nature. Now you understand:
- How to deploy Python apps
- How to separate frontend/backend
- How cloud platforms work
- How to iterate on live code

Pretty cool for a student project! 🌱

---

## Next Steps After Deployment

1. **Demo to professor** with live URL
2. **Add to portfolio** with links
3. **Write deployment blog post** (great for learning)
4. **Explore scaling** (GPU, caching, etc.)

Good luck! 🚀
