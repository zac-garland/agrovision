# Hosting Strategy: AgroVision+ Plant Diagnosis System

**Status:** Planning phase
**Goal:** Understand options, decide on approach

---

## Current Architecture (Local/Development)

```
Flask Backend (port 5000)
├── PlantNet-300K inference
├── Mistral 7B via Ollama
└── JSON knowledge base

Streamlit Frontend (port 8501)
└── Calls Flask API

Local Machine:
├── Models stored in /models/
├── Ollama service running
└── 2 processes: Flask + Streamlit
```

**Current State:** Works perfectly on your laptop. Scales to... 1 user (you).

---

## Hosting Challenges

### 1. **Size**
- ResNet152 weights: ~250MB
- Mistral 7B model: ~5GB
- Total minimum: ~5.5GB just for models
- Plus: Flask + Ollama + Python runtime
- **Total image size:** ~8-10GB

### 2. **Compute**
- ResNet152 inference: 10-30 seconds on CPU
- Mistral reasoning: 5-15 seconds per response
- **Total per request:** 15-45 seconds
- CPU vs GPU: CPU only for now (Ollama can use GPU if available)

### 3. **Cost**
- GPU required for <5 second inference
- GPU hosting: $0.50-2.00 per hour
- 24/7 operation: $360-1440/month

### 4. **Serverless Not Ideal**
- AWS Lambda timeout: 15 minutes max
- Image size limit: 10GB (close!)
- Can work but inefficient
- Cold start: 30-60 seconds
- Model loading: Another 30-60 seconds
- **First request:** 1-2 minutes to respond

---

## Hosting Options Ranked

### Option 1: **Hugging Face Spaces** ✅ BEST FOR DEMO
**What:** Free hosting for ML apps  
**Pros:**
- ✅ Free tier (with limits)
- ✅ Perfect for Streamlit
- ✅ Easy deployment (git push)
- ✅ GPU available (limited)
- ✅ Can host both Flask + Streamlit

**Cons:**
- ❌ Requires public GitHub repo
- ❌ Limited CPU (shares resources)
- ❌ May timeout on inference
- ❌ Sleep mode after inactivity

**Cost:** FREE (with limitations)

**Time to deploy:** 30 minutes

**Best for:** Demo, portfolio, showing professors

---

### Option 2: **Railway** ✅ GOOD FOR SERIOUS DEMO
**What:** Simple cloud platform, pay-as-you-go  
**Pros:**
- ✅ $5/month free credits
- ✅ Easy Git integration
- ✅ Can scale horizontally
- ✅ Persistent storage for models
- ✅ Straightforward pricing

**Cons:**
- ❌ $5-30/month after free tier
- ❌ Model storage takes disk space
- ❌ Inference still slow on shared CPU
- ❌ Need to set up environment vars

**Cost:** $5-20/month (with models cached)

**Time to deploy:** 45 minutes

**Best for:** Real working demo, student project

---

### Option 3: **AWS EC2** ⚠️ ONLY IF SERIOUS
**What:** Virtual machine in the cloud  
**Pros:**
- ✅ Full control
- ✅ Can use GPU (t3.medium = $0.04/hour)
- ✅ Can keep models cached
- ✅ Predictable performance

**Cons:**
- ❌ $10-50/month minimum
- ❌ More complex setup
- ❌ Need to manage server
- ❌ SSL certificates, scaling, etc.

**Cost:** $10-50/month (CPU), $200+/month (GPU)

**Time to deploy:** 2-3 hours

**Best for:** Long-term production, serious use

---

### Option 4: **Google Colab** ⚠️ NOT RECOMMENDED
**What:** Free Jupyter notebooks with GPU  
**Pros:**
- ✅ Free (with limitations)
- ✅ GPU available
- ✅ Fast inference

**Cons:**
- ❌ Not meant for public APIs
- ❌ Sessions timeout after 12 hours
- ❌ Can't run Streamlit properly
- ❌ Not suitable for demos

**Cost:** FREE

**Time to deploy:** Not suitable

**Best for:** Development only, not hosting

---

### Option 5: **Docker + Self-Hosted** 
**What:** Package everything, run anywhere  
**Pros:**
- ✅ Full control
- ✅ Can run locally or on server
- ✅ Reproducible everywhere

**Cons:**
- ❌ You pay for server
- ❌ Need to manage infrastructure
- ❌ Complex networking setup

**Cost:** Whatever server you use ($5-500+/month)

**Time to deploy:** 1-2 hours (Docker setup)

**Best for:** Production deployments

---

## My Recommendation

### For Your Use Case (Student Project + Demo)

**Use:** Hugging Face Spaces (Frontend) + Railway (Backend)

**Why:**
- ✅ Free or very cheap
- ✅ Easy to deploy
- ✅ Works for demos
- ✅ Good learning experience
- ✅ Easy to show professors

**Cost:** $5-10/month or FREE

**Setup time:** 1-2 hours total

---

## Deployment Plan A: Hugging Face Spaces (Simplest)

### What You'd Do

1. **Create Hugging Face account** (free)
   - https://huggingface.co

2. **Create a "Space"**
   - Choose "Streamlit" template
   - Connect your GitHub repo

3. **Push code to GitHub**
   ```bash
   git push origin main
   ```

4. **Space auto-deploys from GitHub**
   - Watches your repo
   - Rebuilds on every push

### Files Needed

```
repo/
├── app.py                          # Streamlit app (only)
├── requirements.txt                # Streamlit deps
├── models/
│   ├── knowledge_base.json        # Small (~1MB)
│   └── metadata/                  # Species mappings (~5MB)
├── .gitignore
└── README.md
```

**Problem:** Can't host Flask backend in Spaces easily.

**Solution:** 
- Option A: Merge Flask into Streamlit (hack, not clean)
- Option B: Host backend elsewhere, call from Space
- Option C: Use Streamlit's built-in capabilities

---

## Deployment Plan B: Railway (Recommended)

### Architecture

```
GitHub Repo
    ↓
    ├── Railway: Backend (Flask API on Railway)
    │   ├── app/
    │   ├── main.py
    │   └── models/ (stored in Railway filesystem)
    │
    └── Hugging Face Spaces: Frontend (Streamlit)
        ├── app.py
        └── Calls Railway backend API
```

### Step-by-Step

#### Step 1: Deploy Backend to Railway

1. **Sign up at Railway**: https://railway.app/

2. **Connect GitHub**
   - Authorize Railway to access your repo

3. **Create new project**
   - Choose "Deploy from GitHub"
   - Select your repo

4. **Configure**
   - Add Python environment
   - Set up `Procfile`:
   ```
   web: python backend/main.py
   ```

5. **Deploy**
   - Railway auto-deploys
   - Get URL like: `https://agrovision-api-production.up.railway.app`

#### Step 2: Deploy Frontend to Hugging Face Spaces

1. **Sign up at Hugging Face**: https://huggingface.co/

2. **Create a Space**
   - Choose Streamlit template
   - Connect GitHub

3. **Update frontend code**
   ```python
   API_BASE = "https://agrovision-api-production.up.railway.app"
   # Instead of: API_BASE = "http://localhost:5000"
   ```

4. **Push to GitHub**
   ```bash
   git push origin main
   ```

5. **Space auto-deploys**
   - Pulls from your repo
   - Streamlit runs
   - Calls Railway backend

#### Step 3: Handle Model Storage

**Problem:** Models are 5.5GB, too big for Hugging Face.

**Solution 1: Store on Railway**
- Models go in Railway backend
- Frontend never needs them
- ✅ Cleanest approach

**Solution 2: Download on startup**
- Space downloads models from Hugging Face Hub
- Takes 2-3 minutes first time
- Cached after that

**Solution 3: Use Hugging Face Hub for models**
- Upload models to HF Hub
- Download in Space startup
- Download in Railway startup

---

## Step-by-Step Setup (Railway + HF Spaces)

### Prerequisites
- GitHub account (free)
- Railway account (free, $5/month credits)
- Hugging Face account (free)

### Phase 1: GitHub Setup (15 min)

```bash
cd /Users/zacgarland/r_projects/agrovision-pivot

# Initialize git (if not done)
git init
git add .
git commit -m "init: agentic plant diagnosis system"

# Create GitHub repo (do this on GitHub.com)
# Name: agrovision-pivot
# Public or private (your choice)

# Connect local repo to GitHub
git remote add origin https://github.com/YOUR_USERNAME/agrovision-pivot.git
git branch -M main
git push -u origin main
```

### Phase 2: Backend on Railway (30 min)

1. **Create `Procfile` in root:**
   ```bash
   web: cd backend && python main.py
   ```

2. **Create `runtime.txt` in root:**
   ```
   python-3.10.12
   ```

3. **Add environment variables to Railway:**
   - `FLASK_ENV=production`
   - `PYTHONUNBUFFERED=1`

4. **Push to GitHub:**
   ```bash
   git add Procfile runtime.txt
   git commit -m "chore: add deployment config"
   git push
   ```

5. **Deploy on Railway:**
   - Go to https://railway.app
   - Create project → Deploy from GitHub
   - Select your repo
   - Railway auto-deploys
   - Get your URL (e.g., `https://agrovision-prod.railway.app`)

### Phase 3: Frontend on Hugging Face (30 min)

1. **Create separate frontend directory for Spaces:**
   ```bash
   mkdir frontend_hf
   cp frontend/app.py frontend_hf/
   cp frontend/requirements.txt frontend_hf/
   ```

2. **Update `frontend_hf/app.py`:**
   ```python
   # Change this:
   # API_BASE = "http://localhost:5000"
   
   # To this:
   API_BASE = "https://agrovision-prod.railway.app"
   ```

3. **Create `frontend_hf/README.md`:**
   ```markdown
   # AgroVision+ Plant Diagnosis
   
   Upload a photo of your plant and get AI diagnosis + recommendations.
   
   Backend API: https://agrovision-prod.railway.app
   ```

4. **Create separate GitHub repo for HF Space:**
   - Name: `agrovision-frontend`
   - Push files from `frontend_hf/`

5. **Create Hugging Face Space:**
   - Go to https://huggingface.co/spaces
   - Create new Space
   - Choose Streamlit template
   - Connect GitHub
   - Select `agrovision-frontend` repo
   - Auto-deploys!

---

## What Happens During Deployment

### Railway Backend
```
1. Git push to GitHub
   ↓
2. Railway webhook triggered
   ↓
3. Railway pulls from GitHub
   ↓
4. Installs dependencies (backend/requirements.txt)
   ↓
5. Downloads models (ResNet weights, Ollama, etc.)
   ↓
6. Starts Flask app
   ↓
7. Ready at: https://agrovision-prod.railway.app
```

**Time:** 5-10 minutes first deploy, 1-2 minutes for updates

### Hugging Face Space
```
1. Git push to GitHub
   ↓
2. HF webhook triggered
   ↓
3. HF pulls from GitHub
   ↓
4. Installs dependencies (frontend/requirements.txt)
   ↓
5. Starts Streamlit app
   ↓
6. Ready at: https://huggingface.co/spaces/YOUR_NAME/agrovision-frontend
```

**Time:** 3-5 minutes first deploy, 1 minute for updates

---

## Alternative: Everything on Railway

If you want simplicity, run both frontend + backend on Railway:

```
1 Railway Project:
├── Backend: Flask on :5000
└── Frontend: Streamlit on :8501

Expose both ports
Access at:
- https://agrovision.railway.app:5000   (API)
- https://agrovision.railway.app:8501   (UI)
```

**Pros:**
- Single deployment
- Shared environment

**Cons:**
- Both need to run on same machine
- Harder to scale separately

---

## Cost Breakdown

### Option A: Hugging Face Spaces (Free) + Railway ($5/month)

| Component | Cost | Notes |
|-----------|------|-------|
| Hugging Face | FREE | Limited CPU, may timeout |
| Railway | $5-10/month | Free credits + usage |
| Domain | FREE | huggingface.co subdomain |
| **Total** | **$5-10/month** | **Good for demos** |

### Option B: Railway Everything ($10-20/month)

| Component | Cost | Notes |
|-----------|------|-------|
| Railway Backend | $5-10/month | Flask + models |
| Railway Frontend | $2-5/month | Streamlit |
| Domain | FREE | railway.app subdomain |
| **Total** | **$10-20/month** | **More reliable** |

### Option C: AWS Free Tier (First Year)

| Component | Cost | Notes |
|-----------|------|-------|
| EC2 t2.micro | FREE | 1 year, then $10+/month |
| RDS (if needed) | FREE | Not needed for this project |
| **Total** | **$0 (year 1)**, then **$10+** | **Learning curve steep** |

---

## My Specific Recommendation For You

### Use This Stack:

1. **Backend: Railway**
   - Runs Flask + Ollama + models
   - $5-10/month
   - Reliable and simple

2. **Frontend: Hugging Face Spaces**
   - Runs Streamlit
   - FREE
   - Calls Railway backend

3. **Code: GitHub**
   - Single repo with `/backend` and `/frontend_hf`
   - Railway pulls from `main` automatically
   - HF Spaces pulls from separate branch/repo

### Why This Stack?

✅ **Cheap:** $5-10/month total (or FREE on HF only)  
✅ **Simple:** Git push → auto-deploys  
✅ **Fast to setup:** 1-2 hours total  
✅ **Easy to show professors:** Public URLs  
✅ **Learning:** You understand deployment  
✅ **Scalable:** Can upgrade Railway to GPU if needed  

---

## Timeline for Deployment

```
AFTER YOU HAVE WORKING CODE (Day 2):

1. Set up GitHub repo (15 min)
   - git init
   - git push to GitHub

2. Deploy to Railway (30 min)
   - Create account
   - Connect repo
   - Deploy
   - Get URL

3. Deploy to HF Spaces (30 min)
   - Create account
   - Create Space
   - Update API_BASE in code
   - Deploy

TOTAL: 75 minutes

Result: Working demo accessible from anywhere
```

---

## What You Need to Know NOW (While Building)

### 1. Keep Dependencies Clean
```bash
# backend/requirements.txt
flask==2.3.0
torch==2.0.0
# ... only what you need
```

### 2. Use Environment Variables
```python
# Don't hardcode API URLs
import os

API_BASE = os.getenv('API_BASE', 'http://localhost:5000')
FLASK_PORT = int(os.getenv('PORT', 5000))
```

### 3. Log Properly
```python
# For debugging on Railway
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### 4. Handle Model Loading
```python
# Models should load once on startup, not per request
@app.before_first_request
def load_models():
    global plant_identifier
    plant_identifier = PlantIdentifier(...)
```

---

## Potential Issues & Solutions

### Issue 1: Models Too Large for Hugging Face
**Solution:** Store models only on Railway backend, Space never downloads them

### Issue 2: Inference Timeout (>30 seconds)
**Solution:** Either:
- Use smaller model (ResNet18 instead of 152)
- Host on Railway with GPU ($100+/month)
- Accept slow first inference

### Issue 3: Ollama Not Available
**Solution:** Pre-install Ollama in Railway container via custom Dockerfile

### Issue 4: Cold Start (First request slow)
**Solution:** Acceptable for demo. Add loading message to Streamlit.

### Issue 5: Models Take Too Long to Download
**Solution:** 
- Add startup script that downloads models on first deploy
- Or use Hugging Face Hub to host models
- Or use smaller models

---

## Decision: What Should You Do?

### Option A: Simplest (Recommended for Right Now)
**Don't worry about hosting yet.**
- Finish building locally
- Get it working on your machine
- Then worry about hosting (1-2 hours before demo)

### Option B: Deploy Early (Better Practice)
**Set up Railway + HF Spaces today**
- Test that deployment works
- Iron out issues early
- Ship early and often (good dev practice)

### Option C: Do Nothing (For Now)
**Demo locally to professor**
- Works fine
- Less setup
- Less to worry about during build

---

## What I Recommend

### For This Week:
1. **Focus on building** (not hosting)
2. **Keep code clean** for eventual hosting
3. **Use environment variables** for URLs
4. **Test locally** end-to-end

### Before Demo (Day 3):
1. **Deploy to Railway** (backend) - 30 min
2. **Deploy to HF Spaces** (frontend) - 30 min
3. **Test public URLs** - 15 min
4. **Get shareable demo link** - 5 min

---

## Questions for You

To finalize the hosting plan, answer these:

1. **Who's your audience?**
   - Just professor? (local demo OK)
   - Class? (need public URL)
   - Portfolio? (need reliable hosting)

2. **Budget?**
   - $0/month? (Hugging Face only)
   - $5-10/month? (Railway + HF)
   - $50+/month? (GPU deployment)

3. **Timeline?**
   - Deploy after working demo? (recommended)
   - Deploy before demo? (early testing)
   - Demo locally only? (simplest)

4. **Inference speed acceptable?**
   - 20-30 seconds per request? (current setup, OK for demo)
   - <5 seconds? (need GPU, expensive)

---

## My Final Recommendation

**Do this:**

1. **Now:** Build locally, don't worry about hosting
2. **Day 2:** Push to GitHub
3. **Day 3 Morning:** Deploy to Railway + HF Spaces (1 hour)
4. **Day 3:** Demo with public URLs

**This way:** You focus on building great code, not deployment complexity.

---

## If You Want to Deploy Right Now

### 5-Minute Start:

```bash
# 1. Create GitHub repo
git init
git add .
git commit -m "init"
git remote add origin https://github.com/YOUR_USERNAME/agrovision-pivot.git
git push -u origin main

# 2. Sign up for Railway
# Go to: https://railway.app
# Connect GitHub
# Create project → Deploy from GitHub

# 3. That's it!
# Railway handles the rest
```

You'll have a working URL in 5-10 minutes.

---

## Next Steps

**Choose your path:**

### Path A: Deploy Now (Recommended for Learning)
→ I'll give you step-by-step Railway + HF Spaces deployment guide

### Path B: Build First, Deploy Later (Safer)
→ Focus on code, I'll help with deployment Day 3

### Path C: Demo Locally Only (Simplest)
→ No deployment needed, show working on your laptop

**Which path do you want to take?**
