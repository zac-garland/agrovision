# AgroVision+ Agentic Plant Diagnosis System

**A multi-turn reasoning system for plant health diagnosis using PlantNet-300K and local LLMs.**

---

## Project Overview

Instead of a single-shot classifier, we build an **agentic system** that:
1. Identifies plant species from photos (PlantNet-300K)
2. Asks clarifying questions if unsure
3. Diagnoses issues with reasoning
4. Provides personalized recommendations
5. Explains its logic to users

**Architecture:**
- **Backend:** Flask API + PlantNet inference + Mistral 7B LLM reasoning
- **Frontend:** Streamlit UI for image upload, Q&A, results display

---

## 📁 Project Structure

```
agrovision-pivot/
├── backend/                    # Person A (Zac) - API & LLM logic
│   ├── app/
│   │   ├── api/               # Flask routes
│   │   ├── services/          # Core business logic
│   │   ├── models/            # Data models
│   │   ├── utils/             # Helper functions
│   │   └── __init__.py
│   ├── main.py                # Flask app entry point
│   ├── requirements.txt        # Python dependencies
│   └── README.md              # Backend-specific docs
│
├── frontend/                   # Person B - UI & visualization
│   ├── app.py                 # Streamlit main app
│   ├── pages/                 # Streamlit pages
│   ├── components/            # Reusable UI components
│   ├── requirements.txt        # Frontend dependencies
│   └── README.md              # Frontend-specific docs
│
├── models/                     # Pre-trained weights & metadata
│   ├── weights/               # PlantNet weights (gitignored)
│   │   ├── resnet18_weights_best_acc.tar
│   │   └── resnet152_weights_best_acc.tar
│   └── knowledge_base.json    # Plant care facts
│
├── docs/                       # 📖 DOCUMENTATION (READ THESE!)
│   ├── API_CONTRACT.md        # Frontend-Backend API spec ⭐
│   ├── HOSTING_QUICK.md       # Hosting decision guide
│   ├── HOSTING.md             # Detailed hosting analysis
│   ├── DEPLOY_STEP_BY_STEP.md # Deployment playbook
│   ├── HOSTING_SUMMARY.md     # Quick reference
│   ├── HOSTING_INDEX.md       # Doc navigation
│   ├── DEVELOPMENT.md         # Development guide & timeline
│   └── ARCHITECTURE.md        # System design
│
├── meta/                       # 📦 NON-ESSENTIAL FILES (Archive)
│   ├── plantnet_minimal_test.py  # Old validation test
│   ├── test-image.jpeg           # Test plant image
│   ├── datascience_env/          # Archived Python environment
│   ├── PROJECT_STATUS.md         # Initial project status
│   ├── HOSTING_DISCUSSION.md     # Hosting planning notes
│   └── README.md                 # Archive documentation
│
├── .gitignore                 # Git configuration
├── START_HERE.md              # 👈 READ THIS FIRST! (5 min)
└── README.md                  # This file

```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama (for local LLM)
- Git

### Backend Setup (Person A - Zac)

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Install Ollama and Mistral
curl https://ollama.ai/install.sh | sh
ollama pull mistral

# 3. Run Flask API
python main.py
# Server starts at http://localhost:5000
```

### Frontend Setup (Person B)

```bash
# 1. Install dependencies
cd frontend
pip install -r requirements.txt

# 2. Run Streamlit app (make sure backend is running)
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 📖 Where to Start

### For New Team Members
1. Read **`START_HERE.md`** (5 minutes)
2. Read **`docs/API_CONTRACT.md`** (15 minutes)
3. Read your component's README (`backend/README.md` or `frontend/README.md`)

### For Backend Dev (Person A)
→ Follow `backend/README.md` for step-by-step guide

### For Frontend Dev (Person B)
→ Follow `frontend/README.md` for step-by-step guide

### For Deployment (Day 3)
→ Follow `docs/DEPLOY_STEP_BY_STEP.md`

---

## 🔑 Key Files

| File | Purpose | Read When |
|------|---------|-----------|
| `START_HERE.md` | Project overview | First time setup |
| `docs/API_CONTRACT.md` | Backend/Frontend spec | Before coding |
| `backend/README.md` | Backend implementation | Person A's guide |
| `frontend/README.md` | Frontend implementation | Person B's guide |
| `docs/DEVELOPMENT.md` | Team workflow & timeline | Need timeline |
| `docs/HOSTING_QUICK.md` | Hosting decision | Before deployment |
| `docs/DEPLOY_STEP_BY_STEP.md` | Deployment guide | Day 3 morning |

---

## API Contract (Backend ↔ Frontend)

**The critical agreement between A and B**

See `docs/API_CONTRACT.md` for complete specification.

**Quick Summary:**
```
POST /identify
  Input: image file
  Output: {species, confidence, top_5}

POST /ask_follow_ups
  Input: {species, diagnosis}
  Output: {questions: [...]}

POST /diagnose
  Input: {species, symptoms, conditions, answers}
  Output: {diagnosis, confidence, recommendations, explanation}
```

---

## 👥 Team Roles

| Person | Role | Responsibilities |
|--------|------|------------------|
| **Zac (A)** | Backend Lead | PlantNet inference, LLM agent, Flask API, knowledge base |
| **Person B** | Frontend Lead | Streamlit UI, image upload, forms, results display |

---

## ⏰ Timeline

```
Days 1-2:     Build locally (backend + frontend)
Day 3, 9 AM:  Deploy to Railway + HF Spaces (1.5 hours)
Day 3, 10 AM: Demo with live public URLs
```

See `docs/DEVELOPMENT.md` for detailed timeline.

---

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes, test locally
git commit -m "feat: description of changes"

# Push and create PR
git push origin feature/your-feature

# After review, merge to main
```

**Rule:** Always use PRs, never push directly to main.

---

## 📦 What's in `/meta`?

Non-essential files kept here to keep root clean:
- Old test scripts
- Test images
- Archived environments
- Project status docs
- Planning notes

See `meta/README.md` for details. **You won't need these for development.**

---

## ❓ Questions?

| Question | Answer |
|----------|--------|
| "What do I do first?" | Read `START_HERE.md` |
| "What's the API?" | Read `docs/API_CONTRACT.md` |
| "How do I build my part?" | Read your component's README |
| "What's the timeline?" | Read `docs/DEVELOPMENT.md` |
| "How do I deploy?" | Read `docs/DEPLOY_STEP_BY_STEP.md` |

---

**Let's build something cool.** 🌱
