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

## Project Structure

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
│   ├── metadata/              # Species mappings
│   │   ├── plantnet300K_species_id_2_name.json
│   │   └── class_idx_to_species_id.json
│   └── knowledge_base.json    # Plant care facts
│
├── docs/                       # Documentation (READ THESE!)
│   ├── API_CONTRACT.md        # Frontend-Backend API spec ⭐
│   ├── HOSTING_QUICK.md       # Hosting decision guide
│   ├── HOSTING.md             # Detailed hosting analysis
│   ├── DEPLOY_STEP_BY_STEP.md # Deployment playbook
│   ├── HOSTING_SUMMARY.md     # Quick reference
│   ├── HOSTING_INDEX.md       # Doc navigation
│   ├── DEVELOPMENT.md         # Development guide
│   └── ARCHITECTURE.md        # System design
│
├── meta/                       # Non-essential files & archives
│   ├── plantnet_minimal_test.py  # Old validation test (reference only)
│   ├── test-image.jpeg           # Test plant image
│   ├── datascience_env/          # Archived Python environment
│   ├── PROJECT_STATUS.md         # Initial project status
│   ├── HOSTING_DISCUSSION.md     # Hosting planning notes
│   └── README.md                 # What's in this folder
│
├── .gitignore                 # Git ignore rules
├── START_HERE.md              # Entry point - READ THIS FIRST! 👈
├── README.md                  # This file
└── requirements-dev.txt       # Dev dependencies
```

---

## Quick Start

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

## Critical Files

### 📍 Entry Points
- **`START_HERE.md`** ← Read this first (5 min overview)
- **`docs/API_CONTRACT.md`** ← The agreement between backend & frontend

### 📚 Development Guides
- **`backend/README.md`** ← Backend implementation guide
- **`frontend/README.md`** ← Frontend implementation guide
- **`docs/DEVELOPMENT.md`** ← Team workflow & timeline

### 🚀 Deployment
- **`docs/HOSTING_QUICK.md`** ← Which hosting option?
- **`docs/DEPLOY_STEP_BY_STEP.md`** ← How to deploy (Day 3)

---

## Development Workflow

### For Person A (Backend - Zac)

**Focus Areas:**
1. Load PlantNet-300K model
2. Build LLM agent with multi-turn reasoning
3. Create Flask API endpoints
4. Integrate knowledge base
5. Handle errors gracefully

**API Endpoints to Implement:**
- `POST /identify` - Identify plant species from image
- `POST /ask_follow_ups` - Generate follow-up questions
- `POST /diagnose` - Get diagnosis with recommendations

See `docs/API_CONTRACT.md` for detailed spec.

**Getting Started:**
```bash
cd backend
# Read: README.md in this folder
# Then implement services in app/services/
# Create API routes in app/api/
# Run: python main.py
```

### For Person B (Frontend)

**Focus Areas:**
1. Image upload UI
2. Plant species display
3. Context question form
4. Results display (diagnosis, recommendations)
5. Styling and UX

**Getting Started:**
```bash
cd frontend
# Read: README.md in this folder
# Build Streamlit app against API endpoints
# Run: streamlit run app.py
```

---

## API Contract

**THE CRITICAL AGREEMENT BETWEEN A AND B**

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

## Team Roles

| Person | Role | Responsibilities |
|--------|------|------------------|
| **Zac (A)** | Backend Lead | PlantNet inference, LLM agent, API endpoints, knowledge base |
| **Person B** | Frontend Lead | Streamlit UI, image upload, form handling, results display |

---

## Timeline

- **Days 1-2:** Backend + Frontend implementation (locally)
- **Day 3 Morning:** Deploy to Railway + HF Spaces (1.5 hours)
- **Day 3 Afternoon:** Demo with live URLs

---

## Git Workflow

```bash
# Person A
git checkout -b feature/backend-api
# Make changes in backend/
git commit -m "feat: add /identify endpoint"
git push origin feature/backend-api
# Create PR

# Person B
git checkout -b feature/frontend-ui
# Make changes in frontend/
git commit -m "feat: add image upload"
git push origin feature/frontend-ui
# Create PR
```

**Rule:** Always merge to `main` via PR, don't push directly.

---

## What's in `/meta`?

Non-essential files for keeping the project root clean:
- `plantnet_minimal_test.py` - Old test (reference only)
- `test-image.jpeg` - Test image
- `datascience_env/` - Archived environment
- `PROJECT_STATUS.md` - Status summary
- `HOSTING_DISCUSSION.md` - Hosting notes

See `meta/README.md` for details.

---

## Questions?

- **What do I do first?** → Read `START_HERE.md`
- **How do I implement my part?** → Read `backend/README.md` or `frontend/README.md`
- **What's the API spec?** → Read `docs/API_CONTRACT.md`
- **How do I deploy?** → Read `docs/DEPLOY_STEP_BY_STEP.md`
- **What's my timeline?** → Read `docs/DEVELOPMENT.md`

---

**Let's build something cool.** 🌱
