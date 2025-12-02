# Quick Reference: Directory Structure

## Root Directory (What You See First)

```
agrovision-pivot/
├── README.md              ← Project overview (START HERE)
├── START_HERE.md          ← 5-minute introduction
├── .gitignore             ← Git config
├── backend/               ← Person A's work
├── frontend/              ← Person B's work
├── docs/                  ← Documentation
├── models/                ← Model weights & knowledge base
├── meta/                  ← Archives & old files
└── resnet weights (2x)    ← Model data
```

## What Each Folder Contains

### `backend/` (Person A)
```
backend/
├── main.py                    ← Flask entry point
├── requirements.txt           ← Dependencies
├── README.md                  ← Your guide
└── app/
    ├── api/                   ← Flask routes
    ├── services/              ← Business logic
    ├── models/                ← Data schemas
    └── utils/                 ← Helpers
```

### `frontend/` (Person B)
```
frontend/
├── app.py                     ← Streamlit entry point
├── requirements.txt           ← Dependencies
├── README.md                  ← Your guide
├── pages/                     ← Streamlit pages
└── components/                ← UI components
```

### `docs/` (Documentation)
```
docs/
├── API_CONTRACT.md            ← ⭐ CRITICAL
├── HOSTING_QUICK.md           ← Which hosting?
├── HOSTING.md                 ← Detailed hosting
├── DEPLOY_STEP_BY_STEP.md     ← How to deploy
├── HOSTING_SUMMARY.md         ← Quick ref
├── HOSTING_INDEX.md           ← Doc index
├── DEVELOPMENT.md             ← Team guide
└── ARCHITECTURE.md            ← System design
```

### `models/` (Pre-trained Weights)
```
models/
├── weights/
│   ├── resnet152_weights_best_acc.tar
│   └── resnet18_weights_best_acc.tar
└── knowledge_base.json        ← Plant facts
```

### `meta/` (Archives & Old Files)
```
meta/
├── README.md                  ← What's in here
├── CLEANUP_SUMMARY.md         ← Cleanup docs
├── HOSTING_DISCUSSION.md      ← Planning notes
├── PROJECT_STATUS.md          ← Initial status
├── plantnet_minimal_test.py   ← Old test (ref)
├── test-image.jpeg            ← Test image
├── *.json                      ← Metadata (archive)
├── activate_env.sh            ← Old script
├── package_installation.log   ← Setup log
└── datascience_env/           ← Archived env
```

---

## File Navigation

### "Where do I...?"

| Task | Location |
|------|----------|
| Code the backend | `backend/app/` |
| Code the frontend | `frontend/` |
| See project overview | `README.md` |
| Get started | `START_HERE.md` |
| Check API spec | `docs/API_CONTRACT.md` |
| Find model weights | `models/weights/` |
| Deploy the app | `docs/DEPLOY_STEP_BY_STEP.md` |
| Understand timeline | `docs/DEVELOPMENT.md` |
| Learn hosting options | `docs/HOSTING_QUICK.md` |
| Reference old files | `meta/` |

---

## File Purposes

### Must Read First
- `README.md` - Project overview
- `START_HERE.md` - Getting started

### Implementation Guides
- `backend/README.md` - Backend guide (Person A)
- `frontend/README.md` - Frontend guide (Person B)

### Critical Documentation
- `docs/API_CONTRACT.md` - Backend/Frontend agreement
- `docs/DEVELOPMENT.md` - Team workflow

### Deployment Guides
- `docs/HOSTING_QUICK.md` - Which option?
- `docs/DEPLOY_STEP_BY_STEP.md` - How to deploy

### For Reference Only
- `meta/` - Old files, archives, notes

---

## Entry Points by Role

### For Backend Developer (Person A)
1. Read: `START_HERE.md` (5 min)
2. Read: `docs/API_CONTRACT.md` (15 min)
3. Read: `backend/README.md` (10 min)
4. **Code:** `backend/app/services/` and `backend/app/api/`

### For Frontend Developer (Person B)
1. Read: `START_HERE.md` (5 min)
2. Read: `docs/API_CONTRACT.md` (15 min)
3. Read: `frontend/README.md` (10 min)
4. **Code:** `frontend/app.py` and components

### For Deployment (Day 3)
1. Read: `docs/DEPLOY_STEP_BY_STEP.md`
2. Follow: Phase 1 → Phase 2 → Phase 3 → Phase 4
3. Done!

### For Learning Timeline
1. Read: `docs/DEVELOPMENT.md`
2. See: Timeline section
3. Plan: Days 1-2 for code, Day 3 for deploy

---

## The Root Rule

### What Should Be in Root?
✅ Essential files only
✅ Entry points (README, START_HERE)
✅ Configuration (.gitignore)
✅ Core folders (backend, frontend, docs, models)

### What Should NOT Be in Root?
❌ Old test files → `/meta/`
❌ Archived environments → `/meta/`
❌ Planning notes → `/meta/`
❌ Metadata archives → `/meta/`
❌ Installation logs → `/meta/`

### Why?
**Clean root = professional repo = happy developers**

---

## Summary

```
Start:    README.md
Plan:     docs/DEVELOPMENT.md
Code:     backend/ or frontend/
Spec:     docs/API_CONTRACT.md
Deploy:   docs/DEPLOY_STEP_BY_STEP.md
Archive:  meta/ (don't need)
```

**Everything is organized. Start building.** 🌱
