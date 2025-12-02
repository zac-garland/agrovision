# Development Guide

---

## Project Status

**Repository Setup:** ✅ Complete
**Backend Structure:** ✅ Ready for implementation
**Frontend Structure:** ✅ Ready for implementation
**API Contract:** ✅ Locked and final

---

## For Person A (Backend - Zac)

### What You Have

```
backend/
├── main.py              # TODO: Create Flask app
├── requirements.txt     # ✅ Ready
├── app/
│   ├── api/             # TODO: Create routes
│   ├── services/        # TODO: Create business logic
│   │   ├── plant_identifier.py    # Load PlantNet
│   │   ├── llm_agent.py           # Agentic reasoning
│   │   └── knowledge_base.py      # Plant facts
│   ├── models/          # TODO: Request/response schemas
│   └── utils/           # TODO: Config, helpers
└── README.md           # ✅ Your guide
```

### Quick Start

```bash
cd backend
pip install -r requirements.txt

# Implement in this order:
# 1. app/services/plant_identifier.py
# 2. app/services/llm_agent.py
# 3. app/services/knowledge_base.py
# 4. app/api/identify.py
# 5. app/api/diagnose.py
# 6. app/api/questions.py
# 7. main.py

python main.py
# Test at http://localhost:5000
```

### Use Claude/Cursor

When you're ready to implement, try:

**Prompt 1:** Plant Identifier
```
"I need to load PlantNet-300K (ResNet152) and run inference on images.
Requirements:
- Auto-detect ResNet152 or ResNet18 weights
- Load from /Users/zacgarland/r_projects/agrovision-pivot/models/weights/
- Load metadata mappings (species names)
- Return top-5 predictions with confidence

Create: app/services/plant_identifier.py with:
- PlantIdentifier class
- __init__(weights_path, metadata_path)
- identify(image_path) → {species, confidence, top_5}

The model weights are already downloaded."
```

**Prompt 2:** LLM Agent
```
"I need an agentic plant diagnosis system using Mistral 7B via Ollama.

Create: app/services/llm_agent.py with:
- PlantDiagnosisAgent class
- __init__(knowledge_base_path)
- diagnose(species, symptoms, conditions, answers) → {diagnosis, recommendations, timeline}
- ask_follow_ups(species, diagnosis) → {should_ask, questions}

Use these patterns:
- Prompt: Include species info from knowledge base
- Parse: JSON from LLM response
- Fallback: Retry if JSON invalid
- Threshold: If confidence < 0.7, ask follow-ups"
```

**Prompt 3:** Flask API
```
"Create Flask API endpoints:
- POST /identify (file upload)
- POST /diagnose (JSON body)
- POST /ask_follow_ups (JSON body)

See docs/API_CONTRACT.md for spec.
Use the services I created.
Return responses in the exact format specified."
```

---

## For Person B (Frontend)

### What You Have

```
frontend/
├── app.py              # TODO: Create Streamlit app
├── requirements.txt    # ✅ Ready
├── pages/              # TODO: Create pages (optional)
└── README.md          # ✅ Your guide
```

### Quick Start

```bash
cd frontend
pip install -r requirements.txt

# Wait for Person A to have /identify endpoint working
# Then implement in this order:
# 1. Image upload + display
# 2. Call /identify endpoint
# 3. Show plant identification
# 4. Call /ask_follow_ups endpoint
# 5. Show questions form
# 6. Call /diagnose endpoint
# 7. Show results

streamlit run app.py
# Opens at http://localhost:8501
```

### Use Claude/Cursor

When you're ready to implement, try:

**Prompt 1:** Basic Structure
```
"Create a Streamlit app for plant diagnosis:
- Page title: '🌱 AgroVision+ Plant Diagnosis'
- File uploader for images (jpg, jpeg, png)
- Display uploaded image

Create: frontend/app.py"
```

**Prompt 2:** Backend Integration
```
"Add to Streamlit app:
- Call POST http://localhost:5000/identify with uploaded image
- Display plant species and confidence
- Show top-5 predictions

Reference: docs/API_CONTRACT.md for exact format"
```

**Prompt 3:** Full Flow
```
"Complete the Streamlit app:
- Step 1: Upload image → /identify
- Step 2: Show plant ID
- Step 3: Ask 'Do you see these symptoms?'
- Step 4: /ask_follow_ups (if confidence low)
- Step 5: Get answers to questions
- Step 6: /diagnose with all info
- Step 7: Show diagnosis + recommendations + timeline

Make it visually nice with metrics and columns."
```

---

## Communication Rules

### Daily Standup (9 AM)

Both people: 2-minute update
- What you built yesterday
- What you're building today
- Any blockers

### API Changes

If API needs to change:
1. **Propose** in Slack
2. **Get agreement** from other person
3. **Update** `docs/API_CONTRACT.md`
4. **Implement** changes

### Blockers

If stuck:
1. Check `docs/API_CONTRACT.md`
2. Check your component README (backend/README.md or frontend/README.md)
3. Ask in Slack (other person might know)
4. Use Claude/Cursor to help debug

---

## Testing Strategy

### Person A: Test Backend Endpoints

```bash
# Test plant identification
curl -X POST http://localhost:5000/identify \
  -F "image=@/Users/zacgarland/r_projects/agrovision-pivot/test-image.jpeg"

# Should return:
# {"success": true, "species": "...", "confidence": 0.85, "top_5": [...]}
```

```bash
# Test ask follow-ups
curl -X POST http://localhost:5000/ask_follow_ups \
  -H "Content-Type: application/json" \
  -d '{
    "species": "Alocasia macrorrhizos",
    "diagnosis": {"issue": "root rot", "confidence": 0.75}
  }'

# Should return:
# {"success": true, "should_ask_follow_ups": true, "questions": [...]}
```

```bash
# Test diagnosis
curl -X POST http://localhost:5000/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "species": "Alocasia macrorrhizos",
    "symptoms": "brown spots, wilting",
    "conditions": {"humidity": "low"},
    "user_answers": {"q1": "a1"}
  }'

# Should return:
# {"success": true, "diagnosis": {...}, "recommendations": {...}, "timeline": {...}}
```

### Person B: Test Frontend with Backend

Once backend is up:

```bash
# Start backend
cd backend
python main.py

# In another terminal, start frontend
cd frontend
streamlit run app.py

# Test flow:
# 1. Upload test-image.jpeg
# 2. See if plant ID works
# 3. Answer questions
# 4. See diagnosis
```

---

## Git Workflow

### First Time Setup

```bash
cd /Users/zacgarland/r_projects/agrovision-pivot

# Initialize git (if not done)
git init

# Create .gitignore (already done)
git add .gitignore

# Initial commit
git commit -m "init: project structure and documentation"
git branch -M main
```

### Person A: Feature Branch

```bash
git checkout -b feature/plant-identifier
# Make changes in backend/
git add backend/
git commit -m "feat: implement PlantNet inference"
git push origin feature/plant-identifier
# Create PR on GitHub

# After review, merge to main
git checkout main
git merge feature/plant-identifier
```

### Person B: Feature Branch

```bash
git checkout -b feature/streamlit-ui
# Make changes in frontend/
git add frontend/
git commit -m "feat: implement image upload"
git push origin feature/streamlit-ui
# Create PR on GitHub

# After review, merge to main
git checkout main
git merge feature/streamlit-ui
```

---

## Timeline

```
TODAY:
  ✅ Project structure
  ✅ Documentation
  ✅ API contract

TOMORROW (Day 1):
  Morning (9-12): Implementation
    - Person A: PlantNet + LLM agent
    - Person B: Image upload + display
  
  Afternoon (12-3): Integration
    - Connect endpoints
    - Test flows
    - Debug issues

  Evening: Polish
    - Error handling
    - Styling
    - Edge cases

Day 2:
  Final testing + presentation prep

Day 3:
  Presentation
```

---

## Resources

- **API Spec:** `docs/API_CONTRACT.md`
- **Backend Guide:** `backend/README.md`
- **Frontend Guide:** `frontend/README.md`
- **Project README:** `README.md`

---

## Success Checklist (Before Demo)

### Backend
- [ ] PlantNet loads weights correctly
- [ ] Inference works (identifies test image)
- [ ] Flask app runs on port 5000
- [ ] `/identify` endpoint works
- [ ] `/ask_follow_ups` endpoint works
- [ ] `/diagnose` endpoint works
- [ ] Error handling in place
- [ ] Ollama running and responding

### Frontend
- [ ] Streamlit runs on port 8501
- [ ] Image upload works
- [ ] `/identify` integration works
- [ ] Shows plant ID + confidence
- [ ] Questions form displays correctly
- [ ] `/diagnose` integration works
- [ ] Results display looks good
- [ ] Error messages helpful

### Integration
- [ ] Full flow works end-to-end
- [ ] No CORS issues
- [ ] No connection errors
- [ ] Demo plant diagnoses correctly
- [ ] UI is responsive and clean

---

## Good Luck! 🌱

You've got a solid plan. Execute it step by step. Communicate. Help each other. Ship it.
