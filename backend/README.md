# Backend: Plant Diagnosis API

**Your job:** Build the Flask API that the frontend calls.

---

## What You're Building

```
Image → PlantNet → Agentic LLM → Recommendations
  ↓       ↓          ↓              ↓
  |    Identify   Diagnose      Format
  |    Species    Issue          Response
  └─→ [Flask API] ←──────────────┘
```

**Three endpoints to implement:**
1. `POST /identify` - Identify plant from image
2. `POST /ask_follow_ups` - Generate clarifying questions
3. `POST /diagnose` - Get full diagnosis + recommendations

See `docs/API_CONTRACT.md` for exact request/response formats.

---

## Project Structure

```
backend/
├── main.py                          # Flask app entry point
├── requirements.txt                 # Python dependencies
├── app/
│   ├── __init__.py
│   ├── api/                         # Flask routes (YOU IMPLEMENT)
│   │   ├── __init__.py
│   │   ├── identify.py              # POST /identify
│   │   ├── diagnose.py              # POST /diagnose
│   │   └── questions.py             # POST /ask_follow_ups
│   ├── services/                    # Core logic (YOU IMPLEMENT)
│   │   ├── __init__.py
│   │   ├── plant_identifier.py      # Load PlantNet, run inference
│   │   ├── llm_agent.py             # Agentic reasoning loop
│   │   └── knowledge_base.py        # Load plant facts
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py               # Request/response validation
│   └── utils/
│       ├── __init__.py
│       └── config.py                # Configuration
└── README.md                        # This file
```

---

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install Ollama + Mistral (if not done)

```bash
# Download Ollama
curl https://ollama.ai/install.sh | sh

# Pull Mistral model
ollama pull mistral

# Verify it works
ollama run mistral "What is plant care?"
```

### 3. Verify PlantNet Weights

Make sure these files exist in `models/weights/`:
```
/Users/zacgarland/r_projects/agrovision-pivot/models/weights/
├── resnet152_weights_best_acc.tar  (or resnet18)
├── plantnet300K_species_id_2_name.json
└── class_idx_to_species_id.json
```

If not, download from: https://github.com/plantnet/PlantNet-300K

---

## Development Checklist

### Phase 1: PlantNet Loading (1.5 hours)

**File:** `app/services/plant_identifier.py`

```python
class PlantIdentifier:
    def __init__(self, weights_path, metadata_path):
        # Load ResNet152 model
        # Load species mappings
        # Set to eval mode
        pass
    
    def identify(self, image_path):
        # Load image
        # Preprocess (resize, normalize)
        # Run inference
        # Return top-5 predictions with confidence
        return {
            'species': 'Alocasia macrorrhizos',
            'confidence': 0.85,
            'top_5': [...]
        }
```

**Test it:**
```bash
python -c "
from app.services.plant_identifier import PlantIdentifier
pi = PlantIdentifier(...)
result = pi.identify('test-image.jpeg')
print(result)
"
```

---

### Phase 2: LLM Agent (2 hours)

**File:** `app/services/llm_agent.py`

```python
class PlantDiagnosisAgent:
    def __init__(self, knowledge_base_path):
        # Load knowledge base (JSON with plant facts)
        # Initialize Ollama client
        pass
    
    def diagnose(self, species, symptoms, conditions, answers=None):
        # Build prompt with context
        # Call Mistral via Ollama
        # Parse JSON response
        # Return diagnosis + recommendations
        return {
            'diagnosis': {
                'primary_issue': '...',
                'confidence': 0.82,
                'reasoning': '...',
                'alternatives': [...]
            },
            'recommendations': {
                'immediate': [...],
                'short_term': [...],
                'long_term': [...]
            },
            'timeline': {...}
        }
    
    def ask_follow_ups(self, species, diagnosis):
        # If confidence < 0.7, generate follow-up questions
        # Otherwise, return empty
        return {
            'should_ask': True/False,
            'questions': [...]
        }
```

**Test it:**
```bash
python -c "
from app.services.llm_agent import PlantDiagnosisAgent
agent = PlantDiagnosisAgent(...)
diagnosis = agent.diagnose('Alocasia macrorrhizos', 'brown spots, wilting', {})
print(diagnosis)
"
```

---

### Phase 3: Knowledge Base (30 min)

**File:** `models/knowledge_base.json`

```json
{
  "Alocasia macrorrhizos": {
    "common_names": ["Elephant Ear", "Giant Taro"],
    "native_to": "Southeast Asia",
    "water": {
      "frequency": "every 5-7 days",
      "preference": "moderate, let soil dry between waterings",
      "tips": "Use distilled water if possible"
    },
    "light": "bright indirect light, tolerates some direct sun",
    "humidity": "moderate to high (50-70%)",
    "temperature": "18-27°C (65-80°F)",
    "common_issues": [
      {
        "symptom": "brown leaf tips",
        "causes": ["low humidity", "chlorine in water"],
        "treatment": "increase humidity, use distilled water"
      },
      {
        "symptom": "wilting leaves",
        "causes": ["overwatering", "root rot", "underwatering"],
        "treatment": "check soil moisture, adjust watering"
      }
    ]
  },
  // Add 4-9 more common plants
  // Start with common ones: Monstera, Pothos, Snake Plant, etc.
}
```

**File:** `app/services/knowledge_base.py`

```python
import json

class KnowledgeBase:
    def __init__(self, path='models/knowledge_base.json'):
        with open(path) as f:
            self.data = json.load(f)
    
    def get_plant_info(self, species_name):
        return self.data.get(species_name, {})
    
    def get_common_issues(self, species_name):
        info = self.get_plant_info(species_name)
        return info.get('common_issues', [])
```

---

### Phase 4: Flask API Routes (1 hour)

**File:** `app/api/identify.py`

```python
from flask import Blueprint, request, jsonify

bp = Blueprint('identify', __name__)

@bp.route('/identify', methods=['POST'])
def identify():
    # Get image from request
    # Call plant_identifier.identify(image)
    # Return response
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    image = request.files['image']
    result = plant_identifier.identify(image)
    
    return jsonify({
        'success': True,
        'species': result['species'],
        'confidence': result['confidence'],
        'top_5': result['top_5']
    })
```

**File:** `app/api/diagnose.py`

```python
from flask import Blueprint, request, jsonify

bp = Blueprint('diagnose', __name__)

@bp.route('/diagnose', methods=['POST'])
def diagnose():
    data = request.get_json()
    
    # Validate required fields
    required = ['species', 'symptoms', 'conditions']
    if not all(f in data for f in required):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    
    # Call agent.diagnose()
    result = agent.diagnose(
        data['species'],
        data['symptoms'],
        data['conditions'],
        data.get('user_answers')
    )
    
    return jsonify({'success': True, **result})
```

**File:** `main.py`

```python
from flask import Flask
from app.api import identify, diagnose, questions

app = Flask(__name__)

# Register blueprints
app.register_blueprint(identify.bp)
app.register_blueprint(diagnose.bp)
app.register_blueprint(questions.bp)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

### Phase 5: Integration & Testing (1 hour)

```bash
# Start the API
python main.py

# In another terminal, test endpoints
curl -X POST http://localhost:5000/identify \
  -F "image=@test-image.jpeg"

curl -X POST http://localhost:5000/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "species": "Alocasia macrorrhizos",
    "symptoms": "brown spots",
    "conditions": {"humidity": "low"}
  }'
```

---

## Handling Ollama/LLM

### Starting Ollama

```bash
# In a separate terminal, keep this running
ollama serve
```

### Calling Ollama from Python

```python
from ollama import generate

response = generate(
    model='mistral',
    prompt='What causes root rot in plants?'
)
print(response)
```

### If Ollama Isn't Working

1. Check it's running: `ollama serve` in another terminal
2. Check model is downloaded: `ollama list`
3. If not: `ollama pull mistral`
4. Test: `ollama run mistral "Hello"`

---

## Debugging Tips

### PlantNet not loading?
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Try loading weights
checkpoint = torch.load('resnet152_weights_best_acc.tar')
print(f"Checkpoint keys: {checkpoint.keys()}")
```

### LLM not responding?
```python
from ollama import generate

# Test basic LLM
response = generate('mistral', 'Hello')
print(response)

# Check if model exists
import subprocess
result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
print(result.stdout)
```

### Flask not starting?
```bash
# Check port 5000 is free
lsof -i :5000

# Check dependencies
pip list | grep -E "flask|torch|ollama"
```

---

## Timeline

```
9:00 AM  - Install deps, test PlantNet loading
10:00 AM - Build LLM agent
11:00 AM - Build knowledge base + API routes
12:00 PM - Test endpoints
1:00 PM  - Ready for frontend integration
```

---

## When Frontend is Ready

Person B will call your endpoints like this:

```python
import requests

# 1. Identify
response = requests.post('http://localhost:5000/identify', files={'image': image})

# 2. Ask follow-ups
response = requests.post('http://localhost:5000/ask_follow_ups', json={
    'species': '...',
    'diagnosis': {...}
})

# 3. Diagnose
response = requests.post('http://localhost:5000/diagnose', json={
    'species': '...',
    'symptoms': '...',
    'conditions': {...},
    'user_answers': {...}
})
```

Your job is to make sure these work perfectly.

---

## Resources

- PlantNet-300K: https://github.com/plantnet/PlantNet-300K
- Ollama: https://ollama.ai
- Flask: https://flask.palletsprojects.com/
- Requests: https://requests.readthedocs.io/

---

**Let's build it.** 🌱
