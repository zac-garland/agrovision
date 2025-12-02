# API Contract: Backend ↔ Frontend Communication

**This is the critical agreement between Person A (Backend) and Person B (Frontend).**

**Status:** LOCKED AS OF [TODAY]
**Backend Implementer:** Person A (Zac)
**Frontend Implementer:** Person B

---

## Overview

Person A implements a Flask API with these 3 endpoints. Person B builds a Streamlit UI that calls these endpoints. They do NOT need to know each other's implementation details.

---

## Endpoint 1: Plant Identification

**Route:** `POST /identify`

**Purpose:** Identify plant species from an image

**Request:**
```json
{
  "image": <binary file data>
}
```

**Form Data:**
```
POST /identify
Content-Type: multipart/form-data

image: [binary image file]
```

**Response (Success - 200):**
```json
{
  "success": true,
  "species": "Alocasia macrorrhizos",
  "confidence": 0.85,
  "top_5": [
    {"name": "Alocasia macrorrhizos", "confidence": 0.85},
    {"name": "Colocasia esculenta", "confidence": 0.10},
    {"name": "Xanthosoma sagittifolium", "confidence": 0.03},
    {"name": "Anthurium andraeanum", "confidence": 0.01},
    {"name": "Scindapsus pictus", "confidence": 0.01}
  ]
}
```

**Response (Error - 400):**
```json
{
  "success": false,
  "error": "No image provided" or "Invalid image format"
}
```

**Python Example (Frontend):**
```python
import requests

files = {'image': open('plant.jpg', 'rb')}
response = requests.post('http://localhost:5000/identify', files=files)
result = response.json()

if result['success']:
    print(f"Plant: {result['species']} ({result['confidence']:.0%})")
    for pred in result['top_5']:
        print(f"  - {pred['name']}: {pred['confidence']:.1%}")
```

---

## Endpoint 2: Ask Follow-Up Questions

**Route:** `POST /ask_follow_ups`

**Purpose:** Generate clarifying questions when plant identification confidence is low

**Request:**
```json
{
  "species": "Alocasia macrorrhizos",
  "diagnosis": {
    "issue": "overwatering",
    "confidence": 0.65
  }
}
```

**Response (Success - 200):**
```json
{
  "success": true,
  "should_ask_follow_ups": true,
  "questions": [
    "How long have you had this plant?",
    "When did you last water it?",
    "Is the soil wet, moist, or dry right now?",
    "Are the spots soft/mushy or hard/crispy?"
  ],
  "reasoning": "Confidence is below 0.70. These questions will help distinguish between overwatering and fungal infection."
}
```

**Response (Success - Plant ID confident - 200):**
```json
{
  "success": true,
  "should_ask_follow_ups": false,
  "questions": [],
  "reasoning": "Confidence is above 0.85. No follow-ups needed."
}
```

**Response (Error - 400):**
```json
{
  "success": false,
  "error": "Invalid species or diagnosis"
}
```

**Python Example (Frontend):**
```python
response = requests.post('http://localhost:5000/ask_follow_ups', json={
    "species": "Alocasia macrorrhizos",
    "diagnosis": {"issue": "root rot", "confidence": 0.75}
})

result = response.json()

if result['success'] and result['should_ask_follow_ups']:
    # Display questions to user
    for q in result['questions']:
        st.write(f"❓ {q}")
else:
    # Skip follow-ups, go straight to diagnosis
    st.write("We're confident in the diagnosis, no questions needed.")
```

---

## Endpoint 3: Get Diagnosis & Recommendations

**Route:** `POST /diagnose`

**Purpose:** Get AI diagnosis, recommendations, and explanations

**Request:**
```json
{
  "species": "Alocasia macrorrhizos",
  "symptoms": "brown spots on leaves, wilting, soil is very wet",
  "conditions": {
    "humidity": "low (30%)",
    "temperature": "20°C (68°F)",
    "light": "bright indirect",
    "location": "bathroom"
  },
  "user_answers": {
    "How long have you had this plant?": "6 months",
    "When did you last water it?": "3 days ago",
    "Is the soil wet or dry?": "very wet"
  }
}
```

**Response (Success - 200):**
```json
{
  "success": true,
  "diagnosis": {
    "primary_issue": "Root rot from overwatering",
    "confidence": 0.82,
    "alternatives": [
      {"issue": "Fungal infection", "likelihood": 0.12},
      {"issue": "Low humidity stress", "likelihood": 0.06}
    ],
    "reasoning": "Given Alocasia macrorrhizos with: brown soft spots + wilting + wet soil + recent watering = root rot. The soft, mushy spots (not crispy) rule out fungal. Low humidity is secondary."
  },
  "recommendations": {
    "immediate": [
      "Remove plant from pot and inspect roots",
      "Cut away black/mushy roots",
      "Repot in fresh, well-draining soil"
    ],
    "short_term": [
      "Don't water for 2 weeks (let soil dry)",
      "Move to well-lit location",
      "Increase air circulation (open a window)"
    ],
    "long_term": [
      "Water only when top 2 inches of soil are dry",
      "Use pot with drainage holes",
      "Water less in winter"
    ],
    "monitoring": [
      "Check for new growth in 2-3 weeks",
      "Ensure leaves are firm (not limp)",
      "No additional brown spots"
    ]
  },
  "timeline": {
    "expected_improvement": "2-3 weeks",
    "full_recovery": "4-6 weeks",
    "when_to_escalate": "If plant shows no improvement in 3 weeks or continues to worsen"
  },
  "explanation": "Root rot happens when roots sit in wet soil too long, suffocating them. This plant is tropical but prefers to dry out between waterings. The dark soft spots are classic root rot symptoms. By improving drainage and reducing watering frequency, the plant should recover."
}
```

**Response (Error - 400):**
```json
{
  "success": false,
  "error": "Missing required field: species"
}
```

**Python Example (Frontend):**
```python
response = requests.post('http://localhost:5000/diagnose', json={
    "species": result['species'],
    "symptoms": user_symptoms,
    "conditions": {
        "humidity": "low",
        "temperature": "20",
        "light": "bright indirect",
        "location": "living room"
    },
    "user_answers": {
        "How long have you had this plant?": user_answer_1,
        "When did you last water it?": user_answer_2
    }
})

result = response.json()

if result['success']:
    diag = result['diagnosis']
    
    # Display diagnosis
    st.subheader(f"🔍 Diagnosis: {diag['primary_issue']}")
    st.write(f"**Confidence:** {diag['confidence']:.0%}")
    st.write(f"**Reasoning:** {diag['reasoning']}")
    
    # Display recommendations
    st.subheader("💡 What to Do")
    st.write("**Today:**")
    for step in result['recommendations']['immediate']:
        st.write(f"- {step}")
    
    # Display timeline
    st.write(f"**Expected improvement:** {result['timeline']['expected_improvement']}")
```

---

## Error Handling

All endpoints follow this error format:

```json
{
  "success": false,
  "error": "<human-readable error message>",
  "error_code": "<error_type>"
}
```

**Possible error codes:**
- `INVALID_IMAGE` - Image file is missing or corrupted
- `INVALID_JSON` - Request body is not valid JSON
- `MISSING_FIELD` - Required field is missing
- `MODEL_ERROR` - PlantNet inference failed
- `LLM_ERROR` - Ollama/LLM error
- `INTERNAL_ERROR` - Unexpected server error

**Frontend should:**
```python
response = requests.post(...)
if response.status_code != 200:
    st.error(f"Error: {response.json()['error']}")
```

---

## Rate Limits & Performance

- **No rate limits** (local API)
- **Expected latency:**
  - `/identify`: 10-30 seconds (ResNet152 on CPU)
  - `/ask_follow_ups`: 2-5 seconds (LLM inference)
  - `/diagnose`: 5-15 seconds (LLM reasoning)

**Frontend should show loading indicators:**
```python
with st.spinner("🤔 AI is thinking..."):
    response = requests.post(...)
```

---

## Testing the API

**Person B:** Before the UI is ready, test endpoints with curl:

```bash
# Test /identify
curl -X POST http://localhost:5000/identify \
  -F "image=@test-image.jpeg"

# Test /ask_follow_ups
curl -X POST http://localhost:5000/ask_follow_ups \
  -H "Content-Type: application/json" \
  -d '{
    "species": "Alocasia macrorrhizos",
    "diagnosis": {"issue": "root rot", "confidence": 0.75}
  }'

# Test /diagnose
curl -X POST http://localhost:5000/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "species": "Alocasia macrorrhizos",
    "symptoms": "brown spots, wilting",
    "conditions": {"humidity": "low"},
    "user_answers": {"q1": "a1"}
  }'
```

---

## Changes to This Contract

**If endpoints change, both people must agree and update this document.**

Examples:
- Adding a new required field
- Changing response format
- Changing endpoint URL
- Adding error codes

**Process:**
1. Propose change in Slack/Discord
2. Get agreement from other person
3. Update this file
4. Both people implement changes

---

## LOCKED FIELDS (DO NOT CHANGE)

These fields are LOCKED and must remain consistent:

- Endpoint URLs: `/identify`, `/ask_follow_ups`, `/diagnose`
- HTTP method: `POST` for all
- Response format: Always `{"success": boolean, ...}`
- Image field in /identify: Always `form-data` with key `image`

---

## Sign-Off

Person A (Backend): _________________________ Date: _______
Person B (Frontend): ________________________ Date: _______

This contract is binding for the duration of this project. Changes require mutual agreement.
