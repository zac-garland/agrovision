# Frontend: Plant Diagnosis UI

**Your job:** Build the Streamlit interface that calls the backend API.

---

## What You're Building

```
User → Upload Image → Ask Questions → View Diagnosis
   ↓        ↓              ↓              ↓
   | [Streamlit UI]       ↓              |
   └─────→ [Flask Backend API] ←─────────┘
           (Person A's job)
```

**Three main screens:**
1. **Plant ID Screen** - Upload image, see plant species
2. **Q&A Screen** - Answer clarification questions (if needed)
3. **Results Screen** - View diagnosis, recommendations, explanation

---

## Project Structure

```
frontend/
├── app.py                           # Main Streamlit app (YOU IMPLEMENT)
├── pages/                           # Streamlit pages (optional)
│   ├── about.py
│   └── faq.py
├── components/                      # Reusable UI components
│   ├── image_uploader.py
│   ├── results_display.py
│   └── question_form.py
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## Setup

### 1. Install Dependencies

```bash
cd frontend
pip install -r requirements.txt
```

### 2. Make Sure Backend is Running

Before starting Streamlit, Person A must have Flask API running:

```bash
# In Person A's terminal:
cd backend
python main.py
# Should see: "Running on http://localhost:5000"
```

### 3. Run Streamlit

```bash
cd frontend
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Development Checklist

### Phase 1: Basic App Structure (30 min)

**File:** `app.py`

```python
import streamlit as st
import requests

st.set_page_config(
    page_title="🌱 Plant Diagnosis",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 AgroVision+ Plant Diagnosis")
st.write("Upload a photo of your plant, and our AI will diagnose the issue.")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1  # 1=upload, 2=questions, 3=results
if 'species' not in st.session_state:
    st.session_state.species = None

# Step 1: Upload Image
st.header("Step 1: Upload Your Plant Photo")
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Display image
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Your Plant")
    
    # Identify plant
    with col2:
        st.write("🔍 Identifying plant species...")
        # Call backend /identify endpoint (implement next)
```

---

### Phase 2: Image Upload + Plant ID (1 hour)

**File:** `app.py` (expand)

```python
import streamlit as st
import requests
from PIL import Image
import io

API_BASE = "http://localhost:5000"

st.title("🌱 AgroVision+ Plant Diagnosis")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'identify_result' not in st.session_state:
    st.session_state.identify_result = None

# STEP 1: Upload Image
st.header("Step 1: Upload Your Plant Photo")
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Your Plant")
    
    with col2:
        if st.button("🔍 Identify Plant", key="identify_btn"):
            with st.spinner("AI is analyzing your plant..."):
                try:
                    # Send image to backend
                    files = {'image': uploaded_file.getvalue()}
                    response = requests.post(
                        f"{API_BASE}/identify",
                        files={'image': uploaded_file},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result['success']:
                            st.session_state.identify_result = result
                            st.session_state.step = 2
                            st.rerun()
                        else:
                            st.error(f"Error: {result['error']}")
                    else:
                        st.error(f"Backend error: {response.status_code}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Is Flask running on port 5000?")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# STEP 2: Show Plant Identification
if st.session_state.step >= 2 and st.session_state.identify_result:
    st.divider()
    
    result = st.session_state.identify_result
    
    st.header("Step 2: Plant Identified")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Plant Species", result['species'], 
                  f"{result['confidence']:.0%} confident")
    
    with col2:
        st.write("**Top 5 Candidates:**")
        for i, pred in enumerate(result['top_5'][:5], 1):
            st.write(f"{i}. {pred['name']} ({pred['confidence']:.1%})")
    
    with col3:
        if st.button("🔄 Try Another Photo", key="try_again"):
            st.session_state.step = 1
            st.session_state.identify_result = None
            st.rerun()
```

---

### Phase 3: Follow-Up Questions (1 hour)

**File:** `app.py` (expand with questions)

```python
# STEP 3: Ask Follow-Up Questions (if needed)
if st.session_state.step >= 2:
    st.divider()
    
    if st.button("📝 Get Diagnosis", key="get_diagnosis_btn"):
        with st.spinner("AI is analyzing..."):
            # First, check if we need follow-up questions
            response = requests.post(
                f"{API_BASE}/ask_follow_ups",
                json={
                    "species": st.session_state.identify_result['species'],
                    "diagnosis": {
                        "issue": "unknown",
                        "confidence": st.session_state.identify_result['confidence']
                    }
                }
            )
            
            if response.status_code == 200:
                qu_result = response.json()
                
                if qu_result['success'] and qu_result['should_ask_follow_ups']:
                    st.session_state.step = 3
                    st.session_state.questions = qu_result['questions']
                    st.session_state.ask_questions = True
                    st.rerun()
                else:
                    # Skip to diagnosis
                    st.session_state.step = 4
                    st.session_state.ask_questions = False
                    st.rerun()

# STEP 4: Answer Questions
if st.session_state.step >= 3 and st.session_state.get('ask_questions'):
    st.divider()
    
    st.header("Step 3: Tell Us More")
    st.write("A few questions to better understand your plant's situation:")
    
    answers = {}
    for q in st.session_state.questions:
        answers[q] = st.text_input(q, key=f"q_{q}")
    
    if st.button("💡 Get Diagnosis", key="diagnose_btn"):
        st.session_state.user_answers = answers
        st.session_state.step = 4
        st.rerun()
```

---

### Phase 4: Diagnosis & Recommendations (1.5 hours)

**File:** `app.py` (expand with diagnosis)

```python
# STEP 5: Show Diagnosis
if st.session_state.step >= 4:
    st.divider()
    
    st.header("Step 4: Diagnosis & Recommendations")
    
    with st.spinner("Getting AI diagnosis..."):
        try:
            # Call /diagnose endpoint
            payload = {
                "species": st.session_state.identify_result['species'],
                "symptoms": "to be provided",  # Get from user somehow
                "conditions": {
                    "humidity": "unknown",
                    "temperature": "unknown",
                    "light": "unknown"
                },
                "user_answers": st.session_state.get('user_answers', {})
            }
            
            response = requests.post(
                f"{API_BASE}/diagnose",
                json=payload
            )
            
            if response.status_code == 200:
                diag = response.json()
                
                if diag['success']:
                    # Display diagnosis
                    diagnosis = diag['diagnosis']
                    
                    st.subheader(f"🔍 {diagnosis['primary_issue']}")
                    st.metric("Confidence", f"{diagnosis['confidence']:.0%}")
                    
                    with st.expander("📖 Why This Diagnosis?"):
                        st.write(diagnosis['reasoning'])
                    
                    # Display recommendations
                    st.subheader("💡 What to Do")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Today**")
                        for step in diag['recommendations']['immediate']:
                            st.write(f"✓ {step}")
                    
                    with col2:
                        st.write("**This Week**")
                        for step in diag['recommendations']['short_term'][:3]:
                            st.write(f"✓ {step}")
                    
                    with col3:
                        st.write("**Going Forward**")
                        for step in diag['recommendations']['long_term'][:3]:
                            st.write(f"✓ {step}")
                    
                    # Timeline
                    st.divider()
                    timeline = diag['timeline']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Improvement", timeline['expected_improvement'])
                    with col2:
                        st.metric("Full Recovery", timeline['full_recovery'])
                    with col3:
                        st.metric("When to Escalate", timeline['when_to_escalate'])
                    
                    # Start Over
                    if st.button("🌱 Diagnose Another Plant"):
                        st.session_state.step = 1
                        st.session_state.identify_result = None
                        st.rerun()
                
                else:
                    st.error(f"Error: {diag['error']}")
        
        except Exception as e:
            st.error(f"Error getting diagnosis: {str(e)}")
```

---

### Phase 5: Polish & Styling (1 hour)

**File:** `app.py` (final touches)

```python
# Add at the top
st.set_page_config(
    page_title="🌱 AgroVision+ Plant Diagnosis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .big-title {
        font-size: 3em;
        font-weight: bold;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 1em;
    }
    .status-good {
        color: #2ecc71;
    }
    .status-warning {
        color: #f39c12;
    }
    .status-danger {
        color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("About")
    st.write("""
    **AgroVision+** uses AI to diagnose plant health issues.
    
    1. Upload a photo
    2. Answer a few questions (if needed)
    3. Get personalized care recommendations
    
    **Powered by:**
    - PlantNet-300K (plant identification)
    - Mistral 7B (reasoning)
    - Local processing (no data sharing)
    """)
    
    st.divider()
    st.subheader("How Confident is the AI?")
    st.info("""
    - 🟢 **>80%**: High confidence
    - 🟡 **50-80%**: Medium (we'll ask questions)
    - 🔴 **<50%**: Low (escalate to botanist)
    """)
```

---

## Testing Backend Connection

Before building UI, test that backend works:

```python
import requests

try:
    response = requests.post(
        'http://localhost:5000/identify',
        files={'image': open('test-image.jpeg', 'rb')}
    )
    print(response.json())
except Exception as e:
    print(f"Error: {e}")
    print("Is Flask running? Run: cd backend && python main.py")
```

---

## Deployment Notes

When ready to deploy:

```bash
# Production Streamlit
streamlit run app.py --logger.level=error

# With Gunicorn backend
gunicorn --workers 4 backend.main:app
```

---

## Timeline

```
9:00 AM  - Setup, basic structure
10:00 AM - Image upload + plant ID screen
11:00 AM - Questions form
12:00 PM - Diagnosis display
1:00 PM  - Polish, test with backend
```

---

## Resources

- Streamlit docs: https://docs.streamlit.io/
- Requests library: https://requests.readthedocs.io/
- API Contract: `docs/API_CONTRACT.md`

---

**Let's build it.** 🌱
