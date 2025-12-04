# Frontend-Backend Integration Guide

## ✅ Status: Ready for Integration!

The backend has been updated to match frontend expectations. All response fields are now compatible.

## 🚀 Quick Start

### Step 1: Start Backend

```bash
cd backend
python app.py
```

**Expected output:**
```
 * Running on http://127.0.0.1:5000
```

### Step 2: Start Frontend

```bash
cd frontend
streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

### Step 3: Test Integration

1. Open browser to `http://localhost:8501`
2. Upload a plant image
3. Click "run diagnosis"
4. Verify all sections display correctly

## ✅ What's Been Fixed

### Backend Response Structure
- ✅ Added `name` field to `plant_species.primary` (maps to `species_name`)
- ✅ Added `name` field to each item in `top_5` array
- ✅ Formatted `disease_detection.primary` to match frontend expectations:
  - `disease` - Condition name
  - `common_name` - Plant common name
  - `confidence` - Confidence score
  - `affected_area_percent` - Calculated from health score

### All Frontend Fields Now Available
- ✅ `plant_species.primary.name`
- ✅ `plant_species.primary.common_name`
- ✅ `plant_species.primary.confidence`
- ✅ `plant_species.top_5[].name`
- ✅ `disease_detection.primary.disease`
- ✅ `disease_detection.primary.confidence`
- ✅ `disease_detection.primary.affected_area_percent`
- ✅ `final_diagnosis.*` (all fields)
- ✅ `treatment_plan.*` (all fields)

## 🔍 Testing Checklist

- [ ] Backend starts successfully
- [ ] Frontend starts successfully
- [ ] Frontend connects to backend
- [ ] Image upload works
- [ ] Diagnosis request completes
- [ ] Plant species displays
- [ ] Disease detection displays (if issues found)
- [ ] Final diagnosis displays
- [ ] Treatment plan displays
- [ ] Processing time shows

## 🐛 Troubleshooting

### Backend Issues
```bash
# Test backend health
curl http://localhost:5000/health

# Test diagnosis endpoint
cd backend
python test_image.py static/test-image2.jpeg
```

### Frontend Issues
- Check browser console (F12) for errors
- Verify `BACKEND_URL` in `frontend/app.py` is correct
- Check Streamlit terminal for errors

### Connection Issues
- Verify both servers are running
- Check ports: Backend=5000, Frontend=8501
- Check firewall settings if accessing from different machine

## 📊 Expected Response Structure

```json
{
  "success": true,
  "diagnosis": {
    "plant_species": {
      "primary": {
        "name": "Pelargonium zonale (L.) L'Hér. ex Aiton",
        "species_name": "Pelargonium zonale (L.) L'Hér. ex Aiton",
        "common_name": "Pelargonium",
        "confidence": 0.397
      },
      "top_5": [
        {
          "name": "...",
          "species_name": "...",
          "common_name": "...",
          "confidence": 0.397
        }
      ]
    },
    "disease_detection": {
      "primary": {
        "disease": "Minor health concerns detected",
        "common_name": "Pelargonium",
        "confidence": 0.04,
        "affected_area_percent": 4.0
      },
      "leaf_analysis": {
        "num_leaves_detected": 1,
        "overall_health_score": 0.96,
        "has_potential_issues": false
      }
    },
    "final_diagnosis": {
      "condition": "Plant appears healthy",
      "confidence": 0.0,
      "severity": "none",
      "reasoning": "..."
    },
    "treatment_plan": {
      "immediate": [],
      "week_1": ["Continue current care routine", "Monitor for any changes"],
      "week_2_3": ["Maintain regular watering schedule"],
      "monitoring": "..."
    },
    "metadata": {
      "processing_time_ms": 1234,
      "diagnosis_source": "llm"
    }
  }
}
```

## 🎯 Next Steps

Once integration is working:

1. **Test with various images** - Different plants, different conditions
2. **Polish UI** - Add loading states, better error messages
3. **Add leaf analysis display** - Show health scores visually
4. **Performance tuning** - Optimize for slower connections
5. **Error handling** - Better user feedback for failures

## 📝 Notes

- Backend automatically falls back to rule-based if LLM is unavailable
- Processing time typically 2-10 seconds depending on image and LLM availability
- Leaf detection works best with clear leaf images
- LLM provides more detailed reasoning when available

