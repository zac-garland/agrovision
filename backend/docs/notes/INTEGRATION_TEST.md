# Frontend-Backend Integration Testing Guide

## ✅ Backend Response Structure Fixed

The backend now matches frontend expectations:

### Plant Species
- ✅ `plant_species.primary.name` - Now includes scientific name
- ✅ `plant_species.primary.species_name` - Still available
- ✅ `plant_species.primary.common_name` - Available
- ✅ `plant_species.primary.confidence` - Available
- ✅ `plant_species.top_5[]` - Array with `name` field

### Disease Detection
- ✅ `disease_detection.primary.disease` - Condition name
- ✅ `disease_detection.primary.common_name` - Plant common name
- ✅ `disease_detection.primary.confidence` - Confidence score
- ✅ `disease_detection.primary.affected_area_percent` - Calculated from health score
- ✅ `disease_detection.leaf_analysis` - Detailed leaf analysis (new!)

### Final Diagnosis
- ✅ `final_diagnosis.condition` - Condition description
- ✅ `final_diagnosis.confidence` - Confidence score
- ✅ `final_diagnosis.severity` - Severity level
- ✅ `final_diagnosis.reasoning` - LLM reasoning

### Treatment Plan
- ✅ `treatment_plan.immediate` - Immediate actions
- ✅ `treatment_plan.week_1` - Week 1 steps
- ✅ `treatment_plan.week_2_3` - Weeks 2-3 steps
- ✅ `treatment_plan.monitoring` - Monitoring notes

## 🚀 Testing Steps

### Step 1: Start Backend Server

```bash
cd /Users/zacgarland/r_projects/agrovision/backend
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

### Step 2: Test Backend Response Structure

In another terminal:

```bash
cd /Users/zacgarland/r_projects/agrovision/backend
python test_image.py static/test-image2.jpeg
```

Verify the response includes:
- `diagnosis.plant_species.primary.name`
- `diagnosis.disease_detection.primary.disease`
- All expected fields

### Step 3: Start Frontend

In a third terminal:

```bash
cd /Users/zacgarland/r_projects/agrovision/frontend
streamlit run app.py
```

This should open at: `http://localhost:8501`

### Step 4: Test Full Integration

1. Open the Streamlit app in your browser
2. Upload a plant image (use `backend/static/test-image2.jpeg` or any plant image)
3. Click "run diagnosis"
4. Verify all sections display correctly:
   - Plant species with name and confidence
   - Disease detection (if issues found)
   - Final diagnosis with reasoning
   - Treatment plan with steps

## 🔍 Expected Frontend Display

### Plant Species Section
- Should show: "predicted species: **[scientific name]**"
- Should show: "common name: [common name]"
- Should show: "model confidence: [percentage]"
- Should show: Top 5 bar chart

### Disease Detection Section
- Only shows if `disease_detection.primary` is not null
- Shows: "predicted disease: **[condition]**"
- Shows: "model confidence: [percentage]"
- Shows: "estimated affected leaf area: [percentage]%"

### Final Diagnosis Section
- Shows: "condition: **[condition]**"
- Shows: "confidence: [percentage]"
- Shows: "severity: [severity]"
- Expandable: "why this diagnosis" with reasoning

### Treatment Plan Section
- Shows: "**right now**" with immediate actions
- Shows: "**this week**" with week_1 steps
- Shows: "**weeks two and three**" with week_2_3 steps
- Shows: "**monitoring notes**" with monitoring text

## 🐛 Troubleshooting

### Backend not responding
- Check if Flask server is running: `curl http://localhost:5000/health`
- Check for errors in Flask terminal
- Verify port 5000 is not in use

### Frontend can't connect
- Verify BACKEND_URL in `frontend/app.py` is `http://localhost:5000/diagnose`
- Check CORS if accessing from different machine
- Check firewall settings

### Missing fields in frontend
- Check browser console for errors
- Verify backend response structure with test script
- Check Streamlit terminal for errors

### Empty/None values
- Check if Ollama is running (for LLM)
- Verify image is valid
- Check backend logs for errors

## 📝 New Fields Available (Optional Enhancements)

The backend now also returns `disease_detection.leaf_analysis` with:
- `num_leaves_detected`
- `overall_health_score`
- `has_potential_issues`
- `individual_leaves[]` - Per-leaf analysis

You can optionally enhance the frontend to display these!

## ✅ Integration Checklist

- [ ] Backend server starts without errors
- [ ] Backend `/health` endpoint works
- [ ] Backend `/diagnose` endpoint returns valid JSON
- [ ] Response includes all required fields
- [ ] Frontend starts without errors
- [ ] Frontend can connect to backend
- [ ] Image upload works
- [ ] Diagnosis request succeeds
- [ ] Plant species displays correctly
- [ ] Disease detection displays (if present)
- [ ] Final diagnosis displays correctly
- [ ] Treatment plan displays correctly
- [ ] Processing time shows

## 🎯 Next Steps

Once basic integration works:

1. **Add leaf analysis display** - Show health scores, lesion percentages
2. **Add error handling UI** - Better error messages in frontend
3. **Add loading states** - Progress indicators for long requests
4. **Add image preview** - Show highlighted lesions (if backend supports)
5. **Polish UI** - Better styling and layout

