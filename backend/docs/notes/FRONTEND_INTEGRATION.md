# Frontend Integration Checklist

## Issues Found

### 1. Plant Species Field Mismatch
- **Frontend expects:** `primary.name`
- **Backend returns:** `primary.species_name`
- **Fix needed:** Add `name` field to match frontend expectations

### 2. Top 5 Species Field Mismatch
- **Frontend expects:** `top_5[].name`
- **Backend returns:** `top_k[].species_name`
- **Fix needed:** Add `name` field to each item in top_5

### 3. Disease Detection Structure
- **Frontend expects:** `disease_detection.primary.disease`
- **Backend returns:** Different structure
- **Fix needed:** Check and align structure

### 4. Treatment Plan Week 2-3
- **Frontend expects:** `week_2_3`
- **Backend returns:** `week_2_3` ✅ (should match)

## Action Plan

1. Update backend response to include `name` field
2. Ensure disease_detection structure matches
3. Test end-to-end integration
4. Update frontend if needed for new fields (leaf_analysis)

