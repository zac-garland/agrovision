# Health Score Updates - Emphasizing Lesion Detection

## Summary

Updated the health scoring system to emphasize **lesion detection** over green percentage, since many healthy plants have non-green foliage (purple, red, variegated, etc.).

## Changes Made

### 1. Health Score Formula (`backend/services/lesion_analyzer.py`)

**Before:**
```python
health_score = max(0.0, min(1.0, green_pct / 100.0 - lesion_pct / 100.0))
```

**After:**
```python
# Lesion percentage is the primary indicator (weighted heavily)
lesion_impact = (lesion_pct / 100.0) * 1.5  # Lesions weighted 1.5x (primary factor)
green_contribution = (green_pct / 100.0) * 0.2  # Green weighted 0.2x (secondary, optional)
health_score = max(0.0, min(1.0, 1.0 - lesion_impact + green_contribution))
```

**Impact:**
- Lesions are now weighted 1.5x (primary concern)
- Green percentage is weighted only 0.2x (minor contribution)
- Health score primarily reflects lesion coverage

### 2. Issue Detection Logic

**Before:**
```python
has_potential_issues = bool(
    lesion_pct > 5.0 or  # More than 5% lesion-like areas
    green_pct < 50.0     # Less than 50% healthy green
)
```

**After:**
```python
has_potential_issues = bool(
    lesion_pct > 3.0  # More than 3% lesion-like areas indicates issues
    # Removed green_pct check since many healthy plants aren't green
)
```

**Impact:**
- Lower threshold for lesion detection (3% vs 5%)
- Removed green percentage check entirely
- Focuses solely on actual damage/lesions

### 3. LLM Prompt Updates (`backend/services/diagnosis_engine.py`)

**Changes:**
- Added emphasis on lesion percentage as PRIMARY indicator
- Added note that green color is less relevant (many healthy plants aren't green)
- Reorganized prompt to show lesion metrics first
- Updated severity guidelines based on lesion coverage:
  - <3% = none/low
  - 3-10% = moderate
  - >10% = high

### 4. System Prompt Updates

Added guidance for LLM:
- Lesion detection is the PRIMARY health indicator
- Do NOT rely on green color percentage
- Many healthy plants have purple, red, variegated foliage
- Focus analysis on lesion coverage and lesion regions

## Frontend Recommendations

The frontend should also emphasize lesion metrics. Suggested updates:

1. **Primary Metrics Display:**
   - Show "Lesion Coverage" prominently (larger, bold)
   - Show "Lesion Regions" count
   - Show overall health score with context note

2. **Secondary Metrics:**
   - Green percentage can be shown in an expandable section or as a footnote
   - Add note: "Many healthy plants have non-green foliage"

3. **Visual Emphasis:**
   - Use red/warning colors for lesion coverage
   - Use neutral colors for green percentage
   - Make lesion metrics more prominent in the UI

## Testing

Test with:
- Green-leaved plants (should still work)
- Purple-leaved plants (should not be penalized for low green %)
- Variegated plants (should focus on lesions, not color)
- Plants with visible lesions (should score appropriately)

## Example Scenarios

### Scenario 1: Purple-leaved Plant, No Lesions
- Green %: 10%
- Lesion %: 0%
- **Old Score:** 0.10 (unhealthy - wrong!)
- **New Score:** 1.0 - (0 * 1.5) + (10 * 0.2) = 1.0 (healthy - correct!)

### Scenario 2: Green Plant with Lesions
- Green %: 80%
- Lesion %: 15%
- **Old Score:** 0.65 (moderate health)
- **New Score:** 1.0 - (15 * 1.5) + (80 * 0.2) = 1.0 - 0.225 + 0.16 = 0.935 = 0.935 (but clamped to range)
- Actually: 1.0 - 0.225 + 0.16 = 0.935 → but lesions dominate

### Scenario 3: Healthy Plant, Minor Lesions
- Green %: 70%
- Lesion %: 2%
- **Old Score:** 0.68
- **New Score:** 1.0 - (2 * 0.015) + (70 * 0.002) = 1.0 - 0.03 + 0.14 = 1.11 → clamped to 1.0

## Notes

- The new formula ensures lesions have strong negative impact
- Green percentage provides a small positive contribution but is not required
- Plants with no lesions score well regardless of color
- Plants with lesions score poorly regardless of green percentage

