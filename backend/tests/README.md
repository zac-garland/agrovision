# Tests Directory

This directory contains all test files for the AgroVision+ Backend.

## Test Files

- **`test_all.py`** - Comprehensive automated test suite that tests all components:
  - Health endpoint
  - PlantNet model loading
  - Model inference
  - Diagnosis endpoint

- **`test_plantnet.py`** - Direct PlantNet model testing
  - Tests model loading
  - Tests inference on test image
  - Useful for debugging model issues

- **`test_endpoint.py`** - API endpoint testing
  - Tests the `/diagnose` POST endpoint
  - Requires Flask server to be running

- **`test_model.py`** - Legacy test file
  - May reference outdated imports
  - Kept for reference but consider using `test_plantnet.py` instead

## Running Tests

### Run All Tests (Recommended)
```bash
cd backend
python tests/test_all.py
```

### Run Individual Tests
```bash
# Test PlantNet model
python tests/test_plantnet.py

# Test diagnosis endpoint (requires Flask server running)
python tests/test_endpoint.py
```

## Notes

- All test files automatically add the parent directory to `sys.path` for imports
- Test images should be in `../static/test-image.jpeg`
- The Flask server must be running for endpoint tests

