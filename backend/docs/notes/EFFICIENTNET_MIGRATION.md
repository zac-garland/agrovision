# EfficientNet B4 Migration Guide

## Summary

The backend has been updated to use EfficientNet B4 instead of ResNet152 for plant species identification. The new model file `efficientnet_b4_weights_best_acc.tar` should provide better accuracy and efficiency.

## Changes Made

### 1. Configuration (`backend/config.py`)
- Added `EFFICIENTNET_MODEL_PATH` pointing to the new EfficientNet B4 model
- Updated `PLANTNET_MODEL_PATH` to use EfficientNet B4 by default

### 2. Model Loading (`backend/models/plantnet_model.py`)
- Updated `PlantNetModel` class to support EfficientNet B4
- Added support for both torchvision EfficientNet and efficientnet-pytorch library
- Improved checkpoint loading to handle different checkpoint formats
- Changed default from ResNet to EfficientNet

### 3. Model Architecture
- EfficientNet B4 uses a different architecture than ResNet
- Final layer uses `classifier[1]` instead of `fc`
- Same number of classes: 1081 (PlantNet-300K species)

## Usage

The model will automatically load EfficientNet B4 by default:

```python
from models.plantnet_model import get_plantnet_model

# This will use EfficientNet B4 by default
model = get_plantnet_model()

# For legacy ResNet152 (if needed):
model = get_plantnet_model(use_efficientnet=False)
```

## Dependencies

The code tries to use EfficientNet in this order:

1. **efficientnet-pytorch** (if installed): `pip install efficientnet-pytorch`
2. **torchvision** (already in requirements.txt): EfficientNet models available in torchvision 0.13+

Current setup uses torchvision 0.17.0 which includes EfficientNet support.

## Model File Location

The EfficientNet B4 model should be located at:
```
models/efficientnet_b4_weights_best_acc.tar
```

## Testing

To test the new model:

```bash
cd backend
python -c "from models.plantnet_model import get_plantnet_model; m = get_plantnet_model(); print('✅ Model loaded successfully')"
```

## Troubleshooting

### Model not found
- Ensure `efficientnet_b4_weights_best_acc.tar` exists in the `models/` folder

### Import errors
- If you get errors about EfficientNet, try: `pip install efficientnet-pytorch`
- Or ensure torchvision >= 0.13.0 is installed

### Checkpoint loading errors
- The code now handles multiple checkpoint formats automatically
- If loading fails, check the checkpoint structure and update the loading code

## Backward Compatibility

- Old ResNet models are still supported via `use_efficientnet=False`
- Existing API endpoints remain unchanged
- All metadata files (species mappings, common names) remain the same

