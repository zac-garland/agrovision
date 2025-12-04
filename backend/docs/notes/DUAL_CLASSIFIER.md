# Dual Classifier System

## Overview

The dual classifier system uses both the fine-tuned houseplant model and the general PlantNet model, automatically selecting the prediction with the highest confidence score. This provides better accuracy for both houseplants and general plant species.

## How It Works

1. **Both models run inference** on the same image
2. **Compare confidence scores** from each model's top prediction
3. **Select the best result** based on highest confidence
4. **Return unified format** compatible with existing API

## Models Used

### 1. Houseplant Model
- **Path**: `models/efficientnet_b4_houseplant_finetuned.tar`
- **Classes**: 47 houseplant species
- **Accuracy**: ~80% on validation set
- **Best for**: Common houseplants (Snake Plant, Monstera, Pothos, etc.)

### 2. PlantNet Model (General)
- **Path**: `models/efficientnet_b4_weights_best_acc.tar`
- **Classes**: 1081 plant species (PlantNet-300K)
- **Best for**: Outdoor plants, garden plants, wild plants

## Selection Logic

The system selects the model with the **highest confidence** for its top prediction:

```python
if houseplant_confidence > plantnet_confidence:
    use_houseplant_model()
else:
    use_plantnet_model()
```

## Benefits

1. **Automatic selection** - No need to manually choose which model
2. **Best of both worlds** - Houseplants get specialized model, others get general model
3. **Higher accuracy** - Uses the model most confident for each image
4. **Transparent** - Response includes which model was selected and confidence scores

## API Response

The diagnosis response includes dual classifier metadata:

```json
{
  "diagnosis": {
    "plant_species": {
      "primary": {...},
      "top_5": [...]
    },
    ...
    "metadata": {
      "model_versions": {
        "plant_classification": "Dual Classifier (Houseplant + PlantNet)",
        "selected_model": "houseplant",
        "plantnet": "EfficientNet B4",
        "houseplant": "EfficientNet B4 (Fine-tuned)"
      },
      "dual_classifier": {
        "selected_model": "houseplant",
        "houseplant_confidence": 0.85,
        "plantnet_confidence": 0.42,
        "confidence_difference": 0.43,
        "houseplant_available": true
      }
    }
  }
}
```

## Usage

The dual classifier is automatically used in the `/diagnose` endpoint - no changes needed to API calls. The system:

1. Loads both models on startup
2. Runs inference on both for each image
3. Selects best result automatically
4. Falls back to PlantNet if houseplant model unavailable

## Model Files Required

- ✅ `models/efficientnet_b4_houseplant_finetuned.tar` (fine-tuned model)
- ✅ `models/efficientnet_b4_weights_best_acc.tar` (general PlantNet model)

If houseplant model is missing, the system automatically falls back to PlantNet only.

## Performance

- **Inference time**: ~2x single model (both models run, but selection is instant)
- **Memory**: Both models loaded in memory
- **Accuracy**: Better than either model alone for mixed datasets

## Example Scenarios

### Houseplant Image
- Houseplant model: 85% confidence (Monstera)
- PlantNet model: 42% confidence (Monstera)
- **Selected**: Houseplant model ✅

### Garden Plant Image
- Houseplant model: 12% confidence (Snake Plant - wrong)
- PlantNet model: 78% confidence (Tomato - correct)
- **Selected**: PlantNet model ✅

### Uncertain Image
- Houseplant model: 45% confidence
- PlantNet model: 47% confidence
- **Selected**: PlantNet model (slightly higher) ✅

