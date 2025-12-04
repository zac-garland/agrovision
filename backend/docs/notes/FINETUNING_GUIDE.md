# Fine-Tuning Guide for House Plant Species

This guide explains how to fine-tune the EfficientNet B4 model on the house plant species dataset.

## Overview

The fine-tuning script (`finetune_houseplants.py`) uses transfer learning to adapt the pre-trained EfficientNet B4 model for house plant species classification. It uses the existing model weights as a starting point and fine-tunes on your house plant dataset.

## Prerequisites

1. **Dataset**: Ensure `house_plant_species/` folder exists in the project root with subfolders for each species
2. **GPU**: Recommended for faster training (CUDA-compatible GPU)
3. **Dependencies**: All required packages should be installed in your virtual environment

## Dataset Structure

The dataset should be organized as follows:

```
house_plant_species/
├── African Violet (Saintpaulia ionantha)/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Aloe Vera/
│   ├── image1.jpg
│   └── ...
└── ... (other species folders)
```

## Quick Start

### Option 1: Run with caffeinate (recommended for overnight training)

```bash
cd backend
./run_finetune.sh
```

This script will:
- Keep your Mac awake during training
- Activate the virtual environment
- Check GPU availability
- Run the fine-tuning script

### Option 2: Manual run

```bash
cd backend

# Activate virtual environment
source ../venv/bin/activate

# Run with caffeinate manually
caffeinate -d -i -m python3 finetune_houseplants.py
```

### Option 3: Run without caffeinate

```bash
cd backend
python3 finetune_houseplants.py
```

## Configuration

Edit `finetune_houseplants.py` to adjust training parameters:

```python
BATCH_SIZE = 32          # Increase if you have more GPU memory
LEARNING_RATE = 0.001    # Lower for fine-tuning
EPOCHS = 10              # Number of training epochs
FREEZE_BACKBONE = True   # Set False to fine-tune all layers
```

## Training Options

### Freeze Backbone (Default)
- Only trains the final classifier layer
- Faster training
- Less likely to overfit
- Good for small datasets

### Unfreeze All Layers
- Fine-tunes entire network
- Slower training
- Better accuracy potential
- Requires more data

Change in code:
```python
FREEZE_BACKBONE = False  # Unfreeze everything
```

## Output

### Model Checkpoint
- Saved to: `models/efficientnet_b4_houseplant_finetuned.tar`
- Contains best model based on validation accuracy
- Includes class mappings and training metadata

### Log File
- Saved to: `backend/finetune_log.txt`
- Contains detailed training progress
- Timestamps for each operation

## Monitoring Training

### View Logs in Real-Time

```bash
# In a separate terminal
tail -f backend/finetune_log.txt
```

### Check GPU Usage

```bash
# On macOS with NVIDIA (if applicable)
# Or use Activity Monitor to check Python process
```

## Expected Training Time

- **CPU**: ~hours per epoch (not recommended)
- **GPU**: ~minutes per epoch (recommended)
- **Total**: Depends on dataset size and epochs

## Troubleshooting

### CUDA Not Available
- Ensure PyTorch is installed with CUDA support
- Check GPU drivers are installed
- Training will still work on CPU but will be much slower

### Out of Memory Errors
- Reduce `BATCH_SIZE` (try 16 or 8)
- Set `FREEZE_BACKBONE = True`
- Close other applications using GPU

### Dataset Not Found
- Verify `house_plant_species/` folder exists in project root
- Check folder permissions
- Ensure at least one image exists in each species folder

### Model Loading Errors
- Ensure `efficientnet_b4_weights_best_acc.tar` exists in `models/`
- Check file permissions
- Script will fall back to ImageNet weights if pretrained model not found

## Using the Fine-Tuned Model

After training completes, the model can be used by updating the model path in `config.py`:

```python
PLANTNET_MODEL_PATH = MODEL_DIR / "efficientnet_b4_houseplant_finetuned.tar"
```

Then restart the backend server to use the fine-tuned model.

## Best Practices

1. **Start with frozen backbone** - Faster and less prone to overfitting
2. **Monitor validation accuracy** - Stop if it stops improving
3. **Save checkpoints** - Script automatically saves best model
4. **Use GPU** - Dramatically faster training
5. **Augment data** - Script includes data augmentation automatically
6. **Train/Val split** - Automatically splits 80/20

## Next Steps

1. Evaluate the fine-tuned model on test images
2. Compare accuracy with base model
3. Iterate: adjust hyperparameters if needed
4. Deploy: Update backend to use fine-tuned model

