# Quick Start: Fine-Tuning House Plant Model

## 🚀 Run Fine-Tuning (Overnight with caffeinate)

```bash
cd backend
./run_finetune.sh
```

That's it! The script will:
- ✅ Keep your Mac awake all night
- ✅ Use GPU if available
- ✅ Save best model automatically
- ✅ Log everything to `backend/finetune_log.txt`

## 📋 Before Running

1. **Check GPU** (optional but recommended):
   ```bash
   cd backend
   python3 check_gpu.py
   ```

2. **Verify dataset exists**:
   ```bash
   ls -d ../house_plant_species/* | head -5
   ```

## ⚙️ Configuration

Edit `backend/finetune_houseplants.py` to adjust:

```python
BATCH_SIZE = 32          # Increase for more GPU memory
EPOCHS = 10              # Number of training epochs
FREEZE_BACKBONE = True   # Set False to fine-tune all layers
```

## 📊 Monitor Progress

In a separate terminal:
```bash
tail -f backend/finetune_log.txt
```

## 📁 Output

- **Model**: `models/efficientnet_b4_houseplant_finetuned.tar`
- **Logs**: `backend/finetune_log.txt`

## 🛑 Stop Training

Press `Ctrl+C` in the terminal running the script.

## ⏱️ Expected Time

- **GPU**: ~10-30 minutes per epoch
- **CPU**: Several hours per epoch (not recommended)

## 🔧 Troubleshooting

**No GPU detected?**
- Training will still work on CPU (just slower)
- Check `python3 check_gpu.py` for details

**Out of memory?**
- Reduce `BATCH_SIZE` to 16 or 8
- Set `FREEZE_BACKBONE = True`

**Can't find dataset?**
- Ensure `house_plant_species/` folder is in project root
- Check folder has subfolders with images

## ✅ After Training

The fine-tuned model will be saved automatically. To use it:

1. Update `backend/config.py`:
   ```python
   PLANTNET_MODEL_PATH = MODEL_DIR / "efficientnet_b4_houseplant_finetuned.tar"
   ```

2. Restart backend server

See `FINETUNING_GUIDE.md` for more details!

