#!/bin/bash
# Script to run fine-tuning with caffeinate to prevent sleep
# Usage: ./run_finetune.sh

echo "🚀 Starting fine-tuning with caffeinate..."
echo "   This will keep your Mac awake during training"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
    echo "📦 Activating virtual environment..."
    source ../venv/bin/activate
elif [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Check GPU availability (CUDA or MPS)
echo "🔍 Checking GPU availability..."
python3 check_gpu.py

# Run fine-tuning with caffeinate
# -d: Prevent display from sleeping
# -i: Prevent system from idle sleeping
# -m: Prevent disk from idle sleeping
echo ""
echo "⚡ Starting fine-tuning (will keep system awake)..."
echo "   Press Ctrl+C to stop training"
echo ""

caffeinate -d -i -m python3 finetune_houseplants.py

# Deactivate virtual environment if it was activated
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

echo ""
echo "✅ Fine-tuning complete!"

