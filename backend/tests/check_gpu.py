#!/usr/bin/env python3
"""Quick script to check GPU availability for fine-tuning."""

import torch
import platform

print("="*60)
print("GPU Availability Check")
print("="*60)
print(f"Platform: {platform.system()} {platform.machine()}")
print(f"PyTorch Version: {torch.__version__}")
print()

# Check CUDA (NVIDIA GPUs)
if torch.cuda.is_available():
    print(f"✅ CUDA is available!")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"   GPU Memory: {gpu_memory:.2f} GB")
    print("\n✅ Ready for CUDA GPU training!")
# Check MPS (Apple Silicon GPUs)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("✅ Apple Silicon GPU (MPS) is available!")
    print("   Metal Performance Shaders enabled")
    
    # Get system info
    import subprocess
    try:
        chip_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
        print(f"   Chip: {chip_info}")
    except:
        pass
    
    print("\n✅ Ready for MPS GPU training on Apple Silicon!")
else:
    print("❌ No GPU acceleration available")
    print("   Training will run on CPU (very slow)")
    print("\n⚠️  GPU Options:")
    if platform.system() == 'Darwin':
        print("   - Apple Silicon Macs: MPS should be available")
        print("   - Install/upgrade PyTorch with MPS support")
        print("   - Check: torch.backends.mps.is_available()")
    else:
        print("   - NVIDIA GPUs: Install CUDA-enabled PyTorch")
        print("   - Ensure GPU drivers are installed")
        print("   - Check GPU is recognized by system")

print("="*60)

