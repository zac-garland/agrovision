"""
Legacy test file - may reference old model wrapper.
Kept for reference but may need updates.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Note: This file references 'plantnet_wrapper' which may not exist.
# Consider using test_plantnet.py instead which uses the current model implementation.

try:
    from plantnet_wrapper import get_plantnet_model
    from PIL import Image
    import json

    print("\n" + "="*60)
    print("🧪 Testing PlantNet Model Loading")
    print("="*60)

    # Load model
    print("\n📦 Initializing PlantNet model...")
    try:
        model = get_plantnet_model()
        print("✅ Model initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}\n")
        exit(1)

    # Try to load test image
    test_image_path = Path(__file__).parent.parent / "static" / "test_image.jpg"
    print(f"🖼️  Looking for test image at: {test_image_path}")

    try:
        img = Image.open(test_image_path)
        print(f"✅ Test image loaded: {img.size}, Mode: {img.mode}")
        
        # Convert to RGB if needed
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Run inference
        print("\n🔍 Running inference...")
        results = model.predict(img, top_k=5)
        
        print("✅ Inference complete!\n")
        print("📊 Top 5 Predictions:")
        print("-" * 60)
        
        for i, pred in enumerate(results["top_k"], 1):
            print(f"  {i}. {pred['common_name']}")
            print(f"     Species: {pred['species_name']}")
            print(f"     Confidence: {pred['confidence']*100:.2f}%")
            print()
        
    except FileNotFoundError:
        print(f"⚠️  Test image not found at '{test_image_path}'")
        print("   To test, place a plant image there and run again")
        print("\n   Model is loaded and ready!")
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()

    print("="*60 + "\n")

except ImportError as e:
    print(f"⚠️  Warning: {e}")
    print("   This test file may be outdated. Use test_plantnet.py instead.")

