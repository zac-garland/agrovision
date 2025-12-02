import sys
from pathlib import Path

# Add parent directory to path so we can import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.plantnet_model import get_plantnet_model
from PIL import Image
import json

# Load model
print("Loading PlantNet model...")
model = get_plantnet_model()

# Load test image - path relative to tests directory
test_image_path = Path(__file__).parent.parent / "static" / "test-image.jpeg"
try:
    img = Image.open(test_image_path)
    print(f"✅ Test image loaded: {img.size}")
    
    # Run inference
    print("\n🔍 Running inference...")
    results = model.predict(img, top_k=5)
    
    print("\n📊 Results:")
    print(json.dumps(results, indent=2))
    
except FileNotFoundError:
    print(f"❌ Test image not found at {test_image_path}")
    print("   Place a test plant image there first")

