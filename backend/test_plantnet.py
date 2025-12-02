from models.plantnet_model import get_plantnet_model
from PIL import Image
import json

# Load model
print("Loading PlantNet model...")
model = get_plantnet_model()

# Load test image
test_image_path = "static/test_image.jpg"
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