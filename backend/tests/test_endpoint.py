import sys
from pathlib import Path

# Add parent directory to path so we can import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json

# Start Flask app first:
# python app.py

# Test image path - relative to tests directory
test_image_path = Path(__file__).parent.parent / "static" / "test-image.jpeg"

if not test_image_path.exists():
    print(f"❌ Test image not found at {test_image_path}")
    print("   Place a test plant image there first")
else:
    with open(test_image_path, 'rb') as f:
        files = {'image': f}
        
        print("Sending request to /diagnose...")
        try:
            response = requests.post(
                "http://127.0.0.1:5000/diagnose",
                files=files,
                timeout=60
            )
            
            print(f"\nStatus Code: {response.status_code}")
            print("\nResponse:")
            print(json.dumps(response.json(), indent=2))
        except requests.exceptions.ConnectionError:
            print("❌ Connection error. Is the Flask server running?")
            print("   Start it with: python app.py")
        except Exception as e:
            print(f"❌ Error: {e}")

