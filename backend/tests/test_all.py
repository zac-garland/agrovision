#!/usr/bin/env python3
"""
Comprehensive test script for AgroVision+ Backend
Tests all components: health endpoint, model loading, and diagnosis endpoint
"""

import sys
import time
from pathlib import Path
import json

# Add parent directory to path so we can import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_success(text):
    """Print success message."""
    print(f"✅ {text}")

def print_error(text):
    """Print error message."""
    print(f"❌ {text}")

def print_warning(text):
    """Print warning message."""
    print(f"⚠️  {text}")

def print_info(text):
    """Print info message."""
    print(f"ℹ️  {text}")

def test_health_endpoint():
    """Test the health endpoint."""
    print_header("TEST 1: Health Endpoint")
    
    try:
        import requests
        
        print_info("Testing GET /health...")
        response = requests.get("http://127.0.0.1:5000/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Health endpoint working! Status: {data.get('status')}")
            print(f"   Service: {data.get('service')}")
            print(f"   Version: {data.get('version')}")
            return True
        else:
            print_error(f"Health endpoint returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to server. Is Flask running?")
        print_info("Start the server with: python app.py")
        return False
    except Exception as e:
        print_error(f"Error testing health endpoint: {e}")
        return False

def test_model_loading():
    """Test PlantNet model loading."""
    print_header("TEST 2: PlantNet Model Loading")
    
    try:
        print_info("Importing PlantNet model...")
        from models.plantnet_model import get_plantnet_model
        
        print_info("Loading model (this may take 10-30 seconds)...")
        start_time = time.time()
        model = get_plantnet_model()
        load_time = time.time() - start_time
        
        print_success(f"Model loaded successfully! ({load_time:.2f}s)")
        print(f"   Device: {model.device}")
        print(f"   Model type: {'ResNet18' if model.use_resnet18 else 'ResNet152'}")
        
        if model.species_id_to_name:
            print(f"   Species loaded: {len(model.species_id_to_name)}")
        if model.common_names_map:
            print(f"   Common names loaded: {len(model.common_names_map)}")
        
        return True, model
        
    except Exception as e:
        print_error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_model_inference(model):
    """Test PlantNet model inference."""
    print_header("TEST 3: PlantNet Model Inference")
    
    # Path relative to tests directory
    test_image_path = Path(__file__).parent.parent / "static" / "test-image.jpeg"
    
    if not test_image_path.exists():
        print_warning(f"Test image not found at {test_image_path}")
        print_info("Skipping inference test. Place a test image to enable this test.")
        return False
    
    try:
        from PIL import Image
        
        print_info(f"Loading test image from {test_image_path}...")
        img = Image.open(test_image_path)
        print_success(f"Image loaded: {img.size}, Mode: {img.mode}")
        
        print_info("Running inference...")
        start_time = time.time()
        results = model.predict(img, top_k=5)
        inference_time = time.time() - start_time
        
        print_success(f"Inference complete! ({inference_time:.2f}s)")
        
        if results.get("primary"):
            primary = results["primary"]
            print(f"\n   Primary prediction:")
            print(f"   - Common name: {primary.get('common_name', 'N/A')}")
            print(f"   - Scientific name: {primary.get('species_name', 'N/A')}")
            print(f"   - Confidence: {primary.get('confidence', 0)*100:.2f}%")
        
        if results.get("top_k"):
            print(f"\n   Top 5 predictions:")
            for i, pred in enumerate(results["top_k"][:5], 1):
                print(f"   {i}. {pred.get('common_name', 'N/A')} "
                      f"({pred.get('confidence', 0)*100:.2f}%)")
        
        return True
        
    except Exception as e:
        print_error(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_diagnosis_endpoint():
    """Test the diagnosis endpoint."""
    print_header("TEST 4: Diagnosis Endpoint")
    
    # Path relative to tests directory
    test_image_path = Path(__file__).parent.parent / "static" / "test-image.jpeg"
    
    if not test_image_path.exists():
        print_warning(f"Test image not found at {test_image_path}")
        print_info("Skipping endpoint test. Place a test image to enable this test.")
        return False
    
    try:
        import requests
        
        print_info("Testing POST /diagnose...")
        
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            print_info("Sending request (this may take a few seconds)...")
            start_time = time.time()
            response = requests.post(
                "http://127.0.0.1:5000/diagnose",
                files=files,
                timeout=60
            )
            request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Diagnosis endpoint working! ({request_time:.2f}s)")
            
            if data.get("success"):
                diagnosis = data.get("diagnosis", {})
                
                # Check plant species
                plant_species = diagnosis.get("plant_species", {})
                if plant_species.get("primary"):
                    primary = plant_species["primary"]
                    print(f"\n   Plant identified:")
                    print(f"   - {primary.get('common_name', 'N/A')}")
                    print(f"   - Confidence: {primary.get('confidence', 0)*100:.2f}%")
                
                # Check metadata
                metadata = diagnosis.get("metadata", {})
                if metadata.get("processing_time_ms"):
                    print(f"   - Processing time: {metadata['processing_time_ms']}ms")
                
                return True
            else:
                print_error(f"Diagnosis returned success=False: {data.get('error')}")
                return False
        else:
            print_error(f"Diagnosis endpoint returned status {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to server. Is Flask running?")
        print_info("Start the server with: python app.py")
        return False
    except Exception as e:
        print_error(f"Error testing diagnosis endpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print_header("AgroVision+ Backend Testing Suite")
    print_info("This script will test all implemented features")
    print_info("Make sure Flask server is running: python app.py")
    
    results = {
        "health_endpoint": False,
        "model_loading": False,
        "model_inference": False,
        "diagnosis_endpoint": False
    }
    
    # Test 1: Health endpoint
    print("\n" + "="*60)
    print("Starting tests...")
    print("="*60)
    
    results["health_endpoint"] = test_health_endpoint()
    
    if not results["health_endpoint"]:
        print_warning("Health endpoint failed. Make sure Flask server is running.")
        print_info("Start server in another terminal: python app.py")
        print_info("Then run this test script again.")
        return
    
    # Test 2: Model loading
    model_loaded, model = test_model_loading()
    results["model_loading"] = model_loaded
    
    if model_loaded and model:
        # Test 3: Model inference
        results["model_inference"] = test_model_inference(model)
    
    # Test 4: Diagnosis endpoint
    results["diagnosis_endpoint"] = test_diagnosis_endpoint()
    
    # Summary
    print_header("Test Results Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name.replace('_', ' ').title()}")
    
    print(f"\n  Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print_success("All tests passed! 🎉")
    else:
        print_warning(f"{total_tests - passed_tests} test(s) failed.")
        print_info("Check the error messages above for details.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

