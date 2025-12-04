"""
Test script for Phase 4: Leaf detection and lesion analysis
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from services.leaf_detector import get_leaf_detector
from services.lesion_analyzer import get_lesion_analyzer

def test_leaf_detection():
    """Test YOLO leaf detection."""
    print("\n" + "="*60)
    print("Testing Leaf Detection (YOLO)")
    print("="*60)
    
    # Load test image
    test_image_path = Path(__file__).parent.parent / "static" / "test-image.jpeg"
    
    if not test_image_path.exists():
        print(f"❌ Test image not found at {test_image_path}")
        return False
    
    try:
        image = Image.open(test_image_path)
        print(f"✅ Loaded test image: {image.size}")
        
        # Get leaf detector
        print("\n🔍 Initializing leaf detector...")
        detector = get_leaf_detector()
        
        if not detector.available:
            print("⚠️  YOLO not available (ultralytics not installed or model not found)")
            print("   Install with: pip install ultralytics")
            return False
        
        # Detect leaves
        print("\n🔍 Detecting leaves...")
        result = detector.detect_leaves(image, confidence_threshold=0.15, return_boxes=True)
        
        print(f"✅ Detected {result['num_leaves']} leaf/leaves")
        
        if result['num_leaves'] > 0:
            for i, (leaf, conf) in enumerate(zip(result['leaves'], result['confidences']), 1):
                print(f"   Leaf {i}: {leaf.size}, confidence: {conf:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lesion_analysis():
    """Test lesion analysis."""
    print("\n" + "="*60)
    print("Testing Lesion Analysis")
    print("="*60)
    
    # Load test image
    test_image_path = Path(__file__).parent.parent / "static" / "test-image.jpeg"
    
    if not test_image_path.exists():
        print(f"❌ Test image not found at {test_image_path}")
        return False
    
    try:
        image = Image.open(test_image_path)
        print(f"✅ Loaded test image: {image.size}")
        
        # Get lesion analyzer
        print("\n🔍 Analyzing leaf for lesions...")
        analyzer = get_lesion_analyzer()
        
        # Analyze
        analysis = analyzer.analyze_leaf(image)
        
        print("\n📊 Analysis Results:")
        print(f"   Health Score: {analysis['health_score']:.3f} (1.0 = perfect)")
        print(f"   Green Percentage: {analysis['green_percentage']:.1f}%")
        print(f"   Lesion Percentage: {analysis['lesion_percentage']:.1f}%")
        print(f"   Potential Issues: {'Yes' if analysis['has_potential_issues'] else 'No'}")
        print(f"   Lesion Regions: {analysis['num_lesion_regions']}")
        
        if analysis['lesion_areas']:
            print("\n   Lesion Areas:")
            for i, area in enumerate(analysis['lesion_areas'][:5], 1):  # Show first 5
                print(f"     {i}. BBox: {area['bbox']}, Area: {area['area']}px")
        
        # Test highlighting
        print("\n🎨 Testing lesion highlighting...")
        highlighted = analyzer.highlight_lesions(image)
        print(f"   ✅ Created highlighted image: {highlighted.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_pipeline():
    """Test integrated leaf detection + lesion analysis."""
    print("\n" + "="*60)
    print("Testing Integrated Pipeline")
    print("="*60)
    
    # Load test image
    test_image_path = Path(__file__).parent.parent / "static" / "test-image.jpeg"
    
    if not test_image_path.exists():
        print(f"❌ Test image not found at {test_image_path}")
        return False
    
    try:
        image = Image.open(test_image_path)
        
        # Step 1: Detect leaves
        detector = get_leaf_detector()
        if not detector.available:
            print("⚠️  Leaf detection not available, using full image")
            leaves = [image]
        else:
            result = detector.detect_leaves(image)
            leaves = result['leaves']
            print(f"✅ Detected {len(leaves)} leaf/leaves")
        
        # Step 2: Analyze each leaf
        analyzer = get_lesion_analyzer()
        
        print("\n📊 Analyzing each leaf:")
        for i, leaf in enumerate(leaves, 1):
            print(f"\n   Leaf {i}:")
            analysis = analyzer.analyze_leaf(leaf)
            print(f"     Health Score: {analysis['health_score']:.3f}")
            print(f"     Green: {analysis['green_percentage']:.1f}%")
            print(f"     Lesions: {analysis['lesion_percentage']:.1f}%")
            print(f"     Issues: {'Yes' if analysis['has_potential_issues'] else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Phase 4 Testing Suite")
    print("="*60)
    
    results = {
        "leaf_detection": False,
        "lesion_analysis": False,
        "integrated_pipeline": False
    }
    
    # Run tests
    results["leaf_detection"] = test_leaf_detection()
    results["lesion_analysis"] = test_lesion_analysis()
    results["integrated_pipeline"] = test_integrated_pipeline()
    
    # Summary
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name.replace('_', ' ').title()}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Total: {passed}/{total} tests passed")
    
    print("\n" + "="*60)

