"""
Test script for Phase 5: LLM Synthesis
Tests LLM model integration, diagnosis engine, and full synthesis pipeline
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from models.llm_model import get_llm_model
from services.diagnosis_engine import get_diagnosis_engine
from models.plantnet_model import get_plantnet_model
from services.leaf_detector import get_leaf_detector
from services.lesion_analyzer import get_lesion_analyzer

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

def test_llm_availability():
    """Test if LLM is available."""
    print_header("TEST 1: LLM Availability Check")
    
    try:
        llm = get_llm_model()
        
        if not llm.available:
            print_warning("LLM model not available")
            print_info("Make sure Ollama is installed and running")
            print_info("Install: curl https://ollama.ai/install.sh | sh")
            print_info("Start: ollama serve")
            print_info("Pull model: ollama pull mistral")
            return False, None
        
        print_success(f"LLM model initialized: {llm.model_name}")
        
        # Test basic generation
        print_info("Testing basic LLM generation...")
        result = llm.generate("Say 'OK' if you can hear me.", max_tokens=20)
        
        if result['success']:
            print_success(f"LLM is working! Response: {result['text'][:100]}")
            return True, llm
        else:
            print_error(f"LLM generation failed: {result.get('error')}")
            print_info("Make sure Ollama service is running: ollama serve")
            return False, None
            
    except Exception as e:
        print_error(f"Error testing LLM: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_llm_diagnosis_generation():
    """Test LLM diagnosis generation."""
    print_header("TEST 2: LLM Diagnosis Generation")
    
    try:
        llm = get_llm_model()
        
        if not llm.available:
            print_warning("LLM not available, skipping test")
            return False
        
        # Test prompt similar to what diagnosis engine uses
        test_prompt = """Analyze the following plant health data:

PLANT INFORMATION:
- Species: Monstera deliciosa (Swiss Cheese Plant)
- Confidence: 85%

HEALTH ANALYSIS:
- Overall Health Score: 0.65 (1.0 = perfect health)
- Leaves Analyzed: 1
- Potential Issues Detected: Yes

Provide a brief diagnosis (2-3 sentences)."""
        
        print_info("Generating diagnosis with LLM...")
        result = llm.generate(
            prompt=test_prompt,
            system_prompt="You are an expert plant pathologist.",
            max_tokens=200
        )
        
        if result['success']:
            print_success("LLM diagnosis generation successful!")
            print(f"\n📝 Generated text:")
            print("-" * 60)
            print(result['text'])
            print("-" * 60)
            return True
        else:
            print_error(f"Generation failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_diagnosis_engine():
    """Test the diagnosis engine."""
    print_header("TEST 3: Diagnosis Engine")
    
    try:
        engine = get_diagnosis_engine()
        
        # Create mock data
        plant_species = {
            'common_name': 'Monstera deliciosa',
            'species_name': 'Monstera deliciosa',
            'confidence': 0.85
        }
        
        leaf_analysis = {
            'num_leaves_detected': 1,
            'overall_health_score': 0.65,
            'has_potential_issues': True,
            'individual_leaves': [{
                'leaf_index': 1,
                'health_score': 0.65,
                'green_percentage': 65.0,
                'lesion_percentage': 8.5,
                'num_lesion_regions': 3,
                'has_potential_issues': True
            }]
        }
        
        print_info("Testing rule-based diagnosis...")
        result = engine.synthesize_diagnosis(
            plant_species=plant_species,
            leaf_analysis=leaf_analysis,
            use_llm=False  # Test rule-based first
        )
        
        print_success(f"Diagnosis generated! Source: {result.get('source')}")
        print(f"\n📋 Final Diagnosis:")
        print(f"   Condition: {result['final_diagnosis']['condition']}")
        print(f"   Severity: {result['final_diagnosis']['severity']}")
        print(f"   Confidence: {result['final_diagnosis']['confidence']}")
        print(f"\n💡 Reasoning:")
        print(f"   {result['final_diagnosis']['reasoning'][:200]}...")
        print(f"\n📝 Treatment Plan:")
        print(f"   Immediate: {len(result['treatment_plan']['immediate'])} actions")
        print(f"   Week 1: {len(result['treatment_plan']['week_1'])} steps")
        print(f"   Weeks 2-3: {len(result['treatment_plan']['week_2_3'])} steps")
        
        return True
        
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_diagnosis_engine_with_llm():
    """Test diagnosis engine with LLM."""
    print_header("TEST 4: Diagnosis Engine with LLM")
    
    try:
        # Check if LLM is available
        llm = get_llm_model()
        if not llm.available:
            print_warning("LLM not available, skipping LLM test")
            return False
        
        engine = get_diagnosis_engine()
        
        # Create mock data
        plant_species = {
            'common_name': 'Tomato',
            'species_name': 'Solanum lycopersicum',
            'confidence': 0.92
        }
        
        leaf_analysis = {
            'num_leaves_detected': 2,
            'overall_health_score': 0.55,
            'has_potential_issues': True,
            'individual_leaves': [
                {
                    'leaf_index': 1,
                    'health_score': 0.60,
                    'green_percentage': 70.0,
                    'lesion_percentage': 12.0,
                    'num_lesion_regions': 5,
                    'has_potential_issues': True
                },
                {
                    'leaf_index': 2,
                    'health_score': 0.50,
                    'green_percentage': 60.0,
                    'lesion_percentage': 18.0,
                    'num_lesion_regions': 7,
                    'has_potential_issues': True
                }
            ]
        }
        
        print_info("Testing LLM-powered diagnosis (this may take 10-30 seconds)...")
        result = engine.synthesize_diagnosis(
            plant_species=plant_species,
            leaf_analysis=leaf_analysis,
            use_llm=True
        )
        
        print_success(f"LLM diagnosis generated! Source: {result.get('source')}")
        print(f"\n📋 Final Diagnosis:")
        print(f"   Condition: {result['final_diagnosis']['condition']}")
        print(f"   Severity: {result['final_diagnosis']['severity']}")
        print(f"   Confidence: {result['final_diagnosis']['confidence']}")
        print(f"\n💡 Reasoning (first 300 chars):")
        print(f"   {result['final_diagnosis']['reasoning'][:300]}...")
        print(f"\n📝 Treatment Plan:")
        if result['treatment_plan']['immediate']:
            print(f"   Immediate ({len(result['treatment_plan']['immediate'])} actions):")
            for i, action in enumerate(result['treatment_plan']['immediate'][:3], 1):
                print(f"     {i}. {action[:80]}...")
        if result['treatment_plan']['week_1']:
            print(f"   Week 1 ({len(result['treatment_plan']['week_1'])} steps):")
            for i, step in enumerate(result['treatment_plan']['week_1'][:2], 1):
                print(f"     {i}. {step[:80]}...")
        
        return True
        
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test full pipeline with real image."""
    print_header("TEST 5: Full Pipeline Integration")
    
    test_image_path = Path(__file__).parent.parent / "static" / "test-image2.jpeg"
    
    if not test_image_path.exists():
        print_warning(f"Test image not found at {test_image_path}")
        print_info("Skipping full pipeline test")
        return False
    
    try:
        print_info("Loading test image...")
        image = Image.open(test_image_path)
        print_success(f"Image loaded: {image.size}")
        
        # Step 1: Plant identification
        print_info("\nStep 1: Plant identification...")
        plantnet = get_plantnet_model()
        plant_results = plantnet.predict(image, top_k=5)
        primary_plant = plant_results["primary"]
        print_success(f"Plant identified: {primary_plant.get('common_name')}")
        
        # Step 2: Leaf detection
        print_info("\nStep 2: Leaf detection...")
        leaf_detector = get_leaf_detector()
        leaf_detection = leaf_detector.detect_leaves(image, return_boxes=True)
        print_success(f"Detected {leaf_detection['num_leaves']} leaf/leaves")
        
        # Step 3: Lesion analysis
        print_info("\nStep 3: Lesion analysis...")
        lesion_analyzer = get_lesion_analyzer()
        leaf_analyses = []
        overall_health_score = 0.0
        
        for leaf_image in leaf_detection['leaves']:
            analysis = lesion_analyzer.analyze_leaf(leaf_image)
            leaf_analyses.append({
                'leaf_index': len(leaf_analyses) + 1,
                'health_score': analysis['health_score'],
                'green_percentage': analysis['green_percentage'],
                'lesion_percentage': analysis['lesion_percentage'],
                'num_lesion_regions': analysis['num_lesion_regions'],
                'has_potential_issues': analysis['has_potential_issues']
            })
            overall_health_score += analysis['health_score']
        
        if leaf_analyses:
            overall_health_score = overall_health_score / len(leaf_analyses)
        
        print_success(f"Health score: {overall_health_score:.3f}")
        
        # Step 4: LLM Synthesis
        print_info("\nStep 4: LLM synthesis...")
        diagnosis_engine = get_diagnosis_engine()
        
        leaf_analysis_data = {
            'num_leaves_detected': leaf_detection['num_leaves'],
            'overall_health_score': overall_health_score,
            'has_potential_issues': any(a['has_potential_issues'] for a in leaf_analyses),
            'individual_leaves': leaf_analyses
        }
        
        # Try LLM first, fallback to rule-based
        llm_available = get_llm_model().available
        print_info(f"LLM available: {llm_available}")
        
        result = diagnosis_engine.synthesize_diagnosis(
            plant_species=primary_plant,
            leaf_analysis=leaf_analysis_data,
            use_llm=llm_available
        )
        
        print_success(f"Diagnosis synthesized! Source: {result.get('source')}")
        print(f"\n📋 Diagnosis Summary:")
        print(f"   Condition: {result['final_diagnosis']['condition']}")
        print(f"   Severity: {result['final_diagnosis']['severity']}")
        print(f"   Treatment steps: {len(result['treatment_plan']['immediate']) + len(result['treatment_plan']['week_1'])}")
        
        return True
        
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 5 tests."""
    print_header("Phase 5 Testing Suite: LLM Synthesis")
    print_info("This script will test LLM integration and diagnosis synthesis")
    
    results = {
        "llm_availability": False,
        "llm_generation": False,
        "diagnosis_engine": False,
        "diagnosis_engine_llm": False,
        "full_pipeline": False
    }
    
    # Test 1: LLM Availability
    llm_available, llm = test_llm_availability()
    results["llm_availability"] = llm_available
    
    if llm_available:
        # Test 2: LLM Generation
        results["llm_generation"] = test_llm_diagnosis_generation()
    
    # Test 3: Diagnosis Engine (Rule-based)
    results["diagnosis_engine"] = test_diagnosis_engine()
    
    # Test 4: Diagnosis Engine with LLM
    if llm_available:
        results["diagnosis_engine_llm"] = test_diagnosis_engine_with_llm()
    
    # Test 5: Full Pipeline
    results["full_pipeline"] = test_full_pipeline()
    
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
        
        if not results["llm_availability"]:
            print_info("\n💡 To enable LLM tests:")
            print_info("   1. Install Ollama: curl https://ollama.ai/install.sh | sh")
            print_info("   2. Pull model: ollama pull mistral")
            print_info("   3. Start service: ollama serve")
    
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

