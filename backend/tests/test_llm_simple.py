#!/usr/bin/env python3
"""
Simple LLM test - tests just the LLM components without loading heavy models
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_ollama_import():
    """Test if ollama package can be imported."""
    print("\n" + "="*60)
    print("  Testing Ollama Package Import")
    print("="*60)
    
    try:
        import ollama
        print("✅ Ollama package imported successfully")
        print(f"   Package version: {ollama.__version__ if hasattr(ollama, '__version__') else 'unknown'}")
        return True
    except ImportError as e:
        print(f"❌ Ollama package not installed: {e}")
        print("   Install with: pip install ollama")
        return False

def test_llm_model():
    """Test LLM model wrapper."""
    print("\n" + "="*60)
    print("  Testing LLM Model Wrapper")
    print("="*60)
    
    try:
        from models.llm_model import get_llm_model
        
        print("ℹ️  Initializing LLM model...")
        llm = get_llm_model()
        
        if not llm.available:
            print("⚠️  LLM model not marked as available")
            print("   This is OK if Ollama isn't running yet")
        
        print(f"✅ LLM model wrapper initialized")
        print(f"   Model name: {llm.model_name}")
        print(f"   Base URL: {llm.base_url}")
        
        return True, llm
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_ollama_connection():
    """Test connection to Ollama service."""
    print("\n" + "="*60)
    print("  Testing Ollama Service Connection")
    print("="*60)
    
    try:
        import ollama
        
        print("ℹ️  Attempting to connect to Ollama...")
        
        # Try to list models (this tests connection)
        try:
            models = ollama.list()
            print("✅ Connected to Ollama service!")
            
            if models and hasattr(models, 'models'):
                model_list = models.models
                print(f"   Found {len(model_list)} model(s):")
                for model in model_list[:5]:  # Show first 5
                    name = model.name if hasattr(model, 'name') else str(model)
                    print(f"     - {name}")
            else:
                print("   No models found. Pull a model with: ollama pull mistral")
            
            return True
            
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("   Make sure Ollama is running:")
            print("   - Start with: ollama serve")
            print("   - Or check if it's running as a service")
            return False
            
    except ImportError:
        print("❌ Ollama package not installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_llm_generation():
    """Test LLM text generation."""
    print("\n" + "="*60)
    print("  Testing LLM Text Generation")
    print("="*60)
    
    try:
        from models.llm_model import get_llm_model
        
        llm = get_llm_model()
        
        print("ℹ️  Testing simple generation...")
        result = llm.generate(
            "Say 'Hello, I am working!' if you can hear me.",
            max_tokens=20
        )
        
        if result['success']:
            print("✅ LLM generation successful!")
            print(f"\n📝 Response:")
            print(f"   {result['text']}")
            return True
        else:
            print(f"❌ Generation failed: {result.get('error')}")
            if 'connection' in result.get('error', '').lower():
                print("   Make sure Ollama service is running: ollama serve")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_diagnosis_engine_simple():
    """Test diagnosis engine without full model loading."""
    print("\n" + "="*60)
    print("  Testing Diagnosis Engine (Simple)")
    print("="*60)
    
    try:
        # Import just the diagnosis engine
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from services.diagnosis_engine import get_diagnosis_engine
        
        engine = get_diagnosis_engine()
        
        # Mock data
        plant_species = {
            'common_name': 'Test Plant',
            'species_name': 'Testus plantus',
            'confidence': 0.85
        }
        
        leaf_analysis = {
            'num_leaves_detected': 1,
            'overall_health_score': 0.65,
            'has_potential_issues': True,
            'individual_leaves': [{
                'health_score': 0.65,
                'green_percentage': 65.0,
                'lesion_percentage': 8.5
            }]
        }
        
        print("ℹ️  Testing rule-based diagnosis...")
        result = engine.synthesize_diagnosis(
            plant_species=plant_species,
            leaf_analysis=leaf_analysis,
            use_llm=False
        )
        
        print("✅ Diagnosis engine working!")
        print(f"   Source: {result.get('source')}")
        print(f"   Condition: {result['final_diagnosis']['condition']}")
        print(f"   Severity: {result['final_diagnosis']['severity']}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Phase 5 Simple Tests")
    print("="*60)
    print("\nTesting LLM components without loading heavy models...")
    
    results = {}
    
    # Test imports
    results['ollama_import'] = test_ollama_import()
    
    # Test model wrapper
    model_ok, llm = test_llm_model()
    results['llm_wrapper'] = model_ok
    
    # Test connection
    results['ollama_connection'] = test_ollama_connection()
    
    # Test generation
    if results['ollama_connection']:
        results['llm_generation'] = test_llm_generation()
    
    # Test diagnosis engine
    results['diagnosis_engine'] = test_diagnosis_engine_simple()
    
    # Summary
    print("\n" + "="*60)
    print("  Test Results Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name.replace('_', ' ').title()}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if not results.get('ollama_connection'):
        print("\n💡 To enable LLM features:")
        print("   1. Install Ollama: curl https://ollama.ai/install.sh | sh")
        print("   2. Pull model: ollama pull mistral")
        print("   3. Start service: ollama serve")
    
    print("\n" + "="*60)

