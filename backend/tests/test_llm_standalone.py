#!/usr/bin/env python3
"""
Standalone LLM test - tests LLM components in isolation
Run this with: python tests/test_llm_standalone.py
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def test_ollama_import():
    """Test if ollama package can be imported."""
    print("\n" + "="*60)
    print("  Test 1: Ollama Package Import")
    print("="*60)
    
    try:
        import ollama
        print("✅ Ollama package imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Ollama package not installed: {e}")
        print("\n💡 Install with:")
        print("   pip install ollama")
        return False

def test_ollama_service():
    """Test if Ollama service is running."""
    print("\n" + "="*60)
    print("  Test 2: Ollama Service Connection")
    print("="*60)
    
    try:
        import ollama
        
        print("ℹ️  Checking Ollama service...")
        
        # Try to list models
        try:
            models = ollama.list()
            print("✅ Ollama service is running!")
            
            # Check for mistral model
            model_names = []
            if hasattr(models, 'models'):
                model_names = [m.name for m in models.models if hasattr(m, 'name')]
            elif isinstance(models, list):
                model_names = [str(m) for m in models]
            elif isinstance(models, dict) and 'models' in models:
                model_names = [m.get('name', '') for m in models['models']]
            
            if model_names:
                print(f"   Available models: {', '.join(model_names[:5])}")
                if any('mistral' in name.lower() for name in model_names):
                    print("   ✅ Mistral model found!")
                    return True
                else:
                    print("   ⚠️  Mistral model not found")
                    print("   💡 Pull it with: ollama pull mistral")
                    return False
            else:
                print("   ⚠️  No models found")
                print("   💡 Pull a model with: ollama pull mistral")
                return False
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Cannot connect to Ollama service: {error_msg}")
            print("\n💡 Make sure Ollama is running:")
            print("   - Start with: ollama serve")
            print("   - Or check if it's running as a service")
            return False
            
    except ImportError:
        print("❌ Ollama package not installed")
        return False

def test_llm_direct():
    """Test LLM directly without wrapper."""
    print("\n" + "="*60)
    print("  Test 3: Direct LLM Generation")
    print("="*60)
    
    try:
        import ollama
        
        print("ℹ️  Testing direct Ollama generation...")
        print("   This may take 10-30 seconds...")
        
        response = ollama.generate(
            model='mistral',
            prompt='Say "OK" if you are working correctly.',
            options={'num_predict': 20}
        )
        
        text = response.get('response', response.get('text', ''))
        
        if text:
            print("✅ LLM generation successful!")
            print(f"\n📝 Response: {text.strip()}")
            return True
        else:
            print("❌ No response received")
            return False
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        
        if 'connection' in error_msg.lower() or 'refused' in error_msg.lower():
            print("\n💡 Ollama service not running")
            print("   Start with: ollama serve")
        elif 'model' in error_msg.lower() and 'not found' in error_msg.lower():
            print("\n💡 Mistral model not found")
            print("   Pull it with: ollama pull mistral")
        else:
            print(f"\n💡 Error details: {error_msg}")
        
        return False

def test_llm_chat():
    """Test LLM chat API."""
    print("\n" + "="*60)
    print("  Test 4: LLM Chat API")
    print("="*60)
    
    try:
        import ollama
        
        print("ℹ️  Testing Ollama chat API...")
        print("   This may take 10-30 seconds...")
        
        # Try chat API
        try:
            response = ollama.chat(
                model='mistral',
                messages=[
                    {'role': 'system', 'content': 'You are a helpful plant expert.'},
                    {'role': 'user', 'content': 'What causes yellow leaves in plants? Give a brief answer.'}
                ],
                options={'num_predict': 100}
            )
            
            text = response.get('message', {}).get('content', '')
            
            if text:
                print("✅ Chat API working!")
                print(f"\n📝 Response (first 200 chars):")
                print(f"   {text[:200]}...")
                return True
            else:
                print("❌ No response received")
                return False
                
        except AttributeError:
            print("⚠️  Chat API not available, trying generate API...")
            return test_llm_direct()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_llm_wrapper():
    """Test our LLM wrapper class."""
    print("\n" + "="*60)
    print("  Test 5: LLM Model Wrapper")
    print("="*60)
    
    # Import config first to avoid circular imports
    try:
        # Read config manually to avoid importing everything
        config_path = backend_dir / "config.py"
        config_vars = {}
        with open(config_path) as f:
            exec(f.read(), config_vars)
        
        LLM_MODEL_NAME = config_vars.get('LLM_MODEL_NAME', 'mistral')
        
        # Now try to import just the LLM model
        # We'll do this more carefully
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "llm_model",
            backend_dir / "models" / "llm_model.py"
        )
        llm_module = importlib.util.module_from_spec(spec)
        
        # Mock config for the import
        import types
        mock_config = types.ModuleType('config')
        mock_config.LLM_MODEL_NAME = LLM_MODEL_NAME
        mock_config.LLM_TEMPERATURE = 0.7
        mock_config.LLM_MAX_TOKENS = 512
        sys.modules['config'] = mock_config
        
        spec.loader.exec_module(llm_module)
        
        print("✅ LLM wrapper module loaded")
        
        llm = llm_module.get_llm_model()
        
        if llm.available:
            print(f"✅ LLM wrapper initialized: {llm.model_name}")
            
            # Test generation
            print("ℹ️  Testing wrapper generation...")
            result = llm.generate("Say 'OK' if working.", max_tokens=20)
            
            if result['success']:
                print("✅ Wrapper generation successful!")
                print(f"   Response: {result['text']}")
                return True
            else:
                print(f"❌ Generation failed: {result.get('error')}")
                return False
        else:
            print("⚠️  LLM wrapper not available")
            return False
            
    except Exception as e:
        print(f"❌ Error loading wrapper: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Phase 5: LLM Testing (Standalone)")
    print("="*60)
    print("\nTesting LLM components in isolation...\n")
    
    results = {}
    
    # Test 1: Package import
    results['ollama_import'] = test_ollama_import()
    
    if results['ollama_import']:
        # Test 2: Service connection
        results['ollama_service'] = test_ollama_service()
        
        if results['ollama_service']:
            # Test 3: Direct generation
            results['direct_generation'] = test_llm_direct()
            
            # Test 4: Chat API
            results['chat_api'] = test_llm_chat()
    
    # Test 5: Wrapper (independent)
    results['wrapper'] = test_llm_wrapper()
    
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
    
    if passed == total:
        print("\n🎉 All LLM tests passed! Phase 5 is ready!")
    else:
        print("\n💡 Setup Instructions:")
        if not results.get('ollama_import'):
            print("   1. Install ollama package: pip install ollama")
        if results.get('ollama_import') and not results.get('ollama_service'):
            print("   2. Start Ollama service: ollama serve")
            print("   3. Pull Mistral model: ollama pull mistral")
    
    print("\n" + "="*60)

