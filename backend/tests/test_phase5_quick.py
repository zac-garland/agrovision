#!/usr/bin/env python3
"""
Quick Phase 5 test - tests Ollama and LLM integration
Run from backend directory: python tests/test_phase5_quick.py
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_success(text):
    print(f"✅ {text}")

def print_error(text):
    print(f"❌ {text}")

def print_info(text):
    print(f"ℹ️  {text}")

def test_ollama_installed():
    """Check if Ollama CLI is installed."""
    print_header("Test 1: Ollama Installation")
    
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print_success("Ollama CLI is installed")
            print(f"   Version: {result.stdout.strip()}")
            return True
        else:
            print_error("Ollama command failed")
            return False
    except FileNotFoundError:
        print_error("Ollama not found in PATH")
        print_info("Install with: curl https://ollama.ai/install.sh | sh")
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_ollama_running():
    """Check if Ollama service is running."""
    print_header("Test 2: Ollama Service")
    
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_success("Ollama service is running!")
            output = result.stdout.strip()
            
            if 'mistral' in output.lower():
                print_success("Mistral model is available")
                return True
            else:
                print_info("Mistral model not found")
                print_info("Pull it with: ollama pull mistral")
                return False
        else:
            print_error("Cannot connect to Ollama service")
            print_info("Start with: ollama serve")
            return False
    except Exception as e:
        print_error(f"Error connecting to Ollama: {e}")
        print_info("Make sure Ollama service is running: ollama serve")
        return False

def test_ollama_python_package():
    """Check if ollama Python package is installed."""
    print_header("Test 3: Ollama Python Package")
    
    try:
        import ollama
        print_success("Ollama Python package is installed")
        return True
    except ImportError:
        print_error("Ollama Python package not installed")
        print_info("Install with: pip install ollama")
        return False

def test_mistral_model():
    """Test if Mistral model can generate text."""
    print_header("Test 4: Mistral Model Test")
    
    try:
        import ollama
        
        print_info("Testing Mistral generation (may take 10-30 seconds)...")
        
        response = ollama.generate(
            model='mistral',
            prompt='Say "OK" if you are working.',
            options={'num_predict': 10}
        )
        
        text = response.get('response', response.get('text', ''))
        
        if text:
            print_success("Mistral model is working!")
            print(f"   Response: {text.strip()}")
            return True
        else:
            print_error("No response from model")
            return False
            
    except ImportError:
        print_error("Ollama package not installed")
        return False
    except Exception as e:
        error_msg = str(e)
        print_error(f"Error: {error_msg}")
        
        if 'connection' in error_msg.lower():
            print_info("Ollama service not running. Start with: ollama serve")
        elif 'not found' in error_msg.lower():
            print_info("Mistral model not found. Pull with: ollama pull mistral")
        
        return False

def main():
    print("\n" + "="*60)
    print("  Phase 5 Quick Test: LLM Setup Verification")
    print("="*60)
    
    results = {}
    
    # Test Ollama CLI
    results['ollama_installed'] = test_ollama_installed()
    
    # Test Ollama service
    if results['ollama_installed']:
        results['ollama_running'] = test_ollama_running()
    
    # Test Python package
    results['python_package'] = test_ollama_python_package()
    
    # Test Mistral model
    if results.get('ollama_running') and results.get('python_package'):
        results['mistral_working'] = test_mistral_model()
    
    # Summary
    print_header("Test Results Summary")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name.replace('_', ' ').title()}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  Total: {passed}/{total} tests passed")
    
    # Instructions
    print_header("Setup Instructions")
    
    if not results.get('ollama_installed'):
        print("1. Install Ollama CLI:")
        print("   curl https://ollama.ai/install.sh | sh")
    elif not results.get('ollama_running'):
        print("1. Start Ollama service (in a separate terminal):")
        print("   ollama serve")
        print("\n   OR check if it's already running as a background service")
    elif not results.get('python_package'):
        print("2. Install Ollama Python package:")
        print("   pip install ollama")
    elif not results.get('mistral_working'):
        print("3. Pull Mistral model:")
        print("   ollama pull mistral")
    
    if all(results.values()):
        print("\n🎉 All checks passed! Phase 5 is ready to test!")
        print("\nNext step: Test the full diagnosis endpoint:")
        print("  python test_image.py static/test-image2.jpeg")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()

