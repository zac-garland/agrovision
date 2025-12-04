#!/usr/bin/env python3
"""
Quick script to test an image via the /diagnose endpoint
Usage: python test_image.py [image_path] [endpoint_url]
"""

import sys
import requests
import json
from pathlib import Path

def test_image(image_path, endpoint="http://127.0.0.1:5000/diagnose"):
    """Test an image file via the diagnosis endpoint."""
    
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    print(f"🔍 Testing image: {image_path}")
    print(f"📡 Endpoint: {endpoint}")
    print("")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            print("⏳ Sending request...")
            response = requests.post(
                endpoint,
                files=files,
                timeout=60
            )
        
        print(f"\n📊 Status Code: {response.status_code}\n")
        
        if response.status_code == 200:
            data = response.json()
            
            # Pretty print the response
            print("✅ Response:")
            print(json.dumps(data, indent=2))
            
            # Extract key info
            if data.get('success') and data.get('diagnosis'):
                diagnosis = data['diagnosis']
                
                # Plant species
                if 'plant_species' in diagnosis and diagnosis['plant_species'].get('primary'):
                    primary = diagnosis['plant_species']['primary']
                    print(f"\n🌿 Plant: {primary.get('common_name', 'Unknown')}")
                    print(f"   Scientific: {primary.get('species_name', 'Unknown')}")
                    conf = primary.get('confidence')
                    if conf is not None:
                        print(f"   Confidence: {conf*100:.1f}%")
                    else:
                        print(f"   Confidence: N/A")
                
                # Disease detection
                if 'disease_detection' in diagnosis:
                    leaf_analysis = diagnosis['disease_detection'].get('leaf_analysis', {})
                    if leaf_analysis:
                        print(f"\n🔬 Leaf Analysis:")
                        print(f"   Leaves detected: {leaf_analysis.get('num_leaves_detected', 0)}")
                        health_score = leaf_analysis.get('overall_health_score')
                        if health_score is not None:
                            print(f"   Health score: {health_score:.3f}")
                        else:
                            print(f"   Health score: N/A")
                        print(f"   Potential issues: {'Yes' if leaf_analysis.get('has_potential_issues') else 'No'}")
                
                # Final diagnosis
                if 'final_diagnosis' in diagnosis:
                    final = diagnosis['final_diagnosis']
                    print(f"\n📋 Final Diagnosis:")
                    print(f"   Condition: {final.get('condition', 'Unknown')}")
                    print(f"   Severity: {final.get('severity', 'Unknown')}")
                    
                    confidence = final.get('confidence')
                    if confidence is not None and isinstance(confidence, (int, float)):
                        print(f"   Confidence: {confidence:.3f}")
                    else:
                        print(f"   Confidence: N/A")
                    
                    reasoning = final.get('reasoning', 'N/A')
                    if reasoning and len(reasoning) > 200:
                        print(f"   Reasoning: {reasoning[:200]}...")
                    elif reasoning:
                        print(f"   Reasoning: {reasoning}")
                    else:
                        print(f"   Reasoning: N/A")
                
                # Treatment plan
                if 'treatment_plan' in diagnosis:
                    treatment = diagnosis['treatment_plan']
                    print(f"\n💊 Treatment Plan:")
                    
                    if treatment.get('immediate'):
                        print(f"   Immediate ({len(treatment['immediate'])} actions):")
                        for i, action in enumerate(treatment['immediate'][:3], 1):
                            print(f"     {i}. {action}")
                    
                    if treatment.get('week_1'):
                        print(f"   Week 1 ({len(treatment['week_1'])} steps):")
                        for i, step in enumerate(treatment['week_1'][:2], 1):
                            print(f"     {i}. {step}")
                    
                    if treatment.get('monitoring'):
                        monitoring = treatment['monitoring']
                        if len(monitoring) > 100:
                            monitoring = monitoring[:100] + "..."
                        print(f"   Monitoring: {monitoring}")
                
                # Metadata (including LLM info)
                if 'metadata' in diagnosis:
                    meta = diagnosis['metadata']
                    print(f"\n⏱️  Processing time: {meta.get('processing_time_ms', 0)}ms")
                    
                    # Show diagnosis source (LLM or rule-based)
                    diagnosis_source = meta.get('diagnosis_source', 'unknown')
                    if diagnosis_source == 'llm':
                        print(f"🤖 Diagnosis Source: LLM (Mistral 7B)")
                        print(f"   ✅ Using AI-powered reasoning")
                    elif diagnosis_source == 'rule_based':
                        print(f"📋 Diagnosis Source: Rule-Based")
                        print(f"   ℹ️  Using deterministic recommendations")
                    else:
                        print(f"📋 Diagnosis Source: {diagnosis_source}")
                    
                    # Show model versions
                    if 'model_versions' in meta:
                        models = meta['model_versions']
                        print(f"\n🔧 Models Used:")
                        for model_name, version in models.items():
                            print(f"   - {model_name}: {version}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except:
                print(response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Is the Flask server running?")
        print("   Start it with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Get image path from command line or use default
    image_path = sys.argv[1] if len(sys.argv) > 1 else "static/test-image2.jpeg"
    endpoint = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:5000/diagnose"
    
    test_image(image_path, endpoint)

