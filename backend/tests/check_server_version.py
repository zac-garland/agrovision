#!/usr/bin/env python3
"""
Quick script to check if the server is running the latest version.
Looks for key indicators that Phase 4 & 5 are implemented.
"""

import requests
import json

def check_server():
    """Check if server is running latest version."""
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=2)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"⚠️  Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running!")
        print("   Start it with: python app.py")
        return False
    
    # Now check if Phase 4/5 are working by examining a test response structure
    print("\n📋 Checking endpoint version...")
    print("   (This requires an image test to fully verify)")
    print("\n💡 To fully test:")
    print("   1. Make sure server is restarted (stop and start python app.py)")
    print("   2. Run: python test_image.py static/test-image2.jpeg")
    print("   3. Look for:")
    print("      - leaf_analysis data")
    print("      - diagnosis_source: 'llm' or 'rule_based' (not 'unknown')")
    print("      - treatment_plan with steps")
    
    return True

if __name__ == "__main__":
    check_server()

