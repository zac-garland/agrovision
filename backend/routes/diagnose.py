from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import time
from config import ALLOWED_EXTENSIONS
from models.plantnet_model import get_plantnet_model
from utils.validators import validate_image_file

diagnose_bp = Blueprint('diagnose', __name__)

@diagnose_bp.route('/diagnose', methods=['POST'])
def diagnose():
    """
    Main diagnosis endpoint.
    
    POST /diagnose
    
    Form Data:
        - image: Image file (jpg, png, etc.)
        - language: Optional language code (default: 'en')
    
    Returns:
        JSON with diagnosis results
    """
    start_time = time.time()
    
    try:
        # Check for file
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image provided",
                "diagnosis": None
            }), 400
        
        file = request.files['image']
        
        # Validate image
        is_valid, result = validate_image_file(file)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": result,  # error_message
                "diagnosis": None
            }), 400
        
        image_pil = result  # The validated PIL image
        
        print(f"\n🔍 Processing image: {file.filename}")
        print(f"   Size: {image_pil.size}, Mode: {image_pil.mode}")
        
        # Get PlantNet model
        plantnet = get_plantnet_model()
        
        # Run PlantNet inference
        print("   Running PlantNet inference...")
        plant_results = plantnet.predict(image_pil, top_k=5)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Build response
        response = {
            "success": True,
            "diagnosis": {
                "plant_species": {
                    "primary": plant_results["primary"],
                    "top_5": plant_results["top_k"]
                },
                "disease_detection": {
                    "primary": None,  # Will add in Phase 4
                    "all_diseases": {}
                },
                "final_diagnosis": {
                    "condition": "Pending disease detection",
                    "confidence": None,
                    "severity": None,
                    "reasoning": "Plant identified. Disease detection coming soon."
                },
                "treatment_plan": {
                    "immediate": [],
                    "week_1": [],
                    "week_2_3": [],
                    "monitoring": ""
                },
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "processing_time_ms": processing_time_ms,
                    "image_name": secure_filename(file.filename),
                    "model_versions": {
                        "plantnet": "ResNet152",
                        "disease_detection": "pending",
                        "llm": "Mistral 7B"
                    }
                }
            },
            "error": None
        }
        
        print("✅ Diagnosis complete")
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": str(e),
            "diagnosis": None
        }), 500

