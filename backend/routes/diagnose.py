from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import time
import numpy as np
from config import ALLOWED_EXTENSIONS
from models.dual_classifier import get_dual_classifier
from services.leaf_detector import get_leaf_detector
from services.lesion_analyzer import get_lesion_analyzer
from services.diagnosis_engine import get_diagnosis_engine
from utils.validators import validate_image_file

diagnose_bp = Blueprint('diagnose', __name__)


def _make_json_serializable(obj):
    """Convert NumPy types and other non-JSON types to Python native types."""
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                       np.int16, np.int32, np.int64, np.uint8, np.uint16,
                       np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):  # Only np.bool_, not np.bool (deprecated)
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    return obj

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
        
        # Get dual classifier (uses both houseplant and PlantNet models)
        classifier = get_dual_classifier()
        
        # Run dual classifier inference (selects best result from both models)
        print("   Running dual classifier inference (houseplant + PlantNet)...")
        plant_results = classifier.predict(image_pil, top_k=5)
        
        # Log which model was selected
        dual_info = plant_results.get("dual_classifier", {})
        selected_model = dual_info.get("selected_model", "unknown")
        print(f"   Selected model: {selected_model} (confidence: {dual_info.get(selected_model + '_confidence', 0):.3f})")
        
        # Phase 4: Leaf detection and lesion analysis
        print("   Running leaf detection and lesion analysis...")
        leaf_detector = get_leaf_detector()
        lesion_analyzer = get_lesion_analyzer()
        
        # Detect and isolate leaves
        leaf_detection = leaf_detector.detect_leaves(
            image_pil, 
            confidence_threshold=0.15,
            return_boxes=True
        )
        
        # Analyze each detected leaf for lesions
        leaf_analyses = []
        overall_health_score = 0.0
        has_potential_issues = False
        
        # Get leaf bounding boxes for visualization
        leaf_boxes = leaf_detection.get('boxes', [])
        
        if leaf_detection['num_leaves'] > 0:
            for i, leaf_image in enumerate(leaf_detection['leaves']):
                analysis = lesion_analyzer.analyze_leaf(leaf_image)
                
                # Get lesion areas (bounding boxes for lesions)
                lesion_areas = analysis.get('lesion_areas', [])
                
                # Get leaf bounding box if available
                leaf_box = leaf_boxes[i] if i < len(leaf_boxes) else None
                
                leaf_analyses.append({
                    'leaf_index': i + 1,
                    'health_score': float(analysis['health_score']),
                    'green_percentage': float(analysis['green_percentage']),
                    'lesion_percentage': float(analysis['lesion_percentage']),
                    'num_lesion_regions': int(analysis['num_lesion_regions']),
                    'has_potential_issues': bool(analysis['has_potential_issues']),  # Ensure Python bool
                    'leaf_box': leaf_box,  # Bounding box for this leaf on original image
                    'lesion_areas': lesion_areas  # List of lesion bounding boxes
                })
                
                # Aggregate overall health (average of all leaves)
                overall_health_score += float(analysis['health_score'])
                if bool(analysis['has_potential_issues']):
                    has_potential_issues = True
            
            if len(leaf_analyses) > 0:
                overall_health_score = overall_health_score / len(leaf_analyses)
        
        # Phase 5: LLM Synthesis for diagnosis and treatment plan
        print("   Synthesizing diagnosis with LLM...")
        diagnosis_engine = get_diagnosis_engine()
        
        # Prepare data for diagnosis engine
        primary_plant = plant_results.get("primary", {})
        leaf_analysis_data = {
            'num_leaves_detected': int(leaf_detection['num_leaves']),
            'overall_health_score': float(overall_health_score),
            'has_potential_issues': bool(has_potential_issues),
            'individual_leaves': leaf_analyses
        }
        
        # Check if user wants LLM or rule-based (default: try LLM first)
        use_llm = request.form.get('use_llm', 'true').lower() == 'true'
        
        # Generate comprehensive diagnosis
        synthesis_result = diagnosis_engine.synthesize_diagnosis(
            plant_species=primary_plant,
            leaf_analysis=leaf_analysis_data,
            use_llm=use_llm
        )
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Build response with synthesized diagnosis
        # Add 'name' field for frontend compatibility (prioritizes common_name over species_name)
        primary_plant_formatted = plant_results["primary"].copy() if plant_results.get("primary") else {}
        if primary_plant_formatted:
            # Set 'name' to common_name if available, otherwise use species_name
            common_name = primary_plant_formatted.get("common_name", "")
            species_name = primary_plant_formatted.get("species_name", "Unknown")
            primary_plant_formatted["name"] = common_name if common_name and common_name != species_name else species_name
        
        # Format top_5 for frontend compatibility
        top_5_formatted = []
        for item in plant_results.get("top_k", []):
            item_copy = item.copy()
            # Set 'name' to common_name if available, otherwise use species_name
            item_common = item_copy.get("common_name", "")
            item_scientific = item_copy.get("species_name", "Unknown")
            item_copy["name"] = item_common if item_common and item_common != item_scientific else item_scientific
            top_5_formatted.append(item_copy)
        
        # Format disease_detection primary for frontend compatibility
        disease_primary = None
        if synthesis_result['final_diagnosis'].get('severity') != 'none':
            disease_primary = {
                "disease": synthesis_result['final_diagnosis'].get('condition', 'Unknown condition'),
                "common_name": primary_plant.get('common_name', ''),
                "confidence": synthesis_result['final_diagnosis'].get('confidence', 0.0),
                "affected_area_percent": (1.0 - leaf_analysis_data['overall_health_score']) * 100 if leaf_analysis_data.get('overall_health_score') else 0.0
            }
        
        response = {
            "success": True,
            "diagnosis": {
                "plant_species": {
                    "primary": primary_plant_formatted,
                    "top_5": top_5_formatted
                },
                "disease_detection": {
                    "leaf_analysis": {
                        "num_leaves_detected": int(leaf_detection['num_leaves']),
                        "overall_health_score": round(float(overall_health_score), 3),
                        "has_potential_issues": bool(has_potential_issues),  # Ensure Python bool
                        "individual_leaves": leaf_analyses,
                        "leaf_boxes": leaf_boxes  # All leaf bounding boxes for visualization
                    },
                    "primary": disease_primary,
                    "all_diseases": {}
                },
                "final_diagnosis": synthesis_result['final_diagnosis'],
                "treatment_plan": synthesis_result['treatment_plan'],
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "processing_time_ms": processing_time_ms,
                    "image_name": secure_filename(file.filename),
                    "model_versions": {
                        "plant_classification": "Dual Classifier (Houseplant + PlantNet)",
                        "selected_model": plant_results.get("dual_classifier", {}).get("selected_model", "plantnet"),
                        "plantnet": "EfficientNet B4",
                        "houseplant": "EfficientNet B4 (Fine-tuned)" if plant_results.get("dual_classifier", {}).get("houseplant_available", False) else "Not available",
                        "leaf_detection": "YOLO11x",
                        "lesion_analysis": "Image Processing",
                        "llm": "Mistral 7B" if synthesis_result.get('source') == 'llm' else "Rule-Based"
                    },
                    "dual_classifier": plant_results.get("dual_classifier", {}),
                    "diagnosis_source": synthesis_result.get('source', 'rule_based')
                }
            },
            "error": None
        }
        
        print("✅ Diagnosis complete")
        # Ensure all values are JSON serializable
        response = _make_json_serializable(response)
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

