from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import time
import numpy as np
from config import ALLOWED_EXTENSIONS
from models.dual_classifier import get_dual_classifier
from models.multi_classifier import get_multi_classifier
from models.species_classifier import get_species_classifier
from models.disease_classifier import get_disease_classifier
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
        
        # Phase 1: Species identification using new unified species classifier
        print("   Running species classifier...")
        species_classifier = get_species_classifier()
        species_results = species_classifier.predict(image_pil, top_k=5)
        
        # Phase 2: Disease detection using disease classifier
        print("   Running disease classifier...")
        disease_classifier = get_disease_classifier()
        print(f"   Disease classifier available: {disease_classifier.available}")
        if not disease_classifier.available:
            print(f"   ⚠️  Disease classifier not available - check model file exists")
        disease_results = disease_classifier.predict(image_pil, top_k=5)
        print(f"   Disease results available: {disease_results.get('available', False)}")
        if disease_results.get('error'):
            print(f"   Disease classifier error: {disease_results.get('error')}")
        
        # Format plant_results for backward compatibility
        # Use species classifier as primary, with disease info
        plant_results = {
            'species': species_results,
            'disease': disease_results,
            'primary': species_results.get('primary'),
            'top_k': species_results.get('top_k', []),
            'model_source': 'unified_classifiers'
        }
        
        # Log results
        if species_results.get('available'):
            primary_species = species_results.get('primary', {})
            print(f"   Species: {primary_species.get('species', 'unknown')} ({primary_species.get('confidence', 0):.3f})")
        
        if disease_results.get('available'):
            primary_disease = disease_results.get('primary', {})
            print(f"   Disease: {primary_disease.get('condition', 'unknown')} ({primary_disease.get('confidence', 0):.3f})")
        
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
        # Filter out non-leaf detections (flowers, pots, soil) by checking green percentage
        leaf_analyses = []
        overall_health_score = 0.0
        has_potential_issues = False
        rejected_detections = 0  # Track how many were filtered out
        
        # Get leaf bounding boxes for visualization
        leaf_boxes = leaf_detection.get('boxes', [])
        
        # Green percentage threshold for accepting a detection as a leaf
        GREEN_THRESHOLD = 50.0  # Minimum 50% green to be considered a leaf
        
        if leaf_detection['num_leaves'] > 0:
            for i, leaf_image in enumerate(leaf_detection['leaves']):
                # Quick green percentage check to filter out non-leaf detections
                green_pct = lesion_analyzer.check_green_percentage(leaf_image)
                
                if green_pct < GREEN_THRESHOLD:
                    # Reject this detection - likely a flower, pot, soil, or other non-leaf object
                    rejected_detections += 1
                    print(f"   ⚠️  Rejected detection {i+1}: only {green_pct:.1f}% green (threshold: {GREEN_THRESHOLD}%)")
                    continue
                
                # This detection passed the green threshold - analyze it for lesions
                analysis = lesion_analyzer.analyze_leaf(leaf_image)
                
                # Get lesion areas (bounding boxes for lesions)
                lesion_areas = analysis.get('lesion_areas', [])
                
                # Get leaf bounding box if available
                leaf_box = leaf_boxes[i] if i < len(leaf_boxes) else None
                
                leaf_analyses.append({
                    'leaf_index': len(leaf_analyses) + 1,  # Re-index after filtering
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
            else:
                # All detections were rejected - set default values
                overall_health_score = 0.5  # Neutral score
                print(f"   ⚠️  All {leaf_detection['num_leaves']} detection(s) were rejected (insufficient green)")
                print(f"   ℹ️  This may indicate the image contains flowers, pots, or soil rather than leaves")
            
            # Log filtering results
            if rejected_detections > 0:
                print(f"   ℹ️  Filtered out {rejected_detections} non-leaf detection(s) (< {GREEN_THRESHOLD}% green)")
            if len(leaf_analyses) > 0:
                print(f"   ✅ Analyzing {len(leaf_analyses)} valid leaf detection(s)")
        
        # Phase 5: LLM Synthesis for diagnosis and treatment plan
        print("   Synthesizing diagnosis with LLM...")
        diagnosis_engine = get_diagnosis_engine()
        
        # Prepare data for diagnosis engine
        # Format species result for compatibility
        species_primary = species_results.get("primary", {})
        primary_plant = {
            'species_name': species_primary.get('species', 'Unknown'),
            'common_name': species_primary.get('species', 'Unknown'),  # Species classifier uses hierarchical names
            'confidence': species_primary.get('confidence', 0.0)
        }
        
        # Format disease result
        disease_primary = disease_results.get("primary", {})
        
        # Get disease classifier confidence for diagnosis FIRST (before any formatting)
        # Extract confidence from the raw disease_primary result
        disease_confidence = 0.0
        if disease_results.get('available') and disease_primary:
            raw_confidence = disease_primary.get('confidence', 0.0)
            disease_confidence = float(raw_confidence) if raw_confidence is not None else 0.0
            if disease_confidence > 0:
                print(f"   ✅ Disease classifier confidence extracted: {disease_confidence:.3f} ({disease_confidence*100:.1f}%)")
            else:
                print(f"   ⚠️  Disease classifier available but confidence is 0")
        else:
            # Disease classifier not available - this is OK, we'll use fallback
            if not disease_results.get('available'):
                print(f"   ⚠️  Disease classifier not available - will use health-based confidence")
                if disease_results.get('error'):
                    print(f"      Error: {disease_results.get('error')}")
            elif not disease_primary:
                print(f"   ⚠️  Disease classifier available but no primary result")
        
        # Build detailed model information for LLM and frontend
        model_details = {
            'species_classifier': {
                'model_name': 'Unified Species Classifier',
                'description': 'Combines houseplant species, PlantNet species, and PlantVillage healthy classes',
                'num_classes': species_results.get('num_classes', 0),
                'available': species_results.get('available', False),
                'top_predictions': species_results.get('top_k', [])[:5],
                'primary': species_primary
            },
            'disease_classifier': {
                'model_name': 'Disease Classifier',
                'description': 'Trained on PlantVillage dataset to identify specific plant diseases',
                'num_classes': disease_results.get('num_classes', 0),
                'available': disease_results.get('available', False),
                'top_predictions': disease_results.get('top_k', [])[:5],
                'primary': disease_primary
            },
            'lesion_analysis': {
                'method': 'Image Processing (HSV color analysis)',
                'description': 'Detects lesions through color analysis (yellowing, browning)',
                'num_leaves': len(leaf_analyses),  # Use filtered count, not original detection count
                'num_detections': int(leaf_detection['num_leaves']),  # Original YOLO detection count
                'rejected_detections': rejected_detections,  # How many were filtered out
                'overall_health_score': float(overall_health_score),
                'has_issues': bool(has_potential_issues),
                'individual_leaves': leaf_analyses
            }
        }
        
        leaf_analysis_data = {
            'num_leaves_detected': len(leaf_analyses),  # Use filtered count
            'num_detections_total': int(leaf_detection['num_leaves']),  # Original YOLO detections
            'rejected_detections': rejected_detections,  # Non-leaf objects filtered out
            'overall_health_score': float(overall_health_score),
            'has_potential_issues': bool(has_potential_issues),
            'individual_leaves': leaf_analyses,
            'disease_classifier': disease_results if disease_results.get('available') else None,
            'disease_confidence': disease_confidence,  # Pass disease classifier confidence
            'model_details': model_details  # Include detailed model info for LLM
        }
        
        # Check if user wants LLM or rule-based (default: try LLM first)
        use_llm = request.form.get('use_llm', 'true').lower() == 'true'
        
        # Generate comprehensive diagnosis
        synthesis_result = diagnosis_engine.synthesize_diagnosis(
            plant_species=primary_plant,
            leaf_analysis=leaf_analysis_data,
            use_llm=use_llm,
            multi_model_results=None  # Using model_details instead
        )
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Build response with synthesized diagnosis
        # Format species results for frontend
        primary_plant_formatted = {
            'species_name': species_primary.get('species', 'Unknown'),
            'common_name': species_primary.get('species', 'Unknown'),
            'name': species_primary.get('species', 'Unknown'),
            'confidence': species_primary.get('confidence', 0.0)
        } if species_primary else {}
        
        # Format top_5 for frontend compatibility
        top_5_formatted = []
        for item in species_results.get("top_k", []):
            species_name = item.get('species', 'Unknown')
            top_5_formatted.append({
                'species_name': species_name,
                'common_name': species_name,
                'name': species_name,
                'confidence': item.get('confidence', 0.0)
            })
        
        # Format disease results for frontend (NO PLANT NAMES - just disease type)
        disease_primary_formatted = None
        if disease_primary and disease_results.get('available'):
            disease_primary_formatted = {
                "disease": disease_primary.get('disease', 'Unknown'),
                "condition": disease_primary.get('condition', 'Unknown'),
                "confidence": disease_primary.get('confidence', 0.0),
                "affected_area_percent": (1.0 - leaf_analysis_data['overall_health_score']) * 100 if leaf_analysis_data.get('overall_health_score') else 0.0
            }
        
        response = {
            "success": True,
            "diagnosis": {
                "plant_species": {
                    "primary": primary_plant_formatted,
                    "top_5": top_5_formatted,
                    "model_source": "unified_species_classifier"
                },
                # Keep disease_detection for leaf_analysis only (not displayed as separate section)
                "disease_detection": {
                    # Leaf analysis (image processing-based health assessment)
                    "leaf_analysis": {
                        "num_leaves_detected": len(leaf_analyses),
                        "num_detections_total": int(leaf_detection['num_leaves']),
                        "rejected_detections": rejected_detections,
                        "overall_health_score": round(float(overall_health_score), 3),
                        "has_potential_issues": bool(has_potential_issues),
                        "individual_leaves": leaf_analyses,
                        "leaf_boxes": leaf_boxes  # All leaf bounding boxes for visualization
                    }
                },
                "final_diagnosis": synthesis_result['final_diagnosis'],
                "treatment_plan": synthesis_result['treatment_plan'],
                "model_results": {
                    # Tabular data for frontend display
                    "species_predictions": [
                        {
                            "rank": i + 1,
                            "species": pred.get('species', 'Unknown'),
                            "confidence": f"{pred.get('confidence', 0)*100:.1f}%"
                        }
                        for i, pred in enumerate(species_results.get('top_k', [])[:10])
                    ],
                    "disease_predictions": [
                        {
                            "rank": i + 1,
                            "disease": pred.get('disease', 'Unknown'),
                            "condition": pred.get('condition', 'Unknown'),
                            "confidence": f"{pred.get('confidence', 0)*100:.1f}%"
                        }
                        for i, pred in enumerate(disease_results.get('top_k', [])[:10])
                    ],
                    "leaf_analysis_table": [
                        {
                            "leaf": f"Leaf {i+1}",
                            "health_score": f"{leaf.get('health_score', 0):.2f}",
                            "lesion_pct": f"{leaf.get('lesion_percentage', 0):.1f}%",
                            "green_pct": f"{leaf.get('green_percentage', 0):.1f}%",
                            "lesion_regions": leaf.get('num_lesion_regions', 0),
                            "has_issues": "Yes" if leaf.get('has_potential_issues') else "No"
                        }
                        for i, leaf in enumerate(leaf_analyses)
                    ],
                    "model_details": model_details
                },
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "processing_time_ms": processing_time_ms,
                    "image_name": secure_filename(file.filename),
                    "model_versions": {
                        "plant_classification": "Unified Species Classifier",
                        "disease_classification": "Disease Classifier",
                        "species_model": "unified_species_classifier",
                        "disease_model": "disease_classifier",
                        "species_available": species_results.get('available', False),
                        "disease_available": disease_results.get('available', False),
                        "leaf_detection": "YOLO11x",
                        "lesion_analysis": "Image Processing",
                        "llm": "Mistral 7B" if synthesis_result.get('source') == 'llm' else "Rule-Based"
                    },
                    "species_classifier": {
                        "available": species_results.get('available', False),
                        "num_classes": species_results.get('num_classes', 0)
                    },
                    "disease_classifier": {
                        "available": disease_results.get('available', False),
                        "num_classes": disease_results.get('num_classes', 0)
                    },
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

