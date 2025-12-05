import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import pandas as pd
from pathlib import Path
import base64

BACKEND_URL = "http://127.0.0.1:5000/diagnose"

# Page configuration - moved to top
st.set_page_config(
    page_title="AgroVision+ Plant Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Banner container with background image */
    .banner-container {
        position: relative;
        width: 100%;
        padding: 60px 20px;
        margin-bottom: 30px;
        background: linear-gradient(rgba(45, 106, 79, 0.7), rgba(64, 145, 108, 0.7)), 
                    url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwMCIgaGVpZ2h0PSIzMDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjEwMDAiIGhlaWdodD0iMzAwIiBmaWxsPSIjZmZmYWM2Ii8+PHRleHQgeD0iNTAwIiB5PSIxNTAiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIyNCIgZmlsbD0iIzMzMzMzMyIgdGV4dC1hbmNob3I9Im1pZGRsZSI+QWdyb1Zpc2lvbiBCYW5uZXI8L3RleHQ+PC9zdmc+');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .banner-content {
        position: relative;
        z-index: 2;
    }
    
    /* Main title styling */
    .banner-title {
        color: #ffffff;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        margin-bottom: 10px;
    }
    
    .banner-subtitle {
        color: #e8f5e9;
        text-align: center;
        font-size: 1.3em;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }
    
    h1 {
        color: #2d6a4f;
        text-align: center;
        padding-bottom: 10px;
        border-bottom: 3px solid #40916c;
    }
    
    /* Subheader styling */
    h2 {
        color: #1b4332;
        margin-top: 30px;
        padding-bottom: 10px;
        border-bottom: 2px solid #95d5b2;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #d1e7dd;
        border-left: 4px solid #198754;
    }
    
    /* Warning styling */
    .stWarning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    
    /* Error styling */
    .stError {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    
    /* Info styling */
    .stInfo {
        background-color: #cfe2ff;
        border-left: 4px solid #0d6efd;
    }
</style>
""", unsafe_allow_html=True)


def call_backend(uploaded_file, language="en"):
    """send the image to the backend /diagnose endpoint"""
    filename = uploaded_file.name if hasattr(uploaded_file, 'name') and uploaded_file.name else 'image.jpg'
    
    uploaded_file.seek(0)
    image_bytes = uploaded_file.read()
    
    files = {"image": (filename, BytesIO(image_bytes))}
    data = {"language": language}

    try:
        response = requests.post(BACKEND_URL, files=files, data=data, timeout=60)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"could not reach backend. error was: {e}")

    if response.status_code != 200:
        raise RuntimeError(f"backend returned status {response.status_code}: {response.text}")

    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError("backend response was not valid json")

    if not payload.get("success", False):
        error_msg = payload.get("error", "unknown error from backend")
        raise RuntimeError(error_msg)

    return payload.get("diagnosis", {})


def show_plant_species(plant_species):
    """show plant species with better visual hierarchy"""
    if not plant_species:
        return

    primary = plant_species.get("primary", {})
    top5 = plant_species.get("top_5", [])

    common_name = primary.get("common_name", "")
    scientific_name = primary.get("species_name") or primary.get("name", "unknown")
    main_name = common_name if common_name and common_name != scientific_name else scientific_name
    conf = primary.get("confidence", 0.0)

    st.subheader("🌿 Plant Species")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### {main_name}")
        if common_name and common_name != scientific_name and main_name == common_name:
            st.markdown(f"*{scientific_name}*")
    
    with col2:
        # Confidence badge
        if conf >= 0.8:
            st.success(f"✅ {conf:.0%}")
        elif conf >= 0.6:
            st.info(f"⚠️ {conf:.0%}")
        else:
            st.warning(f"❓ {conf:.0%}")

    if top5:
        labels = []
        for item in top5:
            item_common = item.get("common_name", "")
            item_scientific = item.get("species_name") or item.get("name", "")
            label = item_common if item_common and item_common != item_scientific else item_scientific
            labels.append(label)
        
        values = [item.get("confidence", 0.0) for item in top5]
        
        st.markdown("**Top 5 Species**")
        chart_data = pd.DataFrame({"Species": labels, "Confidence": values})
        st.bar_chart(chart_data.set_index("Species"), height=400,horizontal=True)


def show_multi_model_comparison(multi_model_data):
    """Show side-by-side comparison of all model predictions."""
    if not multi_model_data:
        return
    
    st.subheader("🤖 Multi-Model Comparison")
    st.caption("Results from different AI models analyzing the same image")
    
    # Create columns for each model
    cols = st.columns(3)
    
    # PlantNet Model
    with cols[0]:
        st.markdown("#### 🌍 PlantNet")
        plantnet = multi_model_data.get("plantnet", {})
        if plantnet.get("available"):
            primary = plantnet.get("primary", {})
            st.markdown(f"**{primary.get('species_name', 'Unknown')}**")
            st.caption(f"Common: {primary.get('common_name', 'N/A')}")
            conf = primary.get("confidence", 0.0)
            if conf >= 0.8:
                st.success(f"Confidence: {conf:.0%}")
            elif conf >= 0.6:
                st.info(f"Confidence: {conf:.0%}")
            else:
                st.warning(f"Confidence: {conf:.0%}")
        else:
            st.error("Not available")
            if plantnet.get("error"):
                st.caption(f"Error: {plantnet.get('error')[:30]}...")
    
    # Houseplant Model
    with cols[1]:
        st.markdown("#### 🏠 Houseplant")
        houseplant = multi_model_data.get("houseplant", {})
        if houseplant.get("available"):
            primary = houseplant.get("primary", {})
            st.markdown(f"**{primary.get('species_name', 'Unknown')}**")
            st.caption(f"Common: {primary.get('common_name', 'N/A')}")
            conf = primary.get("confidence", 0.0)
            if conf >= 0.8:
                st.success(f"Confidence: {conf:.0%}")
            elif conf >= 0.6:
                st.info(f"Confidence: {conf:.0%}")
            else:
                st.warning(f"Confidence: {conf:.0%}")
        else:
            st.info("Not available")
            if houseplant.get("error"):
                st.caption(f"Error: {houseplant.get('error')[:30]}...")
    
    # LLaVA Model
    with cols[2]:
        st.markdown("#### 👁️ LLaVA Vision")
        llava = multi_model_data.get("llava", {})
        if llava.get("available"):
            plant_id = llava.get("plant_identification", {})
            lesion = llava.get("lesion_analysis", {})
            
            st.markdown(f"**{plant_id.get('species_name', 'Unknown')}**")
            st.caption(f"Common: {plant_id.get('common_name', 'N/A')}")
            
            conf_level = plant_id.get("confidence", "medium")
            if conf_level == "high":
                st.success(f"Confidence: {conf_level}")
            elif conf_level == "medium":
                st.info(f"Confidence: {conf_level}")
            else:
                st.warning(f"Confidence: {conf_level}")
            
            # Show lesion detection
            if lesion.get("has_lesions"):
                st.error(f"⚠️ Lesions: {lesion.get('affected_percentage', 0):.1f}%")
                st.caption(f"Type: {lesion.get('lesion_type', 'Unknown')}")
            else:
                st.success("✅ No lesions")
        else:
            st.info("Not available")
            if llava.get("error"):
                st.caption(f"Error: {llava.get('error')[:30]}...")
    
    # Show LLaVA reasoning if available
    if llava.get("available") and llava.get("plant_identification", {}).get("reasoning"):
        with st.expander("📝 LLaVA Detailed Analysis"):
            reasoning = llava.get("plant_identification", {}).get("reasoning", "")
            if reasoning:
                st.write(reasoning)
            
            lesion_reasoning = llava.get("lesion_analysis", {}).get("reasoning", "")
            if lesion_reasoning:
                st.markdown("**Lesion Analysis:**")
                st.write(lesion_reasoning)


def show_disease_section(disease_detection):
    """show disease detection with visual indicators"""
    if not disease_detection:
        return

    st.subheader("🦠 Disease Detection")
    
    primary = disease_detection.get("primary")
    if primary:
        disease_name = primary.get("disease", "unknown")
        common_name = primary.get("common_name", "")
        conf = primary.get("confidence", 0.0)
        affected = primary.get("affected_area_percent", 0)

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Disease", disease_name)
        
        with col2:
            st.metric("Confidence", f"{conf:.0%}")
        
        with col3:
            st.metric("Affected", f"{affected:.1f}%")
        
        if common_name:
            st.caption(f"*{common_name}*")
    else:
        st.success("✅ No disease detected - plant appears healthy!")


def show_final_diagnosis(final_diagnosis):
    """show final diagnosis with visual indicators"""
    if not final_diagnosis:
        return

    st.markdown("### 📋 Final Diagnosis")

    # Get values directly from final_diagnosis (these come from the LLM/rule-based diagnosis)
    condition = final_diagnosis.get("condition", "not available")
    conf = final_diagnosis.get("confidence", 0.0)
    severity = final_diagnosis.get("severity", "not specified")
    reasoning = final_diagnosis.get("reasoning", "")

    severity_colors = {
        "none": "🟢",
        "low": "🟡",
        "moderate": "🟠",
        "medium": "🟠",
        "high": "🔴",
        "critical": "⛔"
    }
    severity_emoji = severity_colors.get(severity.lower(), "❓")

    # Display metrics - these now match the actual diagnosis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Condition", condition)
    
    with col2:
        # Confidence from diagnosis (higher = more confident in the diagnosis)
        st.metric("Confidence", f"{conf:.1%}")
    
    with col3:
        st.markdown(f"**Severity:** {severity_emoji} {severity.title()}")

    # Full reasoning explanation
    if reasoning:
        st.markdown("---")
        st.markdown("#### 📖 Detailed Explanation")
        # Use full reasoning if available, otherwise use reasoning
        reasoning_text = final_diagnosis.get('reasoning_full', reasoning)
        st.markdown(reasoning_text)
        st.caption("💡 This explanation combines AI model predictions with expert analysis")


def draw_bounding_boxes(image, leaf_boxes, individual_leaves=None):
    """Draw bounding boxes on image"""
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    img_width, img_height = annotated_image.size
    
    if leaf_boxes:
        for i, box in enumerate(leaf_boxes):
            if box:
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))
                    
                    if x2 > x1 and y2 > y1:
                        draw.rectangle([x1, y1, x2, y2], outline="lime", width=5)
                        label_y = max(10, y1 - 25)
                        draw.text((x1 + 5, label_y), f"Leaf {i+1}", fill="lime")
    
    if individual_leaves:
        for leaf_idx, leaf_data in enumerate(individual_leaves):
            leaf_box = leaf_data.get('leaf_box')
            lesion_areas = leaf_data.get('lesion_areas', [])
            
            # Initialize leaf coordinates
            leaf_x1 = leaf_y1 = leaf_x2 = leaf_y2 = None
            
            if leaf_box and isinstance(leaf_box, (list, tuple)) and len(leaf_box) == 4:
                leaf_x1, leaf_y1, leaf_x2, leaf_y2 = [int(x) for x in leaf_box]
                
                leaf_x1 = max(0, min(leaf_x1, img_width))
                leaf_y1 = max(0, min(leaf_y1, img_height))
                leaf_x2 = max(0, min(leaf_x2, img_width))
                leaf_y2 = max(0, min(leaf_y2, img_height))
                
                if leaf_x2 > leaf_x1 and leaf_y2 > leaf_y1:
                    draw.rectangle([leaf_x1, leaf_y1, leaf_x2, leaf_y2], outline="lime", width=5)
                    label_y = max(10, leaf_y1 - 25)
                    leaf_num = leaf_data.get('leaf_index', leaf_idx + 1)
                    draw.text((leaf_x1 + 5, label_y), f"Leaf {leaf_num}", fill="lime")
            
            # Draw lesion boxes
            # Lesions are dictionaries with 'bbox' key, coordinates are relative to the cropped leaf image
            if lesion_areas and leaf_x1 is not None and leaf_y1 is not None:
                for lesion in lesion_areas:
                    if not lesion:  # Skip empty/None lesions
                        continue
                    
                    # Extract bbox from lesion dictionary
                    # Lesion format: {'bbox': [x, y, x+w, y+h], 'area': int, 'centroid': [cx, cy]}
                    # After JSON serialization, tuples become lists
                    lesion_bbox = None
                    
                    if isinstance(lesion, dict):
                        lesion_bbox = lesion.get('bbox') or lesion.get('BBox')
                    elif isinstance(lesion, (list, tuple)):
                        # Sometimes lesion might be a list/tuple directly
                        if len(lesion) == 4:
                            lesion_bbox = lesion
                    
                    if lesion_bbox:
                        try:
                            # Handle both list and tuple formats (JSON converts tuples to lists)
                            if isinstance(lesion_bbox, (list, tuple)) and len(lesion_bbox) == 4:
                                # Lesion bbox coordinates are relative to the cropped leaf image
                                lx1, ly1, lx2, ly2 = [int(float(x)) for x in lesion_bbox]
                                
                                # Transform to absolute coordinates on original image
                                # Add the leaf's position offset
                                abs_x1 = leaf_x1 + lx1
                                abs_y1 = leaf_y1 + ly1
                                abs_x2 = leaf_x1 + lx2
                                abs_y2 = leaf_y1 + ly2
                                
                                # Ensure coordinates are within image bounds
                                abs_x1 = max(0, min(abs_x1, img_width))
                                abs_y1 = max(0, min(abs_y1, img_height))
                                abs_x2 = max(0, min(abs_x2, img_width))
                                abs_y2 = max(0, min(abs_y2, img_height))
                                
                                # Draw lesion rectangle if valid
                                if abs_x2 > abs_x1 and abs_y2 > abs_y1:
                                    draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline="red", width=3)
                        except (ValueError, TypeError, IndexError) as e:
                            # Skip invalid lesion coordinates
                            continue
    
    return annotated_image


def show_banner():
    """Display AgroVision banner with background image."""
    # Try to load banner image
    banner_paths = [
        Path(__file__).parent / "static" / "images" / "banner.png",
        Path(__file__).parent / "static" / "images" / "banner.jpg",
        Path(__file__).parent / "static" / "images" / "agrovision_banner.png",
        Path(__file__).parent / "static" / "images" / "agrovision_banner.jpg",
    ]
    
    banner_image = None
    for path in banner_paths:
        if path.exists():
            try:
                banner_image = Image.open(path)
                break
            except Exception as e:
                continue
    
    if banner_image:
        # Resize banner to a reasonable width while maintaining aspect ratio
        max_width = 1200
        if banner_image.width > max_width:
            ratio = max_width / banner_image.width
            new_height = int(banner_image.height * ratio)
            # Use LANCZOS if available, fallback to BICUBIC
            try:
                banner_image = banner_image.resize((max_width, new_height), Image.Resampling.LANCZOS)
            except AttributeError:
                # Older PIL versions
                banner_image = banner_image.resize((max_width, new_height), Image.LANCZOS)
        
        # Convert to RGBA for overlay blending
        banner_rgba = banner_image.convert('RGBA')
        
        # Create overlay with semi-transparent background for better text readability
        overlay = Image.new('RGBA', banner_rgba.size, (45, 106, 79, 180))  # Semi-transparent green
        
        # Blend overlay with banner
        banner_with_overlay = Image.alpha_composite(banner_rgba, overlay).convert('RGB')
        
        # Create a container with the banner image and overlay text
        st.markdown(f"""
        <div style="position: relative; width: 100%; margin-bottom: 30px;">
            <img src="data:image/png;base64,{base64.b64encode(BytesIO().getvalue()).decode()}" 
                 style="width: 100%; height: auto; border-radius: 10px; display: block;" 
                 onerror="this.style.display='none'">
        </div>
        """, unsafe_allow_html=True)
        
        # Display banner image
        st.image(banner_with_overlay, use_container_width=True)
        
        # Add text overlay using a container positioned over the image
        st.markdown("""
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                    text-align: center; width: 100%; z-index: 100;">
            <h1 style="color: white; font-size: 3.5em; font-weight: bold; 
                       text-shadow: 3px 3px 6px rgba(0,0,0,0.8); margin-bottom: 10px; 
                       background: rgba(45, 106, 79, 0.3); padding: 20px; border-radius: 10px;">
                🌱 AgroVision+
            </h1>
            <p style="color: #e8f5e9; font-size: 1.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
                      background: rgba(45, 106, 79, 0.3); padding: 10px; border-radius: 5px;">
                Agentic plant diagnosis for home & hobby growers
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback: Simple text banner
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2d6a4f 0%, #40916c 100%); padding: 60px 20px; border-radius: 10px; text-align: center; margin-bottom: 30px;">
            <h1 style="color: white; font-size: 3.5em; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); margin-bottom: 10px;">
                🌱 AgroVision+
            </h1>
            <p style="color: #e8f5e9; font-size: 1.5em; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                Agentic plant diagnosis for home & hobby growers
            </p>
        </div>
        """, unsafe_allow_html=True)


def show_treatment_plan(plan):
    """show treatment plan with timeline"""
    if not plan:
        return

    st.subheader("💊 Treatment Plan")

    immediate = plan.get("immediate", [])
    week1 = plan.get("week_1", [])
    week23 = plan.get("week_2_3", [])
    monitoring = plan.get("monitoring", "")

    if immediate:
        st.markdown("#### 🚨 Right Now (Within 24 Hours)")
        for step in immediate:
            st.markdown(f"• {step}")

    if week1:
        st.markdown("#### 📅 This Week")
        for step in week1:
            st.markdown(f"• {step}")

    if week23:
        st.markdown("#### 📋 Weeks 2-3")
        for step in week23:
            st.markdown(f"• {step}")

    if monitoring:
        st.markdown("#### 📝 Ongoing Monitoring")
        st.info(monitoring)


def main():
    # Display banner with background image
    show_banner()
    
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        language = st.selectbox("Language", ["en"])
        st.markdown("---")
        st.markdown("### ℹ️ How It Works")
        st.write("1. Upload a plant photo\n2. Click 'Run Diagnosis'\n3. Get instant species ID, disease detection & care plan")

    # Main content
    col_left, col_right = st.columns([1, 3], gap="large")

    with col_left:
        st.markdown("### 📸 Upload Photo")
        
        uploaded_file = st.file_uploader(
            "Choose a plant image",
            type=["jpg", "jpeg", "png"],
            help="JPG, JPEG, or PNG format"
        )

        image = None
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Your Photo", use_container_width=True)
                st.success(f"✅ {uploaded_file.name} loaded")
                st.session_state.run_diagnosis = True
            except Exception as e:
                st.error(f"❌ Error: {e}")

 

    with col_right:
        st.markdown("### 📊 Results")

        if st.session_state.get("run_diagnosis", False):
            with st.spinner("🔄 Analyzing..."):
                try:
                    diagnosis = call_backend(uploaded_file, language=language)
                except RuntimeError as e:
                    st.error(f"❌ {str(e)}")
                    return

            plant = diagnosis.get("plant_species", {})
            disease = diagnosis.get("disease_detection", {})
            final_diag = diagnosis.get("final_diagnosis", {})
            plan = diagnosis.get("treatment_plan", {})
            metadata = diagnosis.get("metadata", {})
            
            leaf_boxes = []
            individual_leaves = []
            if disease and disease.get("leaf_analysis"):
                leaf_analysis = disease["leaf_analysis"]
                leaf_boxes = leaf_analysis.get("leaf_boxes", []) or []
                individual_leaves = leaf_analysis.get("individual_leaves", []) or []
                
                if not leaf_boxes and individual_leaves:
                    leaf_boxes = [leaf.get('leaf_box') for leaf in individual_leaves if leaf.get('leaf_box')]
            
            try:
                uploaded_file.seek(0)
                original_image = Image.open(uploaded_file).convert('RGB')
                annotated_image = draw_bounding_boxes(original_image, leaf_boxes, individual_leaves)
                
                st.markdown("---")
                st.markdown("### 🔍 Detected Regions")
                
                # Status message
                if leaf_boxes:
                    st.success(f"✅ Found {len(leaf_boxes)} leaf region(s)")
                elif individual_leaves:
                    leaf_count = sum(1 for leaf in individual_leaves if leaf.get('leaf_box'))
                    if leaf_count > 0:
                        st.info(f"Found {leaf_count} leaf region(s)")
                
                # Side-by-side image comparison
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(original_image, caption="Original", use_container_width=True)
                with img_col2:
                    st.image(annotated_image, caption="🟢 Leaves  |  🔴 Lesions", use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not visualize regions: {e}")

            st.markdown("---")

            # Use tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Diagnosis", "🌿 Species", "📊 Model Results", "🍃 Leaf Analysis"])
            
            # Tab 1: Final Diagnosis
            with tab1:
                if final_diag:
                    show_final_diagnosis(final_diag)
                else:
                    st.info("No diagnosis available")
            
            # Tab 2: Plant Species
            with tab2:
                if plant:
                    show_plant_species(plant)
                    
                    # Show multi-model comparison if available
                    multi_model = plant.get("multi_model_comparison")
                    if multi_model:
                        st.markdown("---")
                        show_multi_model_comparison(multi_model)
                else:
                    st.info("No species identification available")
            
            # Tab 3: Model Results (Raw predictions)
            with tab3:
                model_results = diagnosis.get("model_results", {})
                if model_results:
                    st.markdown("### 📊 Raw Model Predictions")
                    st.caption("Detailed predictions from AI models")
                    
                    # Species predictions table
                    species_table = model_results.get("species_predictions", [])
                    if species_table:
                        st.markdown("#### 🌿 Species Classifier")
                        df_species = pd.DataFrame(species_table)
                        st.dataframe(df_species, use_container_width=True, hide_index=True)
                        st.caption(f"Top {len(species_table)} species predictions from unified classifier")
                        st.markdown("---")
                    
                    # Disease predictions table
                    disease_table = model_results.get("disease_predictions", [])
                    if disease_table:
                        st.markdown("#### 🦠 Disease Classifier")
                        df_disease = pd.DataFrame(disease_table)
                        st.dataframe(df_disease, use_container_width=True, hide_index=True)
                        st.caption(f"Top {len(disease_table)} disease predictions from PlantVillage-trained model")
                else:
                    st.info("No model results available")
            
            # Tab 4: Leaf Analysis
            with tab4:
                disease = diagnosis.get("disease_detection", {})
                if disease and disease.get("leaf_analysis"):
                    leaf_analysis = disease["leaf_analysis"]
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Leaves Detected", leaf_analysis.get('num_leaves_detected', 0))
                    with col2:
                        st.metric("Overall Health", f"{leaf_analysis.get('overall_health_score', 0):.0%}")
                    with col3:
                        status = "⚠️ Issues" if leaf_analysis.get('has_potential_issues') else "✅ Good"
                        st.metric("Status", status)
                    
                    st.markdown("---")
                    
                    # Individual leaf analysis table
                    model_results = diagnosis.get("model_results", {})
                    leaf_table = model_results.get("leaf_analysis_table", [])
                    if leaf_table:
                        st.markdown("#### Individual Leaf Details")
                        df_leaves = pd.DataFrame(leaf_table)
                        st.dataframe(df_leaves, use_container_width=True, hide_index=True)
                        st.caption("Detailed analysis for each detected leaf")
                else:
                    st.info("No leaf analysis available")
            
            # Metadata footer
            if metadata:
                ms = metadata.get("processing_time_ms")
                if ms:
                    st.caption(f"⏱️ Processing time: {ms}ms")


if __name__ == "__main__":
    if "run_diagnosis" not in st.session_state:
        st.session_state.run_diagnosis = False
    
    main()