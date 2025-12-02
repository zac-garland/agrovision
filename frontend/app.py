import requests
import streamlit as st
from PIL import Image

BACKEND_URL = "http://localhost:5000/diagnose"


def call_backend(image_bytes, language="en"):
    """
    send the image to the backend /diagnose endpoint
    returns parsed json or raises an error
    """
    files = {"image": image_bytes}
    data = {"language": language}

    try:
        response = requests.post(BACKEND_URL, files=files, data=data, timeout=60)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"could not reach backend. error was: {e}")

    if response.status_code != 200:
        raise RuntimeError(
            f"backend returned status {response.status_code}: {response.text}"
        )

    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError("backend response was not valid json")

    if not payload.get("success", False):
        error_msg = payload.get("error", "unknown error from backend")
        raise RuntimeError(error_msg)

    return payload.get("diagnosis", {})


def show_plant_species(plant_species):
    """
    show plant species section
    expects plant_species dict with primary and top_5 fields
    """
    if not plant_species:
        return

    primary = plant_species.get("primary", {})
    top5 = plant_species.get("top_5", [])

    main_name = primary.get("name", "unknown")
    common_name = primary.get("common_name", "")
    conf = primary.get("confidence", 0.0)

    st.subheader("plant species")
    st.write(f"predicted species: **{main_name}**")
    if common_name:
        st.write(f"common name: {common_name}")
    st.write(f"model confidence: {conf:.0%}")

    if top5:
        labels = [item.get("name", "") for item in top5]
        values = [item.get("confidence", 0.0) for item in top5]
        st.markdown("top species probabilities")
        st.bar_chart({"confidence": values}, x=labels)


def show_disease_section(disease_detection):
    """
    show disease detection section
    """
    if not disease_detection:
        return

    primary = disease_detection.get("primary", {})
    disease_name = primary.get("disease", "unknown")
    common_name = primary.get("common_name", "")
    conf = primary.get("confidence", 0.0)

    st.subheader("disease detection")
    st.write(f"predicted disease: **{disease_name}**")
    if common_name:
        st.write(f"common name: {common_name}")
    st.write(f"model confidence: {conf:.0%}")

    affected = disease_detection.get("affected_area_percent")
    if affected is not None:
        st.write(f"estimated affected leaf area: {affected:.1f}%")

    # placeholder for heatmap or mask if backend sends one later
    if disease_detection.get("heatmap_image"):
        st.info("backend returned an affected region map. add image decoding here later.")


def show_final_diagnosis(final_diagnosis):
    """
    show final diagnosis and reasoning
    """
    if not final_diagnosis:
        return

    st.subheader("final diagnosis")

    condition = final_diagnosis.get("condition", "not available")
    conf = final_diagnosis.get("confidence", 0.0)
    severity = final_diagnosis.get("severity", "not specified")
    reasoning = final_diagnosis.get("reasoning", "")

    st.write(f"condition: **{condition}**")
    st.write(f"confidence: {conf:.0%}")
    st.write(f"severity: {severity}")

    if reasoning:
        with st.expander("why this diagnosis"):
            st.write(reasoning)


def show_treatment_plan(plan):
    """
    show treatment plan as simple steps
    """
    if not plan:
        return

    st.subheader("treatment plan")

    immediate = plan.get("immediate", [])
    week1 = plan.get("week_1", [])
    week23 = plan.get("week_2_3", [])
    monitoring = plan.get("monitoring", "")

    if immediate:
        st.markdown("**right now**")
        for step in immediate:
            st.write(f"- {step}")

    if week1:
        st.markdown("**this week**")
        for step in week1:
            st.write(f"- {step}")

    if week23:
        st.markdown("**weeks two and three**")
        for step in week23:
            st.write(f"- {step}")

    if monitoring:
        st.markdown("**monitoring notes**")
        st.write(monitoring)


def main():
    st.set_page_config(page_title="agrovision plant diagnosis", layout="wide")

    st.title("🌱 agrovision")
    st.caption("agentic plant diagnosis for home and hobby growers")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("upload a plant photo")
        uploaded_file = st.file_uploader(
            "choose a leaf or plant image", type=["jpg", "jpeg", "png"]
        )

        image = None
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="uploaded image", use_column_width=True)
            except Exception as e:
                st.error(f"could not open image. error was: {e}")

        language = st.selectbox("language", ["en"], index=0)
        run_button = st.button("run diagnosis")

    with col_right:
        st.subheader("results")

        if run_button:
            if uploaded_file is None:
                st.warning("please upload an image first.")
                return

            if image is None:
                st.error("image could not be loaded.")
                return

            with st.spinner("sending image to backend and waiting for diagnosis"):
                try:
                    image_bytes = uploaded_file.getvalue()
                    diagnosis = call_backend(image_bytes, language=language)
                except RuntimeError as e:
                    st.error(str(e))
                    return

            plant = diagnosis.get("plant_species", {})
            disease = diagnosis.get("disease_detection", {})
            final_diag = diagnosis.get("final_diagnosis", {})
            plan = diagnosis.get("treatment_plan", {})
            metadata = diagnosis.get("metadata", {})

            if plant:
                show_plant_species(plant)

            if disease:
                st.markdown("---")
                show_disease_section(disease)

            if final_diag:
                st.markdown("---")
                show_final_diagnosis(final_diag)

            if plan:
                st.markdown("---")
                show_treatment_plan(plan)

            if metadata:
                st.markdown("---")
                ms = metadata.get("processing_time_ms")
                if ms is not None:
                    st.caption(f"backend processing time: {ms} ms")


if __name__ == "__main__":
    main()
