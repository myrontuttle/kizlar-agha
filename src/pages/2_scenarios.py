import streamlit as st
import json
from db import (
    init_db, get_scenarios, get_scenario, save_scenario, delete_scenario,
    get_profiles, get_profile, get_model_selection
)
from models import Profile, ProfileSchema, Scenario, ScenarioSchema
from ml.llm import list_ollama_models
from ml.swarm_ui import list_image_models

init_db()

st.title("Scenario Management")

profiles = get_profiles()
profile_names = [f"{p.id}: {p.name}" for p in profiles]
scenarios = get_scenarios()
scenario_names = [f"{s.id}: {s.title}" for s in scenarios]

selection = get_model_selection()
llm_model = selection.llm_model if selection and selection.llm_model else ""
image_model = selection.image_model if selection and selection.image_model else ""

# --- Profile selection for scenario generation ---
selected_profile = st.selectbox("Select a profile for scenario", profile_names)
profile_id = int(selected_profile.split(":")[0])
character_profile = get_profile(profile_id)

# --- Scenario selection ---
selected_scenario = st.selectbox("Select a scenario", ["New"] + scenario_names)
if selected_scenario == "New":
    scenario_data = ScenarioSchema(
        profile_id=profile_id,
        title="",
        summary="",
        scene_summaries="[]",
        sample_dialog="",
        greeting="",
        scene_descriptions="[]",
        images="[]"
    )
else:
    scenario_id = int(selected_scenario.split(":")[0])
    scenario = get_scenario(scenario_id)
    scenario_data = ScenarioSchema.model_validate(scenario)

# --- Generate new scenario ---
st.markdown("### Generate New Scenario")
role_request = st.text_input("Role request", value="")
genre_request = st.text_input("Genre request", value="")
if st.button("Generate New Scenario"):
    scenario_obj = Scenario.generate_scenario(
        llm_model, character_profile, role_request, genre_request
    )
    st.session_state["generated_scenario"] = scenario_obj

# --- Show generated scenario ---
if "generated_scenario" in st.session_state:
    scenario_obj = st.session_state["generated_scenario"]
    st.write("Generated Scenario:")
    st.json(scenario_obj.model_dump())
    if st.button("Save Generated Scenario"):
        scenario_obj.profile_id = character_profile.id
        saved = save_scenario(scenario_obj)
        st.success(f"Scenario saved (ID: {saved.id})")
        del st.session_state["generated_scenario"]

# --- Scenario edit form ---
with st.form("scenario_form"):
    title = st.text_input("Title", value=scenario_data.title)
    summary = st.text_area("Summary", value=scenario_data.summary or "")
    scene_summaries = st.text_area("Scene Summaries (JSON array)", value=scenario_data.scene_summaries or "[]")
    sample_dialog = st.text_area("Sample Dialog", value=scenario_data.sample_dialog or "")
    greeting = st.text_area("Greeting", value=scenario_data.greeting or "")
    scene_descriptions = st.text_area("Scene Descriptions (JSON array)", value=scenario_data.scene_descriptions or "[]")
    images = st.text_area("Images (JSON array)", value=scenario_data.images or "[]")
    save = st.form_submit_button("Save Scenario")

    if save:
        scenario_data.title = title
        scenario_data.summary = summary
        scenario_data.scene_summaries = scene_summaries
        scenario_data.sample_dialog = sample_dialog
        scenario_data.greeting = greeting
        scenario_data.scene_descriptions = scene_descriptions
        scenario_data.images = images
        scenario_data.profile_id = character_profile.id
        saved = save_scenario(scenario_data)
        st.success(f"Scenario saved (ID: {saved.id})")

# --- Generate scene descriptions ---
if st.button("Generate Scene Descriptions"):
    scenario_obj = get_scenario(scenario_id) if selected_scenario != "New" else None
    if scenario_obj:
        try:
            scene_desc = scenario_obj.generate_scene_descriptions(llm_model)
            save_scenario(scenario_obj)
            st.success("Scene descriptions generated and saved.")
        except Exception as e:
            st.error(f"Error generating scene descriptions: {e}")

# --- Generate scenario images ---
if st.button("Generate Scenario Images"):
    scenario_obj = get_scenario(scenario_id) if selected_scenario != "New" else None
    if scenario_obj:
        try:
            images = scenario_obj.generate_scenario_images(image_model)
            save_scenario(scenario_obj)
            st.success("Scenario images generated and saved.")
        except Exception as e:
            st.error(f"Error generating scenario images: {e}")

# --- Display images ---
images = scenario_data.images
if images:
    try:
        image_list = json.loads(images)
        if isinstance(image_list, list) and image_list:
            st.write("Scenario Images:")
            for img in image_list:
                if img:
                    st.image(img, caption=f"{img}")
    except Exception as e:
        st.error(f"Error displaying images: {e}")

# --- Remove scenario ---
if selected_scenario != "New":
    if st.button("Remove Scenario"):
        delete_scenario(scenario_id)
        st.warning(f"Removed scenario {scenario_data.title}. Refresh to see changes.")