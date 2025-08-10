import streamlit as st
import json
import threading
from db import (
    get_scenarios_for_profile, init_db, get_scenarios, get_scenario, save_scenario, delete_scenario,
    get_profiles, get_profile, get_model_usage
)
from models import Profile, ProfileSchema, Scenario, ScenarioSchema
from services import generate_scenario, generate_scene_descriptions, generate_scenario_images, stop_models, set_status_to_idle

init_db()

st.title("Scenario Management")

profiles = get_profiles()
profile_names = [f"{p.id}: {p.name}" for p in profiles]
scenarios = get_scenarios()
scenario_names = [f"{s.id}: {s.title}" for s in scenarios]

usage = get_model_usage()
llm_model = usage.llm_model if usage and usage.llm_model else ""
image_model = usage.image_model if usage and usage.image_model else ""
status = usage.status if usage else "idle"

st.write(f"Model usage status: **{status}**")
# --- Button row ---
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if st.button("Stop Models", key="stop_models"):
        stop_models()
        st.success("Models stopped.")
with btn_col2:
    if st.button("Set Status to Idle", key="set_idle"):
        set_status_to_idle()
        st.success("Status set to idle.")

# --- Profile selection for scenario generation ---
selected_profile = st.selectbox("Select a profile for scenario", profile_names)
if not selected_profile:
    st.error("Please create a profile to continue.")
    st.stop()
profile_id = int(selected_profile.split(":")[0])
character_profile = get_profile(profile_id)

# --- Scenario selection ---
scenarios = get_scenarios_for_profile(profile_id)
scenario_names = [f"{s.id}: {s.title}" for s in scenarios]
selected_scenario = st.selectbox("Select a scenario", ["New"] + scenario_names)
if selected_scenario == "New":
    scenario_data = ScenarioSchema(
        profile_id=profile_id,
        title="",
        summary="",
        scene_summaries="[]",
        invitation="",
        scene_descriptions="[]",
        images="[]"
    )
else:
    scenario_id = int(selected_scenario.split(":")[0])
    scenario = get_scenario(scenario_id)
    scenario_data = ScenarioSchema.model_validate(scenario)

# --- Generate new scenario ---
st.markdown("### Generate New Scenario")
ns_col1, ns_col2, ns_col3 = st.columns([3, 1, 1])
with ns_col1:
    special_requests = st.text_input("Special requests", value="")
with ns_col2:
    # checkbox to generate images
    gen_images = st.checkbox("Generate Images", value=True)
with ns_col3:
    if st.button("Generate New Scenario", disabled=(status != "idle")):
        threading.Thread(target=generate_scenario, args=(profile_id, llm_model, special_requests, gen_images), daemon=True).start()
        st.info("Scenario generation started in the background. Refresh to see progress.")

# --- Scenario edit form ---
with st.form("scenario_form"):
    title = st.text_input("Title", value=scenario_data.title)
    summary = st.text_area("Summary", value=scenario_data.summary or "")
    scene_summaries = st.text_area("Scene Summaries (JSON array)", value=scenario_data.scene_summaries or "[]")
    invitation = st.text_area("Invitation", value=scenario_data.invitation or "")
    scene_descriptions = st.text_area("Scene Descriptions (JSON array)", value=scenario_data.scene_descriptions or "[]")
    images = st.text_area("Images (JSON array)", value=scenario_data.images or "[]")
    save = st.form_submit_button("Save Scenario")

    if save:
        scenario_data.title = title
        scenario_data.summary = summary
        scenario_data.scene_summaries = scene_summaries
        scenario_data.invitation = invitation
        scenario_data.scene_descriptions = scene_descriptions
        scenario_data.images = images
        scenario_data.profile_id = character_profile.id
        saved = save_scenario(scenario_data)
        st.success(f"Scenario saved (ID: {saved.id})")

# --- Generate scene descriptions ---
if st.button("Generate Scene Descriptions", disabled=(status != "idle" or selected_scenario == "New")):
    scenario_obj = get_scenario(scenario_id) if selected_scenario != "New" else None
    if scenario_obj:
        threading.Thread(target=generate_scene_descriptions, args=(scenario_obj.id, llm_model), daemon=True).start()
        st.info("Scenario generation started in the background. Refresh to see progress.")

# --- Generate scenario images ---
if character_profile.image_seed is None:
    st.warning("Please set an image seed in the profile to generate images.")
else:
    if st.button("Generate Scenario Images", disabled=(status != "idle" or scenario_data.scene_descriptions == "" or scenario_data.scene_descriptions == "[]")):
        scenario_obj = get_scenario(scenario_id) if selected_scenario != "New" else None
        if scenario_obj:
            threading.Thread(target=generate_scenario_images, args=(scenario_obj.id, image_model), daemon=True).start()
            st.info("Image generation started in the background. Refresh to see progress.")

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
        scenario.delete_images()  # Delete images associated with the scenario=
        delete_scenario(scenario_id)
        st.warning(f"Removed scenario {scenario_data.title}. Refresh to see changes.")