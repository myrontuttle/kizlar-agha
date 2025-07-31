import streamlit as st
from db import get_profiles, get_scenarios, get_scenarios_for_profile, init_db, get_model_usage, save_message, save_model_usage, save_profile, get_messages
from services import generate_profile, generate_profile_image_description, generate_sample_profile_images, generate_scenario, generate_scenario_images, generate_scene_descriptions, stop_models, set_status_to_idle, voice_response
from ml.llm import list_ollama_models
from ml.swarm_ui import list_image_models, seed_from_image
from models import ModelUsageSchema

init_db()

st.write("# Kizlar Agha")

st.write(
    """Use the menu on the left to create profiles, scenarios, and chat with characters."""
)

# Load current usage from DB
usage = get_model_usage()
if usage is None:
    usage = {"llm_model": "", "image_model": "", "status": "idle"}
elif not isinstance(usage, dict):
    usage = usage.model_dump()

st.write(f"Model usage status: **{usage['status']}**")
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
st.markdown("---")

# Generate a full set of profiles, scenarios, and images
if st.button("Surprise Me"):
    if usage['llm_model'] == "":
        llm_models = list_ollama_models()
        usage['llm_model'] = llm_models[0]['name']
        st.write(f"LLM model set to: {usage['llm_model']}")
        save_model_usage(ModelUsageSchema(**usage))
    if usage['image_model'] == "":
        image_models = list_image_models()
        usage['image_model'] = image_models[0]['name']
        st.write(f"Image model set to: {usage['image_model']}")
        save_model_usage(ModelUsageSchema(**usage))
    # Generate up to 5 random profiles if there are not already 5 profiles
    profiles = get_profiles()
    requests = ["Latina", "East Asian", "Northern European Blonde", "Redhead", "Eastern European Brunette"]
    if len(profiles) < len(requests):
        # Use only the last requests needed to get to 5 total profiles
        requests = requests[len(profiles):]
        st.write(f"Generating {len(requests)} random profiles...")
        for request in requests:
            st.write(f"Generating {request} profile")
            generate_profile(usage['llm_model'], request)
            st.success(f"{request} profile generated.")
    # Generate profile image descriptions and a scenario for each profile
    for profile in get_profiles():
        if not profile.profile_image_description:
            st.write(f"Generating image for profile {profile.name}")
            generate_profile_image_description(profile.id, usage['llm_model'])
            st.success(f"Image description generated for profile {profile.name}.")
        else:
            st.write(f"Profile {profile.name} already has an image description.")
        if not get_scenarios_for_profile(profile.id):
            st.write(f"Generating scenario for profile {profile.name}")
            generate_scenario(profile.id, usage['llm_model'])
            st.success(f"Generated scenario for: {profile.name}")
    # Generate scene descriptions for each scenario
    for scenario in get_scenarios():
        if not scenario.scene_descriptions:
            st.write(f"Generating scene descriptions for scenario {scenario.title}")
            generate_scene_descriptions(scenario.id, usage['llm_model'])
            st.success(f"Generated scene descriptions for: {scenario.title}. Generating images.")
        else:
            st.write(f"Scenario {scenario.title} already has scene descriptions.")
    # Generate a profile image for each profile
    for profile in get_profiles():
        if not profile.profile_image_path:
            st.write(f"Generating profile image for: {profile.name}")
            profile = generate_sample_profile_images(profile.id, usage['image_model'], num_images=1)
            st.success(f"Profile image generated for: {profile.name}.")
        images = profile.get_images()
        if not profile.image_seed:
            if images:
                # Select first image seed as profile image_seed so we can generate scenario images
                profile.image_seed = seed_from_image(images[0])
                save_profile(profile)
                st.success(f"Image seed generated for {profile.name}.")
            else:
                profile.image_seed = -1
                save_profile(profile)
                st.warning(f"No image seed generated for {profile.name}. Using -1 (random).")
    # Generate scenario images for each scenario
    for scenario in get_scenarios():
        if not scenario.images:
            st.write(f"Generating images for scenario {scenario.title}")
            generate_scenario_images(scenario.id, usage['image_model'])
            st.success(f"Images generated for: {scenario.title}.")
        else:
            st.write(f"Scenario {scenario.title} already has images.")
        # Voice messages if not already voiced
        messages = get_messages(scenario.id)
        for msg in messages:
            if not msg.speech:
                st.write(f"Voicing speech for message {msg.id} in scenario {scenario.title}")
                voice_response(msg.content, profile.voice)
                st.success(f"Speach generated for message {msg.id}.")
st.markdown("---")
