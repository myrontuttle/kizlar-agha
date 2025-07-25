import streamlit as st
import json
import threading
from db import init_db, get_profiles, get_profile, save_profile, delete_profile, get_model_usage
from models import Profile, ProfileSchema, Scenario, ScenarioSchema
from services import generate_profile, generate_profile_image_description, generate_sample_profile_images, generate_main_profile_image, stop_models, set_status_to_idle
from ml.swarm_ui import list_image_models, seed_from_image
from ml.llm import list_ollama_models
from utils import settings

init_db()

st.title("Profile Management")

profiles = get_profiles()

usage = get_model_usage()
llm_model = usage.llm_model if usage and usage.llm_model else settings.INFERENCE_DEPLOYMENT_NAME
image_model = usage.image_model if usage and usage.image_model else None
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

# Display existing profiles in a table format
cols = st.columns([3, 2, 1])
cols[0].markdown("**Profile Details**")
cols[1].markdown("**Images**")
cols[2].markdown("**Remove**")

for i, profile in enumerate(profiles):
    # Prepare profile details as a string or dict
    if hasattr(profile, "model_dump"):
        details = profile.model_dump()
    else:
        details = vars(profile)
    details_str = "\n".join(f"**{k}**: {v}" for k, v in details.items() if k != "profile_image_path")

    # Display each row
    row = st.columns([3, 2, 1])
    with row[0]:
        st.markdown(f"**{profile.name}**<br>"
                    f"*Background:* {profile.background}<br>"
                    f"*Personality:* {profile.personality}<br>"
                    f"*Interests:* {profile.interests}<br>"
                    f"*Physical Characteristics:* {profile.physical_characteristics}",
                    unsafe_allow_html=True)
    with row[1]:
        if not getattr(profile, "profile_image_description", None):
            if st.button("Generate Profile Image Description", key=f"generate_profile_image_description_{i}", disabled=(status != "idle")):
                try:
                    threading.Thread(target=generate_profile_image_description, args=(profile.id, llm_model), daemon=True).start()
                    st.success("Profile Image Description started in background. Refresh to see progress.")
                except Exception as e:
                    st.error(f"Error generating image description: {e}")
        if getattr(profile, "profile_image_path", None):
            images = profile.get_images()
            if images:
                for img in images:
                    if img:
                        img_seed = seed_from_image(img)
                        try:
                            st.image(img, caption=f"{img_seed}", width=120)
                            with st.popover(f"View Full Image {img_seed}"):
                                st.image(img, caption=f"{img_seed}")
                            if st.button(f"Make Main Image {img_seed}", key=f"main_image_{i}_{img_seed}", disabled=(status != "idle")):
                                threading.Thread(target=generate_main_profile_image, args=(profile.id, image_model, img_seed), daemon=True).start()
                                st.info("Image generation started in the background. Refresh to see progress.")
                                break  # Exit loop after setting main image
                        except Exception as e:
                            st.error(f"Error displaying image {img_seed}: {e}")
                if st.button(f"Delete All Images {getattr(profile, 'name', i)}", key=f"delete_images_{i}"):
                    profile.delete_images()
                    save_profile(profile)
                    st.success(f"All images deleted for profile {getattr(profile, 'name', i)}")
        else:
            if st.button("Generate Profile Images", key=f"generate_profile_images_{i}", disabled=(status != "idle")):
                threading.Thread(target=generate_sample_profile_images, args=(profile.id, image_model), daemon=True).start()
                st.info("Image generation started in the background. Refresh to see progress.")
    with row[2]:
        if st.button("Remove", key=f"remove_{i}"):
            profile.delete_images()  # Delete images associated with the profile
            delete_profile(profile.id)
            st.warning(f"Removed profile {getattr(profile, 'name', i)}. Refresh to see changes.")

# --- Profile creation form ---
st.markdown("---")
st.header("Create New Profile")

# Add a text input for region_request above the button
special_requests = st.text_input("Any special requests for this profile (e.g., region, hair color)?", value="")

# Generate a new profile
if st.button("Generate New Profile", disabled=(status != "idle")):
    threading.Thread(target=generate_profile, args=(llm_model, special_requests), daemon=True).start()
    st.info("Profile generation started in the background. Refresh to see progress.")

profile_names = [f"{p.id}: {p.name}" for p in profiles]
selected = st.selectbox("Select a profile", ["New"] + profile_names)

if selected == "New":
    profile_data = ProfileSchema(
        name="",
        background="",
        personality="",
        interests="",
        physical_characteristics="",
        image_model="",
        image_seed="",
        profile_image_description="",
        profile_image_path="",
        chat_model="",
    )
else:
    profile_id = int(selected.split(":")[0])
    profile = get_profile(profile_id)
    profile_data = ProfileSchema.model_validate(profile)

# --- Image Model Fetch Button (outside form) ---
if st.button("Fetch Image Models", key="fetch_image_models"):
    image_models = list_image_models()
    st.session_state["profile_image_models"] = image_models

# --- Chat Model Fetch Button (outside form) ---
if st.button("Fetch Chat Models", key="fetch_chat_models"):
    chat_models = list_ollama_models()
    st.session_state["profile_chat_models"] = chat_models

with st.form("profile_form"):
    name = st.text_input("Name", value=profile_data.name)
    background = st.text_area("Background", value=profile_data.background or "")
    personality = st.text_area("Personality", value=profile_data.personality or "")
    interests = st.text_area("Interests", value=profile_data.interests or "")
    physical_characteristics = st.text_area(
        "Physical Characteristics",
        value=profile_data.physical_characteristics or ""
    )

    # --- Image Model Usage ---
    st.markdown("**Image Model**")
    image_models = st.session_state.get("profile_image_models", [])
    if image_models:
        image_model = st.selectbox(
            "Select Image Model",
            image_models,
            index=image_models.index(profile_data.image_model) if profile_data.image_model in image_models else 0,
            key="profile_image_model_select"
        )
    else:
        image_model = profile_data.image_model or ""
        st.info("Click 'Fetch Image Models' to load available image models.")

    image_seed = st.text_input("Image Seed", value=profile_data.image_seed or "")
    profile_image_description = st.text_area(
        "Profile Image Description",
        value=profile_data.profile_image_description or ""
    )
    profile_image_path = st.text_area(
        "Profile Image Path",
        value=profile_data.profile_image_path or ""
    )

    # --- Chat Model Usage ---
    st.markdown("**Chat Model**")
    chat_models = st.session_state.get("profile_chat_models", [])
    if chat_models:
        chat_model = st.selectbox(
            "Select Chat Model",
            chat_models,
            index=chat_models.index(profile_data.chat_model) if profile_data.chat_model in chat_models else 0,
            key="profile_chat_model_select"
        )
    else:
        chat_model = profile_data.chat_model or ""
        st.info("Click 'Fetch Chat Models' to load available chat models.")

    st.markdown("---")
    save = st.form_submit_button("Save")

    if save:
        profile_data.name = name
        profile_data.background = background
        profile_data.personality = personality
        profile_data.interests = interests
        profile_data.physical_characteristics = physical_characteristics
        profile_data.image_model = image_model
        profile_data.image_seed = image_seed
        profile_data.profile_image_description = profile_image_description
        profile_data.profile_image_path = profile_image_path
        profile_data.chat_model = chat_model
        saved = save_profile(profile_data)
        st.success(f"Profile saved (ID: {saved.id})")
