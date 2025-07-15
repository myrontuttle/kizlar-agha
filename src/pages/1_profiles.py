import streamlit as st
import json
from db import init_db, get_profiles, get_profile, save_profile, delete_profile, get_model_selection
from models import Profile, ProfileSchema, Scenario, ScenarioSchema
from ml.swarm_ui import list_image_models, seed_from_image
from ml.llm import list_ollama_models
from utils import settings

init_db()

st.title("Profile Management")

profiles = get_profiles()

selection = get_model_selection()
llm_model = selection.llm_model if selection and selection.llm_model else settings.INFERENCE_DEPLOYMENT_NAME
image_model = selection.image_model if selection and selection.image_model else None

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
        st.markdown(details_str)
    with row[1]:
        if getattr(profile, "profile_image_path", None):
            images = profile.get_images()
            if images:
                for img in images:
                    if img:
                        img_seed = seed_from_image(img)
                        st.image(img, caption=f"{img_seed}", width=120)
                        with st.popover(f"View Full Image {img_seed}"):
                            st.image(img, caption=f"{img_seed}")
                        if st.button(f"Make Main Image {img_seed}", key=f"main_image_{i}_{img_seed}"):
                            profile.generate_main_profile_image(image_model, img_seed)
                            save_profile(profile)
                            st.success(f"Main image set for profile {getattr(profile, 'name', i)}")

        else:
            if st.button("Generate Profile Images", key=f"generate_profile_images_{i}"):
                try:
                    st.session_state["profile_image_path"] = profile.generate_sample_profile_images(image_model)
                    save_profile(profile)
                    st.success("Refresh to see images")
                except Exception as e:
                    st.error(f"Error generating images: {e}")
    with row[2]:
        if st.button("Remove", key=f"remove_{i}"):
            profile.delete_images()  # Delete images associated with the profile
            delete_profile(profile.id)
            st.warning(f"Removed profile {getattr(profile, 'name', i)}. Refresh to see changes.")

# Add a text input for region_request above the button
region_request = st.text_input("Which region should this new profile be from?", value="")

# Generate a new profile
if st.button("Generate New Profile"):
    profile = Profile.generate_profile(llm_model, region_request)
    st.session_state["generated_profile"] = profile

# Show the generated profile if it exists in session_state
if "generated_profile" in st.session_state:
    profile = st.session_state["generated_profile"]
    st.write("Generated Profile:")
    st.json(vars(profile))
    if st.button("Save Generated Profile"):
        saved = save_profile(profile)
        st.success(f"Profile saved (ID: {saved.id})")
        # Optionally, remove from session_state after saving
        del st.session_state["generated_profile"]

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

    # --- Image Model Selection ---
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
    profile_image_path = st.text_area(
        "Profile Image Path",
        value=profile_data.profile_image_path or ""
    )

    # --- Chat Model Selection ---
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
        profile_data.profile_image_path = profile_image_path
        profile_data.chat_model = chat_model
        saved = save_profile(profile_data)
        st.success(f"Profile saved (ID: {saved.id})")

# Display the generated image if available
filenames = st.session_state.get("profile_image_path", "")
if filenames:
    st.write(f"Generated image files: `{filenames}`")
    if isinstance(filenames, str):
        images = json.loads(filenames)
        if isinstance(images, list) and images:
            for img in images:
                if img:
                    st.image(img, caption=f"{img}")
