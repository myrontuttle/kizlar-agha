import streamlit as st
import json
from db import init_db, get_profiles, get_profile, save_profile, delete_profile
from profile import Profile, ProfileSchema
from ml.swarm_ui import image_from_prompt

init_db()

st.title("Profile Management")

profiles = get_profiles()  # Should return a list of Profile or ProfileSchema instances

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
                        st.image(img, caption=f"{img}", width=120)
        else:
            if st.button("Generate Images", key=f"generate_images_{i}"):
                try:
                    st.session_state["profile_image_path"] = profile.generate_images()
                    save_profile(profile)
                    st.success("Refresh to see images")
                except Exception as e:
                    st.error(f"Error generating images: {e}")
    with row[2]:
        if st.button("Remove", key=f"remove_{i}"):
            delete_profile(profile.id)
            st.warning(f"Removed profile {getattr(profile, 'name', i)}. Refresh to see changes.")

# Generate a new profile
if st.button("Generate New Profile"):
    profile = Profile.generate_profile()
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

with st.form("profile_form"):
    name = st.text_input("Name", value=profile_data.name)
    background = st.text_input("Background", value=profile_data.background or "")
    personality = st.text_input("Personality", value=profile_data.personality or "")
    interests = st.text_input("Interests", value=profile_data.interests or "")
    physical_characteristics = st.text_input(
        "Physical Characteristics",
        value=profile_data.physical_characteristics or ""
    )
    image_model = st.text_input("Image Model", value=profile_data.image_model or "")
    image_seed = st.text_input("Image Seed", value=profile_data.image_seed or "")
    profile_image_path = st.text_input(
        "Profile Image Path",
        value=profile_data.profile_image_path or ""
    )
    chat_model = st.text_input("Chat Model", value=profile_data.chat_model or "")

    st.markdown("---")
    save = st.form_submit_button("Save", type="primary")

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

