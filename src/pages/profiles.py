import streamlit as st
from db import init_db, get_profiles, get_profile, save_profile
from profile import ProfileSchema

init_db()

st.title("Profile Editor")

profiles = get_profiles()
profile_names = [f"{p.id}: {p.name}" for p in profiles]
selected = st.selectbox("Select a profile", ["New"] + profile_names)

if selected == "New":
    profile_data = ProfileSchema(
        name="",
        image_model="",
        physical_characteristics="",
        profile_image_path="",
        chat_model="",
        personality="",
        background=""
    )
else:
    profile_id = int(selected.split(":")[0])
    profile = get_profile(profile_id)
    profile_data = ProfileSchema.from_orm(profile)

with st.form("profile_form"):
    name = st.text_input("Name", value=profile_data.name)
    image_model = st.text_input("Image Model", value=profile_data.image_model or "")
    physical_characteristics = st.text_input("Physical Characteristics", value=profile_data.physical_characteristics or "")
    profile_image_path = st.text_input("Profile Image Path", value=profile_data.profile_image_path or "")
    chat_model = st.text_input("Chat Model", value=profile_data.chat_model or "")
    personality = st.text_input("Personality", value=profile_data.personality or "")
    background = st.text_input("Background", value=profile_data.background or "")
    st.markdown("---")
    submitted = st.form_submit_button("Save")

    if submitted:
        profile_data.name = name
        profile_data.image_model = image_model
        profile_data.physical_characteristics = physical_characteristics
        profile_data.profile_image_path = profile_image_path
        profile_data.chat_model = chat_model
        profile_data.personality = personality
        profile_data.background = background
        saved = save_profile(profile_data)
        st.success(f"Profile saved (ID: {saved.id})")