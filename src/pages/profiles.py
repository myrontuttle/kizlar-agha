import streamlit as st
from db import init_db, get_profiles, get_profile, save_profile
from profile import ProfileSchema
from swarm_ui import image_from_prompt

init_db()

# Add custom CSS for button alignment
st.markdown(
    """
    <style>
    /* This will apply margin to the second submit button in the form */
    [data-testid="stFormSubmitButton"]:nth-of-type(1) button {
        margin-top: 1.7em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

    col1, col2 = st.columns([3, 1])
    with col1:
        physical_characteristics = st.text_input(
            "Physical Characteristics",
            value=profile_data.physical_characteristics or "",
            key="physical_characteristics"
        )
    with col2:
        generate = st.form_submit_button("Generate Image")

    chat_model = st.text_input("Chat Model", value=profile_data.chat_model or "")
    personality = st.text_input("Personality", value=profile_data.personality or "")
    background = st.text_input("Background", value=profile_data.background or "")
    st.markdown("---")
    save = st.form_submit_button("Save", type="primary")

    if save:
        profile_data.name = name
        profile_data.image_model = image_model
        profile_data.physical_characteristics = physical_characteristics
        profile_data.profile_image_path = st.session_state.get("profile_image_path", "")
        profile_data.chat_model = chat_model
        profile_data.personality = personality
        profile_data.background = background
        saved = save_profile(profile_data)
        st.success(f"Profile saved (ID: {saved.id})")

    if generate:
        filename = image_from_prompt(physical_characteristics)
        if filename:
            st.session_state["profile_image_path"] = filename
            st.success(f"Image generated: {filename}")
        else:
            st.error("Failed to generate image.")

# Display the generated image if available
filename = st.session_state.get("profile_image_path", "")
if filename:
    st.write(f"Generated image file: `{filename}`")
    st.image(filename, caption="Generated Image")

