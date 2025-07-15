import streamlit as st
from db import init_db, get_model_selection, save_model_selection
from models import ModelSelection, ModelSelectionSchema
from ml.llm import list_ollama_models
from ml.swarm_ui import list_image_models

init_db()

st.title("Model Selection")

# Load current selection from DB
selection = get_model_selection()
if selection is None:
    selection = {"llm_model": "", "image_model": ""}
elif not isinstance(selection, dict):
    selection = selection.model_dump()

# --- LLM Model Selection ---
st.header("LLM Model")
if st.button("Fetch LLM Models"):
    llm_models = list_ollama_models()
    st.session_state["llm_models"] = llm_models
else:
    llm_models = st.session_state.get("llm_models", [])

if llm_models:
    llm_model = st.selectbox(
        "Select LLM Model",
        llm_models,
        index=llm_models.index(selection["llm_model"]) if selection["llm_model"] in llm_models else 0,
        key="llm_model_select"
    )
    if st.button("Update LLM Model"):
        selection["llm_model"] = "ollama_chat/" + llm_model["name"]
        save_model_selection(ModelSelectionSchema(**selection))
        st.success(f"LLM model updated to: {selection["llm_model"]}")
else:
    st.info("Click 'Fetch LLM Models' to load available LLM models.")

# --- Image Model Selection ---
st.header("Image Model (SwarmUI)")
if st.button("Fetch Image Models"):
    image_models = list_image_models()
    st.session_state["image_models"] = image_models
else:
    image_models = st.session_state.get("image_models", [])

if image_models:
    image_model = st.selectbox(
        "Select Image Model",
        image_models,
        index=image_models.index(selection["image_model"]) if selection["image_model"] in image_models else 0,
        key="image_model_select"
    )
    if st.button("Update Image Model"):
        selection["image_model"] = image_model
        save_model_selection(ModelSelectionSchema(**selection))
        st.success(f"Image model updated to: {image_model}")
else:
    st.info("Click 'Fetch Image Models' to load available image models.")

# --- Show current selection ---
st.markdown("---")
st.write("**Current Selection:**")
st.json({
    "llm_model": selection["llm_model"],
    "image_model": selection["image_model"],
})