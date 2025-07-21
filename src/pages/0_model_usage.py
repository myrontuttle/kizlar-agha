import streamlit as st
from db import init_db, get_model_usage, save_model_usage
from models import ModelUsage, ModelUsageSchema
from services import stop_and_clear_error
from ml.llm import list_ollama_models
from ml.swarm_ui import list_image_models

init_db()

st.title("Model Usage")

# Load current usage from DB
usage = get_model_usage()
if usage is None:
    usage = {"llm_model": "", "image_model": "", "status": "idle"}
elif not isinstance(usage, dict):
    usage = usage.model_dump()

st.write(f"Model usage status: **{usage['status']}**")
if st.button("Stop Models & Clear Errors", key="clear_error"):
    stop_and_clear_error()
    st.success("Errors cleared and status reset to idle.")

# --- LLM Model sage ---
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
        index=llm_models.index(usage["llm_model"]) if usage["llm_model"] in llm_models else 0,
        key="llm_model_select"
    )
    if st.button("Update LLM Model"):
        usage["llm_model"] = "ollama_chat/" + llm_model["name"]
        save_model_usage(ModelUsageSchema(**usage))
        st.success(f"LLM model updated to: {usage["llm_model"]}")
else:
    st.info("Click 'Fetch LLM Models' to load available LLM models.")

# --- Image Model Usage ---
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
        index=image_models.index(usage["image_model"]) if usage["image_model"] in image_models else 0,
        key="image_model_select"
    )
    if st.button("Update Image Model"):
        usage["image_model"] = image_model
        save_model_usage(ModelUsageSchema(**usage))
        st.success(f"Image model updated to: {image_model}")
else:
    st.info("Click 'Fetch Image Models' to load available image models.")

# --- Show current usage ---
st.markdown("---")
st.write("**Current usage:**")
st.json({
    "llm_model": usage["llm_model"],
    "image_model": usage["image_model"],
    "status": usage["status"]
})