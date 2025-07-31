import streamlit as st
from db import init_db, get_model_usage, save_model_usage
from models import ModelUsage, ModelUsageSchema
from services import stop_models, set_status_to_idle
from ml.llm import list_ollama_models
from ml.swarm_ui import list_image_models
from utils import docker_client

init_db()

st.title("Model Usage")

# Load current usage from DB
usage = get_model_usage()
if usage is None:
    usage = {"llm_model": "", "image_model": "", "tts_model": "", "status": "idle"}
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

# --- TTS model usage ---
st.header("TTS Model")
tts_models = ['orpheus', 'chatterbox', 'kokoro']
tts_model = st.selectbox(
    "Select TTS Model",
    tts_models,
    index=0,  # Default to the first model
    key="tts_model_select"
)
if st.button("Update TTS Model"):
    usage["tts_model"] = tts_model
    save_model_usage(ModelUsageSchema(**usage))
    st.success(f"TTS model updated to: {tts_model}")

# --- Show current usage ---
st.markdown("---")
st.write("**Current usage:**")
st.json({
    "llm_model": usage["llm_model"],
    "image_model": usage["image_model"],
    "tts_model": usage["tts_model"],
    "status": usage["status"]
})

# --- Show containers ---
st.markdown("---")
containers = docker_client.containers.list(all=True)

# Display each container's name and status
for container in containers:
    st.write(f"**{container.name}** — Status: `{container.status}`")