import threading
import streamlit as st
import json
from db import (
    delete_message, get_messages, get_scenarios_for_profile, get_scenario,
    get_profiles, get_profile, get_model_usage, save_message
)
from services import respond_to_chat, stop_models, set_status_to_idle, add_message, voice_response
from models import MessageSchema

st.write("# Chat")

usage = get_model_usage()
llm_model = usage.llm_model if usage and usage.llm_model else ""
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

col1, col2 = st.columns(2)

# --- Profile selection for scenario generation ---
with col1:
    profiles = get_profiles()
    if not profiles:
        st.error("Please create a profile to continue.")
        st.stop()
    profile_names = [f"{p.id}: {p.name}" for p in profiles]
    selected_profile = st.selectbox("Select a profile for scenario", profile_names)
    if not selected_profile:
        st.error("Please select a profile to continue.")
        st.stop()
    profile_id = int(selected_profile.split(":")[0])
    character_profile = get_profile(profile_id)

    # --- Scenario selection ---
    scenarios = get_scenarios_for_profile(profile_id)
    if not scenarios:
        st.error("No scenarios available for the selected profile.")
        st.stop()
    if len(scenarios) == 1:
        scenario = scenarios[0]
        scenario_id = scenario.id
        st.write(f"Only one scenario available: {scenario.title}")
    else:
        scenario_names = [f"{s.id}: {s.title}" for s in scenarios]
        selected_scenario = st.selectbox("Select a scenario", scenario_names)
        scenario_id = int(selected_scenario.split(":")[0])
        scenario = get_scenario(scenario_id)

    # --- Summarize the profile, scenario, and scene ---
    st.write(f"**Profile:** {character_profile.name}")
    st.write(f"**Scenario:** {scenario.title} - {scenario.summary}")

# --- Scene selection ---
with col2:
    if not scenario:
        st.error("Please select a scenario to continue.")
        st.stop()
    scenes = scenario.get_scene_summaries_as_array()
    if not scenes:
        st.error("No scenes available in the selected scenario.")
        st.stop()
    scene_names = [f"Scene {i}: {s}" for i, s in enumerate(scenes)]
    selected_scene = st.selectbox("Select a scene", scene_names, index=0)
    scene_num = int(selected_scene.split(":")[0].split(" ")[1])
    # --- Show corresponding scene image ---
    if scenario.images:
        try:
            images = json.loads(scenario.images)
            # Flatten if needed (handles [["img1"], ["img2"]] or ["img1", "img2"])
            if images and isinstance(images[0], list):
                images = [item for sublist in images for item in sublist]
            if scene_num < len(images):
                st.image(images[scene_num], caption=f"{scenes[scene_num]}")
        except Exception as e:
            st.warning(f"Could not load scene image: {e}")

# --- Chat section ---
with st.container(height=400):
    # --- Chat history in a scrollable div ---
    if "messages" not in st.session_state or st.session_state.get("scenario_id") != scenario_id:
        previous_messages = get_messages(scenario_id)
        st.session_state.messages = [
            {"role": msg.role, "content": msg.content, "order": msg.order, "speech": msg.speech, "id": msg.id}
            for i, msg in enumerate(previous_messages)
        ]
        st.session_state.scenario_id = scenario_id

    # Track if a message was edited or deleted to rerun after change
    rerun_needed = False

    for idx, msg in enumerate(st.session_state.messages):
        col_msg, col_edit_delete = st.columns([9, 1])
        with col_msg:
            if msg['role'] == 'character':
                if 'speech' in msg and msg['speech']:
                    # If the message has speech, play it
                    st.audio(msg['speech'], format="audio/wav")
                else:
                    if st.button("Speak", disabled=(status != "idle"), key=f"voice_{msg['id']}"):
                        threading.Thread(target=voice_response, args=(msg['id'], character_profile.voice), daemon=True).start()
                        st.info("Speech generation started in the background")
                st.markdown(f"**{character_profile.name}:**<br>{msg['content']}", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div  style='text-align: right;'><strong>You:</strong><br>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
        with col_edit_delete:
            if st.button("✏️", key=f"edit_{idx}"):
                st.session_state["edit_index"] = idx
                st.session_state["edit_content"] = msg["content"]
                rerun_needed = True
            if st.button("🗑️", key=f"delete_{idx}"):
                # Remove from session state
                st.session_state.messages.pop(idx)
                # Remove from DB as well
                delete_message(msg['id'])
                rerun_needed = True
                break  # Prevent index errors after deletion
            if st.button("↺", key=f"regenerate_{idx}", disabled=(status != "idle")):
                # Regenerate the character's response
                if msg['role'] == 'character':
                    send_msg = st.session_state.messages[idx - 1] if idx > 0 else None
                    # Remove the current message from the list to avoid duplication
                    st.session_state.messages.pop(idx)
                    delete_message(msg['id'])
                else:
                    send_msg = msg  # Use the current message if it's from the user
                try:
                    character_response = respond_to_chat(
                        llm_model=llm_model,
                        profile_id=profile_id,
                        scenario_id=scenario_id,
                        scene_num=scene_num,
                        message=send_msg['content']
                    )
                    char_msg = add_message(scenario_id, "character", character_response)
                    st.session_state.messages.append(
                        {
                            "role": "character",
                            "content": character_response,
                            "order": char_msg.order,
                            "speech": char_msg.speech,
                            "id": char_msg.id
                        }
                    )
                    rerun_needed = True
                    break  # Prevent index errors after deletion
                except Exception as e:
                    st.error(f"Error regenerating message: {e}")

        # Show edit box if this message is being edited
        if st.session_state.get("edit_index") == idx:
            new_content = st.text_area(
                "Edit message", value=st.session_state.get("edit_content", ""), key=f"edit_box_{idx}"
            )
            if st.button("Save", key=f"save_{idx}"):
                st.session_state.messages[idx]["content"] = new_content
                # Update in DB as well
                message_data = MessageSchema(
                    id=msg['id'],
                    scenario_id=scenario_id,
                    role=msg['role'],
                    content=new_content,
                    order=msg['order'],
                    speech=msg['speech']
                )
                save_message(message_data)
                st.session_state["edit_index"] = None
                st.session_state["edit_content"] = ""
                rerun_needed = True
            if st.button("Cancel", key=f"cancel_{idx}"):
                st.session_state["edit_index"] = None
                st.session_state["edit_content"] = ""
                rerun_needed = True
        st.markdown("---")  # Separator line
    if rerun_needed:
        st.rerun()

# --- Clear the input before widget is instantiated ---
if st.session_state.get("clear_input", False):
    st.session_state["chat_input"] = ""
    st.session_state["clear_input"] = False

# --- Chat input at the bottom (outside scrollable area) ---
user_message = st.text_area("You:", key="chat_input", placeholder="Type your message here...")
if st.button("Send", key="send_message", disabled=(status != "idle")):
    try:
        add_message(scenario_id, "user", user_message)
        st.session_state.messages.append({"role": "user", "content": user_message})
        character_response = respond_to_chat(
            llm_model=llm_model,
            profile_id=profile_id,
            scenario_id=scenario_id,
            scene_num=scene_num,
            message=user_message
        )
        char_msg = add_message(scenario_id, "character", character_response)
        st.session_state.messages.append(
            {
                "role": "character",
                "content": character_response,
                "order": char_msg.order,
                "speech": char_msg.speech,
                "id": char_msg.id
            }
        )
        st.session_state["clear_input"] = True
        st.rerun()
    except Exception as e:
        st.session_state.messages.append({"role": "character", "content": f"Error: {e}"})
        st.session_state["clear_input"] = True
        st.rerun()