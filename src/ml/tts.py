
import re
import time
import docker
import os
import requests
from utils import docker_client, logger, settings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)

FILES_DIR = os.path.join(os.path.dirname(__file__), "/kizlar-agha/files/speech")
TTS_CONTAINER = "orpheus-fastapi"
ADDITIONAL_TTS_CONTAINER = "orpheus-fastapi-llama-cpp-server-1"

def start_tts_container():
    """Start the TTS container if not already running."""
    containers = docker_client.containers.list(all=True)
    if TTS_CONTAINER not in [c.name for c in containers]:
        logger.info(f"{TTS_CONTAINER} container not found.")
        for container in containers:
            logger.info(f"- {container.name}")
        return
    tts = docker_client.containers.get(TTS_CONTAINER)
    if tts.status == "running":
        logger.info(f"{TTS_CONTAINER} container is already running.")
        return
    else:
        try:
            docker_client.containers.get(TTS_CONTAINER).start()
            if TTS_CONTAINER == "orpheus-fastapi" and ADDITIONAL_TTS_CONTAINER in [c.name for c in containers]:
                docker_client.containers.get(ADDITIONAL_TTS_CONTAINER).start()
            # Wait for the container to be ready
            while docker_client.containers.get(TTS_CONTAINER).status != "running":
                logger.info(f"{TTS_CONTAINER} container is "
                            f"{docker_client.containers.get(TTS_CONTAINER).status}")
                time.sleep(1)
            if TTS_CONTAINER == "orpheus-fastapi" and ADDITIONAL_TTS_CONTAINER in [c.name for c in containers]:
                while docker_client.containers.get(ADDITIONAL_TTS_CONTAINER).status != "running":
                    logger.info(f"{ADDITIONAL_TTS_CONTAINER} container is "
                                f"{docker_client.containers.get(ADDITIONAL_TTS_CONTAINER).status}")
                    time.sleep(1)
            logger.info(f"{TTS_CONTAINER} container started successfully.")
        except docker.errors.APIError as e:
            logger.error(f"Error starting {TTS_CONTAINER} container: {e}")

def stop_tts_container():
    """Stop the TTS container."""
    try:
        container = docker_client.containers.get(TTS_CONTAINER)
        if container.status == "running":
            container.stop()
            logger.info(f"{TTS_CONTAINER} container stopped successfully.")
        else:
            logger.info(f"{TTS_CONTAINER} container is not running.")
        if TTS_CONTAINER == "orpheus-fastapi":
            additional_container = docker_client.containers.get(ADDITIONAL_TTS_CONTAINER)
            if additional_container.status == "running":
                additional_container.stop()
                logger.info(f"{ADDITIONAL_TTS_CONTAINER} container stopped successfully.")
            else:
                logger.info(f"{ADDITIONAL_TTS_CONTAINER} container is not running.")
    except docker.errors.NotFound:
        logger.error(f"{TTS_CONTAINER} container not found.")
    except docker.errors.APIError as e:
        logger.error(f"Error stopping {TTS_CONTAINER} container: {e}")

@retry(
    wait=wait_fixed(30),
    stop=stop_after_attempt(2),
    after=lambda retry_state: logger.warning(
        f"Retrying get_tts_audio due to error: {retry_state.outcome.exception()}"
    ),
)
def get_tts_audio(input: str, model: str="orpheus", voice: str = "", response_format: str = "wav", speed: float = 0.5) -> str:
    """Get TTS audio from the TTS service."""
    start_tts_container()
    logger.info(f"Getting TTS audio for input: {input}, model: {model}, voice: {voice}, response_format: {response_format}, speed: {speed}")
    r = requests.post(
        url=settings.TTS_API_URL,
        json={
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed
        },
        headers={'Content-type': 'application/json'}
    )
    if r.status_code != 200:
        logger.error(f"Error getting TTS audio: {r.text}")
        return ""
    audio_data = r.content
    # Create FILES_DIR if it doesn't already exist
    os.makedirs(FILES_DIR, exist_ok=True)
    audio_file_path = os.path.join(FILES_DIR, f"{int(time.time())}.{response_format}")
    with open(audio_file_path, "wb") as audio_file:
        audio_file.write(audio_data)
    logger.info(f"TTS audio saved to {audio_file_path}")
    return audio_file_path

def remove_action_text(content):
    """Remove anything between asterisks from the content."""
    if not content:
        return ""
    # Remove text between asterisks (e.g., *action text*) and any line breaks
    content = re.sub(r"\*.*?\*", "", content, flags=re.DOTALL)
    content = re.sub(r"\n+", " ", content)  # Replace multiple newlines with a single space
    content = content.strip()  # Remove leading and trailing whitespace
    return content
