import os
from typing import Optional
import time
import requests
import websocket
import json
import docker
import re
from utils import docker_client, logger, settings

FILES_DIR = os.path.join(os.path.dirname(__file__), "/kizlar-agha/files")
PROMPT = "A futuristic cityscape at sunset"

def start_swarmui_session() -> Optional[str]:
    """Start a new SwarmUI session. Returns session_id."""
    # Check if the SwarmUI container is available
    containers = docker_client.containers.list(all=True)
    if settings.SWARMUI_CONTAINER not in [c.name for c in containers]:
        logger.warning(f"{settings.SWARMUI_CONTAINER} not available. Available containers:")
        for container in containers:
            logger.warning(f"- {container.name}")
        return
    # Get the swarmui container
    swarmui = docker_client.containers.get(settings.SWARMUI_CONTAINER)
    if swarmui.status == "running":
        logger.info(f"Container {settings.SWARMUI_CONTAINER} is running.")
        # Try to get a session ID
        r = requests.post(
            f"{settings.SWARMUI_API_URL}/GetNewSession",
            json={},
            headers={'Content-type': 'application/json'}
        )
        if r.status_code == 200:
            return r.json().get("session_id")
        else:
            logger.error(f"Error getting session ID: {r.status_code} - {r.text}")
            return
    else:
        # If the container is not running, start it
        logger.info(f"Container {settings.SWARMUI_CONTAINER} is not running. Starting it..")
        try:
            docker_client.containers.get(settings.SWARMUI_CONTAINER).start()
        except docker.errors.APIError as e:
            logger.error(f"Error starting container {settings.SWARMUI_CONTAINER}: {e}")
            return
        # Try to get a session ID
        wait_time = 60
        for _ in range(wait_time):
            try:
                r = requests.post(
                    f"{settings.SWARMUI_API_URL}/GetNewSession",
                    json={},
                    headers={'Content-type': 'application/json'}
                )
                if r.status_code == 200:
                    return r.json().get("session_id")
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError(f"SwarmUI API did not become ready in {wait_time} seconds.")

def stop_swarmui():
    """Stop the SwarmUI container."""
    try:
        swarmui = docker_client.containers.get(settings.SWARMUI_CONTAINER)
        if swarmui.status == "running":
            logger.info(f"Stopping container {settings.SWARMUI_CONTAINER}..")
            swarmui.stop()
            logger.info("Container stopped.")
        else:
            logger.info(f"Container {settings.SWARMUI_CONTAINER} is not running.")
    except docker.errors.NotFound:
        logger.error(f"Container {settings.SWARMUI_CONTAINER} not found.")
    except docker.errors.APIError as e:
        logger.error(f"Error stopping container {settings.SWARMUI_CONTAINER}: {e}")

def get_current_status(session_id) -> Optional[dict]:
    """Get the current status of the SwarmUI session."""
    r = requests.post(
        url=f"{settings.SWARMUI_API_URL}/GetCurrentStatus",
        json={"session_id": session_id},
        headers={'Content-type': 'application/json'}
    )
    if r.status_code == 200:
        return r.json()
    else:
        logger.error(f"Error getting status: {r.status_code} - {r.text}")
        return None

def list_image_models(session_id: Optional[str] = None) -> Optional[list[str]]:
    """List available models in SwarmUI."""
    if not session_id:
        session_id = start_swarmui_session()
        if not session_id:
            logger.error("Failed to start SwarmUI session.")
            return None
    r = requests.post(
        url=f"{settings.SWARMUI_API_URL}/ListModels",
        json={
            "session_id": session_id,
            "path": "", # Empty path to use root
            "depth": 2
        },
        headers={'Content-type': 'application/json'}
    )
    if r.status_code == 200:
        models = r.json().get("files", [])
        return [model["name"] for model in models]
    else:
        logger.error(f"Error listing models: {r.status_code} - {r.text}")
        return None

def select_model(session_id, model):
    """Forcibly loads a model immediately on some or all backends."""
    r = requests.post(
        url=f"{settings.SWARMUI_API_URL}/SelectModel",
        json={
            "session_id": session_id,
            "model": model
        },
        headers={'Content-type': 'application/json'}
    )
    if r.status_code == 200:
        success = r.json().get("success")
        if (success):
            logger.info(f"Model {model} loaded")
        else:
            logger.warning(r.text)
    else:
        logger.error(f"Error loading model: {r.status_code} - {r.text}")

def select_model_ws(session_id, model):
    """Select a model using the SwarmUI SelectModelWS websocket API."""
    ws_url = f"{settings.SWARMUI_WS_URL}/SelectModelWS"
    ws = websocket.create_connection(ws_url)
    # Send the SelectModelWS command
    ws.send(json.dumps({
        "session_id": session_id,
        "model": model
    }))
    # Wait for a response
    while True:
        try:
            msg = ws.recv()
        except websocket.WebSocketConnectionClosedException as e:
            logger.warning(f"WebSocket closed: {e}")
            break
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            break
        if not msg.strip():
            continue  # skip empty messages
        try:
            event = json.loads(msg)
        except json.JSONDecodeError:
            logger.warning(f"Non-JSON message received: {msg!r}")
            continue  # skip non-JSON messages
        if event.get("success"):
            logger.info(f"Model {model} loaded via websocket.")
            break
        else:
            print(f"Failed to load model {model}: {event}")
    ws.close()

def generate_images_ws(
        session_id: str,
        model: str,
        prompt: str,
        neg_prompt: str = "",
        images=1,
        seed=-1,
        width=1024,
        height=1024,
        steps=1,
        cfgscale=1,
        sampler="euler_ancestral",
        scheduler=""
    ) -> list[str]:
    ws_url = f"{settings.SWARMUI_WS_URL}/GenerateText2ImageWS"
    ws = websocket.create_connection(ws_url)
    # Start image generation
    ws.send(json.dumps(
        {
            "session_id": session_id,
            "model": model,
            "prompt": prompt,
            "negativeprompt": neg_prompt,
            "images": images,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "cfgscale": cfgscale,
            "sampler": sampler,
            "scheduler": scheduler
        })
    )
    logger.info("Listening for T2I websocket updates..")
    image_paths = []
    complete = False
    while True:
        try:
            msg = ws.recv()
        except websocket.WebSocketConnectionClosedException as e:
            logger.info(f"WebSocket closed: {e}")
            break
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            break
        if not msg.strip():
            continue  # skip empty messages
        try:
            event = json.loads(msg)
        except json.JSONDecodeError:
            logger.warning(f"Non-JSON message received: {msg!r}")
            continue  # skip non-JSON messages
        if event.get("status"):
            # Print the status of the event
            logger.info(f"Status: {event['status']}")
        if event.get("gen_progress"):
            # Print the generation progress
            logger.info(f"Generation progress: batch_index: {event['gen_progress']['batch_index']}, "
                        f"overall_percent: {event['gen_progress']['overall_percent']}, "
                        f"current_percent: {event['gen_progress']['current_percent']}")
            if event['gen_progress']['overall_percent'] == 1.0 and event['gen_progress']['batch_index'] == str(images-1):
                complete = True
                continue
        if event.get("image"):
            image_info = event["image"]
            if isinstance(image_info, dict) and "image" in image_info:
                image_paths.append(image_info["image"])
            elif isinstance(image_info, str):
                image_paths.append(image_info)
            else:
                logger.warning(f"Unexpected image format: {image_info}")
            logger.debug(f"Image paths: {image_paths}")
            if complete:
                logger.info("Image generation complete.")
                break
    ws.close()
    return image_paths

def generate_seed_search(session_id, model, prompt: str, num_images=1):
    """Generate a set of images with random seed."""
    neg_prompt = "logo timestamp artist name artist watermark web address copyright " \
    "notice emblem comic title character border dog cow butterfly loli child kids teens text"
    return generate_images_ws(
        session_id=session_id,
        model=model,
        prompt=prompt,
        neg_prompt=neg_prompt,
        images=num_images,
        seed=-1,  # Random seed
        steps=9,
        cfgscale=3,
        sampler="euler_ancestral",
    )

def generate_target(session_id, model, prompt: str, seed):
    """Generate a target image with a specific seed."""
    neg_prompt = "logo timestamp artist name artist watermark web address copyright " \
    "notice emblem comic title character border dog cow butterfly loli child kids teens text"
    return generate_images_ws(
        session_id=session_id,
        model=model,
        prompt=prompt,
        neg_prompt=neg_prompt,
        images=1,
        seed=seed,  # Specific seed
        steps=12,
        cfgscale=4,
        sampler="euler_ancestral",
        scheduler="karras",
    )

def download_image(image_path, dest_folder):
    """Download an image from a URL to a specified folder."""
    image_url = image_path if image_path.startswith("http") else f"{settings.SWARMUI_BASE_URL}/{image_path}"
    logger.info(f"Downloading image from {image_url} to {dest_folder}")
    if not image_url:
        logger.error("Image URL is empty. Cannot download.")
        return None
    if not os.path.exists(dest_folder):
        logger.info(f"Destination folder {dest_folder} does not exist. Creating it.")
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder, exist_ok=True)
    filename = os.path.join(dest_folder, os.path.basename(image_url))
    r = requests.get(image_url)
    r.raise_for_status()
    with open(filename, "wb") as f:
        f.write(r.content)
    logger.info(f"Image downloaded to {filename}")
    return filename

def image_from_prompt(
        prompt: str,
        model: Optional[str] = None,
        preset: Optional[str] = None,
        seed: Optional[int] = None
    ):
    """Generate an image from a prompt using SwarmUI."""
    if not prompt:
        logger.error("Prompt is empty. Please provide a valid prompt.")
        return
    logger.info(f"Generating image from prompt: {prompt} with model: {model}, preset: {preset}, seed: {seed}")
    # Start a SwarmUI session
    session_id = start_swarmui_session()
    if not session_id:
        logger.error("Failed to start SwarmUI session.")
        return
    status = get_current_status(session_id)
    if status:
        logger.info(f"Current status: {status}")
    else:
        logger.error("Failed to get current status.")
        return
    if not model:
        models = list_image_models(session_id)
        logger.info(f"Available models: {models}")
        if not models:
            logger.error("No models available in SwarmUI.")
            return
        model = models[0]
    select_model_ws(session_id, model)
    image_files = []
    if preset == "seed_search":
        logger.info(f"Generating seed search images")
        image_urls = generate_seed_search(
            session_id,
            model,
            prompt,
        )
    elif preset == "target" and seed is not None:
        logger.info(f"Generating target image with seed: {seed}")
        image_urls = generate_target(
            session_id,
            model,
            prompt,
            seed,
        )
    else:
        logger.info("Generating random image")
        image_urls = generate_images_ws(
            session_id,
            model,
            prompt,
        )
    for image_url in image_urls:
        image_files.append(download_image(image_url, FILES_DIR))
    return image_files

def seed_from_image(image_path: str) -> Optional[int]:
    """Extract a seed from an image filename."""
    if not image_path:
        logger.error("Image path is empty. Cannot extract seed.")
        return None
    filename = os.path.basename(image_path)
    # Take the first digit sequence in the filename as the seed
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group(0))
    logger.warning(f"No seed found in image filename: {filename}")
    return None

if __name__ == "__main__":
    image_from_prompt(PROMPT)