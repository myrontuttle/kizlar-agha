import os
from typing import Optional
import time
import requests
import websocket
import json
import docker
from utils import docker_client

SWARMUI_CONTAINER = "swarmui"
SWARMUI_BASE_URL = "http://host.docker.internal:7801"
SWARMUI_API_URL = "http://host.docker.internal:7801/API"
SWARMUI_WS_URL = "ws://host.docker.internal:7801/API"
FILES_DIR = os.path.join(os.path.dirname(__file__), "../files")
PROMPT = "A futuristic cityscape at sunset"

def start_swarmui_session() -> Optional[str]:
    """Start a new SwarmUI session. Returns session_id."""
    # Check if the SwarmUI container is available
    containers = docker_client.containers.list(all=True)
    if SWARMUI_CONTAINER not in [c.name for c in containers]:
        print(f"{SWARMUI_CONTAINER} not available. Available containers:")
        for container in containers:
            print(f"- {container.name}")
        return
    # Get the swarmui container
    swarmui = docker_client.containers.get(SWARMUI_CONTAINER)
    if swarmui.status == "running":
        print(f"Container {SWARMUI_CONTAINER} is running.")
        # Try to get a session ID
        r = requests.post(
            f"{SWARMUI_API_URL}/GetNewSession",
            json={},
            headers={'Content-type': 'application/json'}
        )
        if r.status_code == 200:
            return r.json().get("session_id")
        else:
            print(f"Error getting session ID: {r.status_code} - {r.text}")
            return
    else:
        # If the container is not running, start it
        print(f"Container {SWARMUI_CONTAINER} is not running. Starting it...")
        try:
            docker_client.containers.get(SWARMUI_CONTAINER).start()
        except docker.errors.APIError as e:
            print(f"Error starting container {SWARMUI_CONTAINER}: {e}")
            return
        # Try to get a session ID
        wait_time = 60
        for _ in range(wait_time):
            try:
                r = requests.post(
                    f"{SWARMUI_API_URL}/GetNewSession",
                    json={},
                    headers={'Content-type': 'application/json'}
                )
                if r.status_code == 200:
                    return r.json().get("session_id")
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError(f"SwarmUI API did not become ready in {wait_time} seconds.")

def get_current_status(session_id) -> Optional[dict]:
    """Get the current status of the SwarmUI session."""
    r = requests.post(
        url=f"{SWARMUI_API_URL}/GetCurrentStatus",
        json={"session_id": session_id},
        headers={'Content-type': 'application/json'}
    )
    if r.status_code == 200:
        return r.json()
    else:
        print(f"Error getting status: {r.status_code} - {r.text}")
        return None

def list_models(session_id) -> Optional[list[str]]:
    """List available models in SwarmUI."""
    r = requests.post(
        url=f"{SWARMUI_API_URL}/ListModels",
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
        print(f"Error listing models: {r.status_code} - {r.text}")
        return None

def select_model(session_id, model_name):
    """Forcibly loads a model immediately on some or all backends."""
    r = requests.post(
        url=f"{SWARMUI_API_URL}/SelectModel",
        json={
            "session_id": session_id,
            "model": model_name
        },
        headers={'Content-type': 'application/json'}
    )
    if r.status_code == 200:
        success = r.json().get("success")
        if (success):
            print(f"Model {model_name} loaded")
        else:
            print(r.text)
    else:
        print(f"Error loading model: {r.status_code} - {r.text}")

def select_model_ws(session_id, model_name):
    """Select a model using the SwarmUI SelectModelWS websocket API."""
    ws_url = f"{SWARMUI_WS_URL}/SelectModelWS"
    ws = websocket.create_connection(ws_url)
    # Send the SelectModelWS command
    ws.send(json.dumps({
        "session_id": session_id,
        "model": model_name
    }))
    # Wait for a response
    try:
        while True:
            msg = ws.recv()
            data = json.loads(msg)
            if data.get("success"):
                print(f"Model {model_name} loaded via websocket.")
                break
            else:
                print(f"Failed to load model {model_name}: {data}")
    except websocket.WebSocketConnectionClosedException as e:
        print(f"WebSocket closed: {e}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        ws.close()

def generate_images_ws(
        session_id: str,
        images: int,
        prompt: str,
        model_name: str,
        width=1024,
        height=1024,
        steps=3,
        cfgscale=2,
        seed=-1
    ):
    ws_url = f"{SWARMUI_WS_URL}/GenerateText2ImageWS"
    ws = websocket.create_connection(ws_url)
    # Start image generation
    ws.send(json.dumps(
        {
            "session_id": session_id,
            "images": images,
            "prompt": prompt,
            "model": model_name,
            "width": width,
            "height": height,
            "steps": steps,
            "cfgscale": cfgscale,
            "seed": seed
        })
    )
    print("Listening for T2I websocket updates...")
    while True:
        try:
            msg = ws.recv()
        except websocket.WebSocketConnectionClosedException as e:
            print(f"WebSocket closed: {e}")
            break
        except Exception as e:
            print(f"WebSocket error: {e}")
            break
        if not msg.strip():
            continue  # skip empty messages
        try:
            event = json.loads(msg)
        except json.JSONDecodeError:
            print(f"Non-JSON message received: {msg!r}")
            continue  # skip non-JSON messages
        if event.get("status"):
            # Print the status of the event
            print(f"Status: {event['status']}")
        if event.get("gen_progress"):
            # Print the generation progress
            print(f"Generation progress: {event['gen_progress']}")
        if event.get("image"):
            image_info = event["image"]
            if isinstance(image_info, dict) and "image" in image_info:
                image_path = image_info["image"]
            elif isinstance(image_info, str):
                image_path = image_info
            else:
                print(f"Unexpected image format: {image_info}")
            break #TODO: Handle multiple images
    ws.close()
    return image_path

def download_image(image_path, dest_folder):
    """Download an image from a URL to a specified folder."""
    image_url = image_path if image_path.startswith("http") else f"{SWARMUI_BASE_URL}/{image_path}"
    print(f"Downloading image from {image_url} to {dest_folder}")
    if not image_url:
        print("Image URL is empty. Cannot download.")
        return None
    if not os.path.exists(dest_folder):
        print(f"Destination folder {dest_folder} does not exist. Creating it.")
    if not os.path.isdir(dest_folder):
        print(f"Destination {dest_folder} is not a directory. Cannot download image.")
        return None
    os.makedirs(dest_folder, exist_ok=True)
    filename = os.path.join(dest_folder, os.path.basename(image_url))
    r = requests.get(image_url)
    r.raise_for_status()
    with open(filename, "wb") as f:
        f.write(r.content)
    print(f"Image downloaded to {filename}")
    return filename

def image_from_prompt(prompt: str):
    """Generate an image from a prompt using SwarmUI."""
    if not prompt:
        print("Prompt is empty. Please provide a valid prompt.")
        return
    print(f"Generating image from prompt: {prompt}")
    # Start a SwarmUI session
    session_id = start_swarmui_session()
    if not session_id:
        print("Failed to start SwarmUI session.")
        return
    status = get_current_status(session_id)
    if status:
        print(f"Current status: {status}")
    else:
        print("Failed to get current status.")
        return
    models = list_models(session_id)
    print(f"Available models: {models}")
    select_model_ws(session_id, models[0])
    image_url = generate_images_ws(
        session_id,
        1,
        prompt,
        models[0]
    )
    filename = download_image(image_url, FILES_DIR)
    return filename

if __name__ == "__main__":
    image_from_prompt(PROMPT)