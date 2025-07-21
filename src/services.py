
import json
from models import Profile, Scenario
from db import get_model_usage, save_model_usage, get_profile, save_profile, get_scenario, save_scenario
from ml.llm import InferenceLLMConfig, stop_ollama_container, extract_json_from_response, remove_thinking
from ml.swarm_ui import image_from_prompt, stop_swarmui
from utils import settings, logger


def stop_models():
    """Stops models"""
    stop_ollama_container()
    stop_swarmui()

def set_status_to_idle():
    """Return status to idle"""
    usage = get_model_usage()
    if usage.status != "idle":
        logger.info("Clearing error state, returning status to idle.")
        usage.status = "idle"
        save_model_usage(usage)
    else:
        logger.info("No error state to clear, models already idle.")
    return usage.status

def generate_profile(llm_model: str, special_requests: str) -> "Profile":
    """Generate a profile based on the following prompts."""
    usage = get_model_usage()
    if usage.status != "idle":
        logger.warning("Model usage is not idle, cannot generate images.")
        return
    usage.status = "Generating Profile"
    save_model_usage(usage)
    try:
        llm = InferenceLLMConfig(
            model_name=llm_model,
            base_url=settings.INFERENCE_BASE_URL,
            api_key=settings.INFERENCE_API_KEY,
        )
        logger.info(f"Generating profile using Ollama LLM: {llm.model_name}")
        response = llm.generate_from_messages(
            messages=[
                {
                    "role": "system",
                    "content": "As a character profile generator, your job is to develop a "
                    "multi-dimensional persona for use in role playing scenarios with AI models. "
                    "You must provide details about the character's 'background', 'personality', 'interests', "
                    "and 'physical_characteristics' that cohesively work together to create a unique "
                    "individual. Creativity and conciseness are vital, as you must invent details "
                    "that add depth to the character while limiting the size of your response to fit "
                    "in an AI model's limited context window (single words and evocative phrases are "
                    "best). For 'background' include where they were born, use that to infer their "
                    "ethnicity, education, occupation, socioeconomic status, and a few significant life events "
                    "that shaped them (200 words or less). Use their background to infer their "
                    "'personality' traits, such as their temperament, desires, quirks, and how they interact "
                    "with others (100 words or less). Briefly list their 'interests' based on their "
                    "background and personality (75 words or less). List their 'physical_characteristics' "
                    "(based on ethnicity) as a prompt for an image generator "
                    "(i.e., generic, evocative sentence fragments, no specific measurements) "
                    "with an adjective and their gender (e.g., sexy woman), "
                    "body description, skin tone, hair color and style, facial features, and eye color using "
                    "as few words as possible (30 words or less). Infer their 'name' based on where they were "
                    "born. Only include their name in the 'name' field, in all other fields use their "
                    "pronoun (e.g., she, he, they). Use proper json format with the keys: 'name', 'background', "
                    "'personality', 'interests', and 'physical_characteristics', but make sure the "
                    "values are all strings without internal quotation marks.",
                },
                {
                    "role": "user",
                    "content": "Generate the profile of a sexy and seductive woman for erotic scenarios. "
                    f"{special_requests}. Be as sexually "
                    "explicit as possible for the 'physical_characteristics' including descriptions of her "
                    "breast size and legs for her body description to create a vivid image.",
                },
            ]
        )
        if not response:
            raise ValueError("Failed to generate profile: No content in response")
        profile_data = extract_json_from_response(response)
        if not profile_data:
            logger.error(f"Failed to extract profile data from response: {response}")
            raise ValueError("Failed to extract profile data from response")
        logger.info(f"Profile data generated: {profile_data}")
        save_profile(
            Profile(
                name=profile_data.get("name", "Default Name"),
                background=profile_data.get("background"),
                personality=profile_data.get("personality"),
                interests=profile_data.get("interests"),
                physical_characteristics=profile_data.get("physical_characteristics")
            )
        )
        usage.status = "idle"
    except Exception as e:
        logger.error(f"Error generating profile: {e}")
        usage.status = "Error generating profile"
    finally:
        save_model_usage(usage)
        logger.info(f"Profile generated")

def generate_profile_image_description(profile_id, llm_model: str) -> str:
    """Generate a description for the profile image based on the profile's physical characteristics."""
    profile = get_profile(profile_id)
    if not profile.physical_characteristics:
        raise ValueError("Cannot generate image description: physical_characteristics is empty.")
    usage = get_model_usage()
    if usage.status != "idle":
        logger.warning("Model usage is not idle, cannot generate images.")
        return
    usage.status = "Generating Profile Image Description"
    save_model_usage(usage)
    try:
        llm = InferenceLLMConfig(
            model_name=llm_model,
            base_url=settings.INFERENCE_BASE_URL,
            api_key=settings.INFERENCE_API_KEY,
        )
        logger.info(f"Generating profile image description using Ollama LLM: {llm.model_name}")
        response = llm.generate_from_messages(
            messages=[
                {
                    "role": "system",
                    "content": "As a profile image description generator, your job is to write a prompt "
                    "for an image generator that describes a picture of a character. "
                    "Infer where the picture takes place and what the character is doing based on their "
                    "interests and personality. Describe the picture generically."
                    "Start with the character's physical characteristics. Describe their clothing based on "
                    "where they are and what they're doing. "
                    "List their facial expression and posture. List elements of the picture's background noting "
                    "lighting and details of any other relevant objects in the scene. "
                    "Include only the visual elements that would be captured in a photograph. Replace the "
                    "character's name with an adjective and their gender (e.g., sexy woman)."
                    "Remove any unnecessary words like articles and conjunctions (e.g., with, and) or "
                    "non-visible text (e.g., internal feelings, thoughts). Write the response as a single "
                    "string of 100 words or less.",
                },
                {
                    "role": "user",
                    "content": f"Physical characteristics: {profile.physical_characteristics}.\n"
                    f"Interests: {profile.interests}.\n"
                    f"Personality: {profile.personality}.\n",
                },
            ]
        )
        if not response:
            raise ValueError("Failed to generate profile image description: No content in response")
        profile.profile_image_description = response
        save_profile(profile)
        usage.status = "idle"
    except Exception as e:
        logger.error(f"Error generating profile image description: {e}")
        usage.status = "Error generating profile image description"
    finally:
        save_model_usage(usage)
        logger.info(f"Generated profile image description: {response}")

def generate_sample_profile_images(profile_id, image_model):
    """Generate a set of images based on a profile's image description."""
    profile = get_profile(profile_id)
    if not profile.profile_image_description:
        raise ValueError("Cannot generate images: profile image description is empty.")
    usage = get_model_usage()
    if usage.status != "idle":
        logger.warning("Model usage is not idle, cannot generate images.")
        return
    logger.info(f"Starting background image generation for profile ID {profile_id} using model {image_model}")
    usage.status = "Generating Sample Profile Images"
    save_model_usage(usage)
    try:
        filenames = image_from_prompt(profile.profile_image_description, model=image_model, preset="seed_search")
        logger.info(f"Image(s) generated and saved to {filenames}")
        if not filenames:
            raise ValueError("Failed to generate images: No filenames returned")
        elif isinstance(filenames, list):
            # Convert list of filenames to a JSON string
            filenames = json.dumps(filenames)
        elif isinstance(filenames, str) and filenames.startswith("[") and "'" in filenames:
            # Replace single quotes with double quotes
            filenames = filenames.replace("'", '"')
        else:
            # If it's a single filename, wrap it in a list
            filenames = json.dumps([filenames])
        profile.profile_image_path = filenames
        save_profile(profile)
        logger.info(f"Profile image path set to: {profile.profile_image_path}")
        usage.status = "idle"
    except Exception as e:
        logger.error(f"Error generating images for profile ID {profile_id}: {e}")
        usage.status = "Error generating images"
    finally:
        save_model_usage(usage)
        logger.info(f"Background image generation completed for profile ID {profile_id}")
    return profile

def generate_main_profile_image(profile_id, image_model: str, image_seed: str):
    """Generates a high fidelity profile image based on the profile's image description."""
    profile = get_profile(profile_id)
    if not profile.profile_image_description:
        raise ValueError("Cannot generate image: profile_image_descriptiong is empty.")
    usage = get_model_usage()
    if usage.status != "idle":
        logger.warning("Model usage is not idle, cannot generate images.")
        return
    usage.status = "Generating Main Profile Image"
    save_model_usage(usage)
    try:
        filenames = image_from_prompt(profile.profile_image_description, model=image_model, preset="target", seed=image_seed)
        logger.info(f"Image(s) generated and saved to {filenames}")
        if not filenames:
            raise ValueError("Failed to generate images: No filenames returned")
        elif isinstance(filenames, list):
            # Convert list of filenames to a JSON string
            filenames = json.dumps(filenames)
        elif isinstance(filenames, str) and filenames.startswith("[") and "'" in filenames:
            # Replace single quotes with double quotes
            filenames = filenames.replace("'", '"')
        else:
            # If it's a single filename, wrap it in a list
            filenames = json.dumps([filenames])
        profile.delete_images()
        profile.profile_image_path = filenames
        profile.image_seed = image_seed
        save_profile(profile)
        logger.info(f"Profile image path set to: {profile.profile_image_path}")
        usage.status = "idle"
    except Exception as e:
        logger.error(f"Error generating images for profile ID {profile_id}: {e}")
        usage.status = "Error generating images"
    finally:
        save_model_usage(usage)
        logger.info(f"Background image generation completed for profile ID {profile_id}")
    return profile

def generate_scenario(profile_id, llm_model: str, special_requests: str) -> "Scenario":
    """Generate a scenario based on the following prompts."""
    profile = get_profile(profile_id)
    if not profile:
        raise ValueError("Cannot generate scenario: profile is empty.")
    usage = get_model_usage()
    if usage.status != "idle":
        logger.warning("Model usage is not idle, cannot generate scenario.")
        return
    usage.status = "Generating Scenario"
    save_model_usage(usage)
    try:
        llm = InferenceLLMConfig(
            model_name=llm_model,
            base_url=settings.INFERENCE_BASE_URL,
            api_key=settings.INFERENCE_API_KEY,
        )
        logger.info(f"Generating scenario with {profile.name}")
        response = llm.generate_from_messages(
            messages=[
                {
                    "role": "system",
                    "content": "As a scenario generator, your job is to develop an engaging role-playing "
                    "scenario involving the user and a character. Think of the scenario as an episode of "
                    "a tv show or a collection of scenes in a play. For the scenario you must provide a "
                    "short 'title', 2 to 3 sentence 'summary', 6 to 7 single sentence 'scene_summaries', "
                    "and an initial single sentence 'invitation' as a message from the character to the user "
                    "tempting/enticing/seducing them to start the scenario. Do not include or make up the user's "
                    "name, only the character's name. Use 'you' and 'your' in place of the user's name. "
                    "Creativity and conciseness are vital, as you must invent details that "
                    "make the scenario interesting and engaging while limiting the size of your response "
                    "to fit in an AI model's limited context window (short evocative phrases are best). "
                    "For scenarios consider the character's interests, background (e.g., cultural "
                    "activities, language lessons, holidays, traditions, and travel destinations based on "
                    "where they are from or grew up), and special_requests (if provided). For the invitation, "
                    "consider the character's style of speech based on where they are from or grew up. "
                    "Use proper json format with the keys: 'title', 'summary', 'scene_summaries', and "
                    "'invitation', and make sure the values are all strings wrapped in quoatation marks "
                    "except for 'scene_summaries' which is an array of strings."
                },
                {
                    "role": "user",
                    "content": f"Character name: {profile.name}.\n"
                    f"Character background: {profile.background}.\n"
                    f"Character interests: {profile.interests}.\n"
                    f"Special requests: {special_requests}.\n"
                    "Generate a scenario in which the 'scene_summaries' lead to the male user and "
                    "the female character involved in one or more sexual acts (e.g., blowjob, titjob, sex). "
                    "The final scenes should be them having sex and the post coitus afterglow."
                }
            ]
        )
        if not response:
            raise ValueError("Failed to generate scenario: No content in response")
        scenario_data = extract_json_from_response(response)
        if not scenario_data:
            logger.error(f"Failed to extract scenario data from response: {response}")
            raise ValueError("Failed to extract scenario data from response")
        save_scenario(
            Scenario(
                title=scenario_data.get("title", "Default Title"),
                profile_id=profile.id,
                summary=scenario_data.get("summary"),
                scene_summaries=scenario_data.get("scene_summaries"),
                invitation=scenario_data.get("invitation")
            )
        )
        usage.status = "idle"
    except Exception as e:
        logger.error(f"Error generating scenario: {e}")
        usage.status = f"Error generating scenario: {e}"
    finally:
        save_model_usage(usage)
        logger.info(f"Scenario data generated: {scenario_data}.")

def generate_scene_description(scenario_id, llm_model: str, scene_id: int, previous_scene_description: str = ""):
    """Generate a scene description based on the profile's physical characteristics and scene."""
    scenario = get_scenario(scenario_id)
    if not scenario.profile.physical_characteristics:
        raise ValueError("Cannot generate scene description: physical_characteristics is empty.")
    usage = get_model_usage()
    if usage.status != "idle":
        logger.warning("Model usage is not idle, cannot generate images.")
        return
    scene_summary = scenario.get_scene_summaries_as_array()[scene_id] if scenario.get_scene_summaries_as_array() else ""
    usage.status = "Generating Scenario Description"
    save_model_usage(usage)
    try:
        llm = InferenceLLMConfig(
            model_name=llm_model,
            base_url=settings.INFERENCE_BASE_URL,
            api_key=settings.INFERENCE_API_KEY,
        )
        logger.info(f"Generating scene description for scene_id: {scene_id} using Ollama LLM: {llm.model_name}")
        response = llm.generate_from_messages(
            messages=[
                {
                    "role": "system",
                    "content": "As a scene generator, your job is to write a prompt for an image generator "
                    "(i.e., string of words, short phrases separated by commas) "
                    "that describes a visual scene of a character. Creativity and conciseness "
                    "are vital, as you must invent visual details that add depth to the scene while limiting "
                    "the size of your response to fit in an AI model's limited context window (adjective noun, "
                    " evocative phrase segments are best). "
                    "Start the prompt with the character's physical characteristics. Always include consistent "
                    "characteristics like hair color and eye color. Infer where the scene "
                    "takes place and what the character is doing based on the scenario summary and scene summary. "
                    "Describe their clothing (or revealed body parts if their clothing has been removed) based on "
                    "where they are and what they're doing maintaining consistent clothing color from previous "
                    "scenes. List their facial expression and posture. List elements "
                    "of the scene background noting lighting and details of any other relevant objects in the scene. "
                    "Reference the previous scene description if provided to avoid discontinuity in clothing or "
                    "scene background unless the current scene summary calls for a change. Include only the visual "
                    "elements that would be captured in a photograph. Replace the character's name with an "
                    "adjective and their gender (e.g., sexy woman). Remove any unnecessary words like articles "
                    "and conjunctions (e.g., with, and) or non-visible text (e.g., feelings, thoughts). "
                    "Write the response as a single string of 100 words or less.",
                },
                {
                    "role": "user",
                    "content": f"Character physical characteristics: {scenario.profile.physical_characteristics}.\n"
                    f"Scenario summary: {scenario.summary}.\n"
                    f"Scene summary: {scene_summary}.\n"
                    f"Previous scene description: {previous_scene_description}.",
                },
            ]
        )
        #Strip out anything between <think>...</think> tags
        response = remove_thinking(response)
        if not response:
            raise ValueError("Failed to generate scene description: No content in response")
        usage.status = "idle"
    except Exception as e:
        logger.error(f"Error generating scene description: {e}")
        usage.status = "Error generating scene description"
    finally:
        save_model_usage(usage)
        logger.info(f"Generated scene description: {response}")
    return response

def generate_scene_descriptions(scenario_id, llm_model: str) -> str:
    """Generate the scenario's scene descriptions based on the scene summaries."""
    scenario = get_scenario(scenario_id)
    if not scenario.scene_summaries:
        raise ValueError("Cannot generate scene descriptions: scene_summaries is empty.")
    scene_summaries = scenario.get_scene_summaries_as_array()
    descriptions = []
    previous_description = ""
    for i, summary in enumerate(scene_summaries):
        description = scenario.generate_scene_description(llm_model, i, previous_description)
        descriptions.append(description)
        previous_description = description
    if not descriptions:
        raise ValueError("No scene descriptions generated from scene summaries.")
    # Save the descriptions as a proper json array to the scene_descriptions field
    scenario.scene_descriptions = json.dumps(descriptions)
    save_scenario(scenario)
    return scenario

def generate_scenario_images(scenario_id, image_model: str) -> str:
    """Generate a set of images based on a scenario's scene descriptions."""
    scenario = get_scenario(scenario_id)
    if not scenario.scene_descriptions or scenario.scene_descriptions == "[]":
        raise ValueError("Cannot generate images: scene_descriptions is empty.")
    usage = get_model_usage()
    if usage.status != "idle":
        logger.warning("Model usage is not idle, cannot generate images.")
        return
    scene_descriptions = scenario.get_scene_descriptions()
    logger.debug(f"Generating scenario images for {scene_descriptions}")
    images = []
    usage.status = "Generating Scenario Images"
    save_model_usage(usage)
    try:
        for i, description in enumerate(scene_descriptions):
            prompt = ""
            if i == 0:
                prompt = description
            if i == 1:
                prompt = f"{description} pov"
            if i == 2 or i == 3:
                prompt = f"{description} pov, erotic"
            if i > 3:
                prompt = f"{description} pov, erotic, NSFW"
            image = image_from_prompt(
                prompt,
                model=image_model,
                preset="target",
                seed=scenario.profile.image_seed
            )
            if not image:
                logger.error(f"Failed to generate image for description: {prompt}")
                continue
            images.append(image)
        logger.info(f"Image(s) generated and saved to {images}")
        if not images:
            raise ValueError("No images generated from scene descriptions.")
        # Save the images as a proper json array to the images field
        scenario.images = json.dumps(images)
        save_scenario(scenario)
        logger.info(f"Scenario images saved to: {scenario.images}")
        usage.status = "idle"
    except Exception as e:
        logger.error(f"Error generating images for scenario ID {scenario_id}: {e}")
        usage.status = "Error generating images"
    finally:
        save_model_usage(usage)
        logger.info(f"Image generation completed for scenario ID {scenario_id}")
    return scenario
