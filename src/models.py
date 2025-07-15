import os
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from pydantic import BaseModel
from base import Base
import json
from ml.llm import InferenceLLMConfig, stop_ollama_container, extract_json_from_response, remove_thinking
from ml.swarm_ui import image_from_prompt
from utils import settings, logger

Base = declarative_base()


class ModelSelection(Base):
    __tablename__ = "model_selections"
    id = Column(Integer, primary_key=True)
    llm_model = Column(String, nullable=False)
    image_model = Column(String, nullable=True)

    def model_dump(self):
        return {
            "id": self.id,
            "llm_model": self.llm_model,
            "image_model": self.image_model,
        }

class ModelSelectionSchema(BaseModel):
    id: int | None = None
    llm_model: str
    image_model: str | None = None

    class Config:
        from_attributes = True

class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    background = Column(String)
    personality = Column(String)
    interests = Column(String)
    physical_characteristics = Column(String)
    image_model = Column(String)
    image_seed = Column(String)
    profile_image_path = Column(String)
    chat_model = Column(String)
    scenarios = relationship("Scenario", back_populates="profile")

    def model_dump(self, *args, **kwargs):
        """Override to return a dictionary representation of the profile."""
        return {
            "id": self.id,
            "name": self.name,
            "background": self.background,
            "personality": self.personality,
            "interests": self.interests,
            "physical_characteristics": self.physical_characteristics,
            "image_model": self.image_model,
            "image_seed": self.image_seed,
            "profile_image_path": self.profile_image_path,
            "chat_model": self.chat_model
        }

    @classmethod
    def generate_profile(cls, llm_model: str, region_request: str) -> "Profile":
        """Generate a profile based on the following prompts."""
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
                    "(i.e., as a string of words separated by commas, no specific measurements like height) "
                    "with an adjective and their gender (e.g., sexy woman), body description, skin tone, "
                    "hair color and style, facial features, and eye color using as few words as possible "
                    "(30 words or less). Infer their 'name' based on where they were born. "
                    "Use proper json format with the keys: 'name', 'background', "
                    "'personality', 'interests', and 'physical_characteristics', but make sure the "
                    "values are all strings without internal quotation marks.",
                },
                {
                    "role": "user",
                    "content": "Generate the profile of a sexy and seductive woman for erotic scenarios. "
                    f"She should be from {region_request}. Be as sexually "
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
        logger.info(f"Profile data generated: {profile_data}. Stopping Ollama.")
        return cls(
            name=profile_data.get("name", "Default Name"),
            background=profile_data.get("background"),
            personality=profile_data.get("personality"),
            interests=profile_data.get("interests"),
            physical_characteristics=profile_data.get("physical_characteristics")
        )

    def generate_sample_profile_images(self, image_model: str) -> str:
        """Generate a set of images based on the profile's physical characteristics."""
        if not self.physical_characteristics:
            raise ValueError("Cannot generate image: physical_characteristics is empty.")
        logger.info("Rendering images from physical characteristics, background, and interests.")
        prompt = f"SFW. {self.physical_characteristics} {self.background} {self.interests}"
        filenames = image_from_prompt(prompt, model=image_model, preset="seed_search")
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
        self.profile_image_path = filenames
        logger.info(f"Profile image path set to: {self.profile_image_path}")
        return self.profile_image_path

    def get_images(self):
        """Get the list of image paths associated with this profile."""
        if not self.profile_image_path:
            return []
        try:
            return json.loads(self.profile_image_path)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in profile_image_path: {self.profile_image_path}")
            return []

    def delete_images(self):
        """Delete the profile images from the filesystem."""
        image_paths = self.get_images()
        if not image_paths:
            logger.warning("No images to delete.")
            return
        for path in image_paths:
            try:
                os.remove(path)
                logger.info(f"Deleted image: {path}")
            except OSError as e:
                logger.error(f"Error deleting image {path}: {e}")
        self.profile_image_path = None
        logger.info("All profile images deleted and profile_image_path cleared.")

    def generate_main_profile_image(self, image_model: str, image_seed: str):
        """Generates a high fidelity profile image based on the profile's physical characteristics."""
        if not self.physical_characteristics:
            raise ValueError("Cannot generate image: physical_characteristics is empty.")
        logger.info("Rendering images from physical characteristics, background, and interests.")
        prompt = f"SFW. {self.physical_characteristics} {self.background} {self.interests}"
        filenames = image_from_prompt(prompt, model=image_model, preset="target", seed=image_seed)
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
        self.delete_images()
        self.profile_image_path = filenames
        self.image_seed = image_seed
        logger.info(f"Profile image path set to: {self.profile_image_path}")
        return self.profile_image_path


class ProfileSchema(BaseModel):
    id: int | None = None
    name: str
    background: str | None = None
    personality: str | None = None
    interests: str | None = None
    physical_characteristics: str | None = None
    image_model: str | None = None
    image_seed: str | None = None
    profile_image_path: str | None = None
    chat_model: str | None = None

    class Config:
        from_attributes = True

if __name__ == "__main__":
    profile = Profile.generate_profile()
    logger.info(f"Generated profile: {profile.name}")


class Scenario(Base):
    __tablename__ = "scenarios"
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profiles.id'), nullable=False)
    title = Column(String, nullable=False)
    summary = Column(String)
    scene_summaries = Column(String)
    sample_dialog = Column(String)
    greeting = Column(String)
    scene_descriptions = Column(String)
    images = Column(String)
    profile = relationship("Profile", back_populates="scenarios")

    def model_dump(self, *args, **kwargs):
        """Override to return a dictionary representation of the scenario."""
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "title": self.title,
            "summary": self.summary,
            "scene_summaries": self.scene_summaries,
            "sample_dialog": self.sample_dialog,
            "greeting": self.greeting,
            "scene_descriptions": self.scene_descriptions,
            "images": self.images,
        }

    @classmethod
    def generate_scenario(
        cls,
        llm_model: str,
        character_profile: Profile,
        role_request: str,
        genre_request: str
        ) -> "Scenario":
        """Generate a scenario based on the following prompts."""
        llm = InferenceLLMConfig(
            model_name=llm_model,
            base_url=settings.INFERENCE_BASE_URL,
            api_key=settings.INFERENCE_API_KEY,
        )
        logger.info(f"Generating scenario using Ollama LLM: {llm.model_name}")
        response = llm.generate_from_messages(
            messages=[
                {
                    "role": "system",
                    "content": "As a scenario generator, your job is to develop an engaging role-playing "
                    "scenario involving the user and a character. Think of the scenario as an episode of "
                    "a tv show or a collection of scenes in a play. For the scenario you must provide a "
                    "short 'title', 2 to 3 sentence 'summary', 3 to 6 single sentence 'scene_summaries', "
                    "'sample_dialog' containing one sentence each between the user and the character "
                    "(prefaced with '#user:' or '#character:'), "
                    "and an initial single sentence 'greeting' from the character to the user to start the "
                    "scenario.  Creativity and conciseness are vital, as you must invent details that "
                    "make the scenario interesting and engaging while limiting the size of your response "
                    "to fit in an AI model's limited context window (short evocative phrases are best). "
                    "For scenarios consider the character's interests, background (e.g., cultural "
                    "activities, language lessons, holidays, traditions, and travel destinations based on "
                    "where they are from or grew up), role_request (if provided), and genre_request (if "
                    "provided). Avoid repeating previous scenarios. Creativity and conciseness are key. "
                    "Use proper json format with the keys: 'title', 'summary', 'scene_summaries', "
                    "'sample_dialog', and 'greeting', and make sure the values are all strings except for"
                    "'scene_summaries' which is an array of strings."
                },
                {
                    "role": "user",
                    "content": f"Character name: {character_profile.name}.\n"
                    f"Character background: {character_profile.background}.\n"
                    f"Character interests: {character_profile.interests}.\n"
                    f"Role request: {role_request}.\n"
                    f"Genre request: {genre_request}.\n"
                    "Generate a scenario in which the 'scene_summaries' gradually lead to the user and the "
                    "character involved in a sexual act."
                }
            ]
        )
        if not response:
            raise ValueError("Failed to generate scenario: No content in response")
        scenario_data = extract_json_from_response(response)
        if not scenario_data:
            logger.error(f"Failed to extract scenario data from response: {response}")
            raise ValueError("Failed to extract scenario data from response")
        logger.info(f"Scenario data generated: {scenario_data}. Stopping Ollama.")
        return cls(
            title=scenario_data.get("title", "Default Title"),
            profile_id=character_profile.id,
            summary=scenario_data.get("summary"),
            scene_summaries=scenario_data.get("scene_summaries"),
            sample_dialog=scenario_data.get("sample_dialog"),
            greeting=scenario_data.get("greeting")
        )

    def get_scene_summaries_as_array(self) -> list:
        """Get the plot points as an array."""
        if self.scene_summaries:
            try:
                return json.loads(self.scene_summaries)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode scene summaries: {self.scene_summaries}")
                return []
        return []

    def generate_scene_description(self, llm_model: str, scene_id: int, previous_scene_description: str = ""):
        """Generate a scene description based on the profile's physical characteristics and scene."""
        if not self.profile.physical_characteristics:
            raise ValueError("Cannot generate scene description: physical_characteristics is empty.")
        scene_summary = self.get_scene_summaries_as_array()[scene_id] if self.get_scene_summaries_as_array() else ""
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
                    "(i.e., a string of words and short phrases separated by commas) "
                    "that describes a visual scene of a character. Creativity and conciseness "
                    "are vital, as you must invent visual details that add depth to the scene while limiting "
                    "the size of your response to fit in an AI model's limited context window (single words "
                    "and evocative phrases are best). "
                    "Start the prompt with the character's physical characteristics. Infer where the scene "
                    "takes place and what the character is doing based on the scenario summary and scene summary. "
                    "Describe their clothing (or revealed body parts if their clothing has been removed) based on "
                    "where they are and what they're doing. List their facial expression and posture. List elements "
                    "of the scene background noting lighting and details of any other relevant objects in the scene. "
                    "Reference the previous scene description if provided to avoid discontinuity in clothing or "
                    "scene background unless the current scene summary calls for a change. Include only the visual "
                    "elements that would be captured in a photograph. Remove any unnecessary words like articles "
                    "and conjunctions. "
                    "Write the response as a single string of 100 words or less separated by commas and periods.",
                },
                {
                    "role": "user",
                    "content": f"Character physical characteristics: {self.profile.physical_characteristics}.\n"
                    f"Scenario summary: {self.summary}.\n"
                    f"Scene summary: {scene_summary}.\n"
                    f"Previous scene description: {previous_scene_description}.",
                },
            ]
        )
        #Strip out anything between <think>...</think> tags
        response = remove_thinking(response)
        if not response:
            raise ValueError("Failed to generate scene description: No content in response")
        logger.info(f"Generated scene description: {response}.\nStopping Ollama.")
        return response

    def generate_scene_descriptions(self, llm_model: str) -> str:
        """Generate the scenario's scene descriptions based on the scene summaries."""
        if not self.scene_summaries:
            raise ValueError("Cannot generate scene descriptions: scene_summaries is empty.")
        scene_summaries = self.get_scene_summaries_as_array()
        descriptions = []
        previous_description = ""
        for i, summary in enumerate(scene_summaries):
            description = self.generate_scene_description(llm_model, i, previous_description)
            descriptions.append(description)
            previous_description = description
        if not descriptions:
            raise ValueError("No scene descriptions generated from scene summaries.")
        # Save the descriptions as a proper json array to the scene_descriptions field
        self.scene_descriptions = json.dumps(descriptions)
        return self.scene_descriptions

    def generate_scenario_images(self, image_model: str) -> str:
        """Generate a set of images based on the scenario's scene descriptions."""
        if not self.scene_descriptions:
            raise ValueError("Cannot generate images: scene_descriptions is empty.")
        scene_descriptions = json.loads(self.scene_descriptions)
        images = []
        for description in scene_descriptions:
            image = image_from_prompt(
                description,
                model=image_model,
                preset="target",
                seed=self.profile.image_seed
            )
            if not image:
                logger.error(f"Failed to generate image for description: {description}")
                continue
            images.append(image)
        if not images:
            raise ValueError("No images generated from scene descriptions.")
        # Save the images as a proper json array to the images field
        self.images = json.dumps(images)
        return self.images


class ScenarioSchema(BaseModel):
    id: int | None = None
    profile_id: int
    title: str
    summary: str | None = None
    scene_summaries: str | None = None
    sample_dialog: str | None = None
    greeting: str | None = None
    scene_descriptions: str | None = None
    images: str | None = None

    class Config:
        from_attributes = True