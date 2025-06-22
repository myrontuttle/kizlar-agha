from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
from pydantic import BaseModel
import json
from ml.llm import InferenceLLMConfig, stop_ollama_container, extract_json_from_response, remove_thinking
from ml.swarm_ui import image_from_prompt
from utils import settings, logger

Base = declarative_base()

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
    def generate_profile(cls) -> "Profile":
        """Generate a profile based on the following prompts with profile image."""
        llm = InferenceLLMConfig(
            model_name=settings.INFERENCE_DEPLOYMENT_NAME,
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
                    "best). For 'background' include where and when they were born, use that to infer their "
                    "ethnicity, education, occupation, socioeconomic status, and a few significant life events "
                    "that shaped them (200 words or less). Use their background to infer their "
                    "'personality' traits, such as their temperament, desires, quirks, and how they interact "
                    "with others (100 words or less). Briefly describe their 'interests' based on their "
                    "background and personality (50 words or less). List their 'physical_characteristics' "
                    "(based on ethnicity) as a prompt for an image generator "
                    "(i.e., as a string of words separated by commas) "
                    "with an adjective and their gender (e.g., sexy woman), body description, skin tone, "
                    "hair color and style, facial features, and eye color using as few words as possible (25 words or less). "
                    "Infer their 'name' based on where and when they were born. "
                    "Use proper json format with the keys: 'name', 'background', "
                    "'personality', 'interests', and 'physical_characteristics', but make sure the "
                    "values are all strings.",
                },
                {
                    "role": "user",
                    "content": "Generate the profile of a sexy and seductive woman for erotic scenarios. "
                    "She can be any ethnicity except black or African. Be as sexually explicit as possible "
                    "for the 'physical_characteristics' including descriptions of her breasts and legs "
                    "for her body description to create a vivid image.",
                },
            ]
        )
        if not response:
            raise ValueError("Failed to generate profile: No content in response")
        profile_data = extract_json_from_response(response)
        if not profile_data:
            logger.error(f"Failed to extract profile data from response: {response}")
            raise ValueError("Failed to extract profile data from response")
        logger.info(f"Profile data extracted: {profile_data}. Stopping Ollama.")
        stop_ollama_container()
        return cls(
            name=profile_data.get("name", "Default Name"),
            background=profile_data.get("background"),
            personality=profile_data.get("personality"),
            interests=profile_data.get("interests"),
            physical_characteristics=profile_data.get("physical_characteristics")
        )

    def generate_scene_description(self):
        """Generate a scene description based on the profile's background and physical characteristics."""
        if not self.physical_characteristics:
            raise ValueError("Cannot generate scene description: physical_characteristics is empty.")
        llm = InferenceLLMConfig(
            model_name=settings.INFERENCE_DEPLOYMENT_NAME,
            base_url=settings.INFERENCE_BASE_URL,
            api_key=settings.INFERENCE_API_KEY,
        )
        logger.info(f"Generating scene description using Ollama LLM: {llm.model_name}")
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
                    "Start the prompt with the character's physical characteristics. Infer a generic location "
                    "where the scene takes place and what the character is doing in the scene based on one of "
                    "their interests. Describe their clothing based on where they are and what they're doing. "
                    "List their facial expression and posture. List elements of the scene background noting "
                    "lighting and details of any other relevant objects in the scene. Include only the visual "
                    "elements that would be captured in a photograph. Remove any unnecessary words like articles "
                    "and conjunctions. "
                    "Write the response as a single string of 75 words or less separated by commas and periods.",
                },
                {
                    "role": "user",
                    "content": f"Generate an erotic scene description for {self.name} with the following physical "
                    f"characteristics: {self.physical_characteristics}. Use the following character background: "
                    f"{self.background}. ",
                },
            ]
        )
        #Strip out anything between <think>...</think> tags
        response = remove_thinking(response)
        if not response:
            raise ValueError("Failed to generate scene description: No content in response")
        logger.info(f"Generated scene description: {response}.\nStopping Ollama.")
        stop_ollama_container()
        return response

    def generate_images(self):
        """Generate a set of images based on the profile's background and physical characteristics."""
        if not self.physical_characteristics:
            raise ValueError("Cannot generate image: physical_characteristics is empty.")
        scene_description = self.generate_scene_description()
        if not scene_description:
            raise ValueError("Cannot generate image: scene description is empty.")
        logger.info("Rendering images from scene description.")
        filenames = image_from_prompt(scene_description, preset="seed_search")
        logger.info(f"Image(s) generated and saved to {filenames}")
        self.profile_image_path = json.dumps(filenames) if filenames else None
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