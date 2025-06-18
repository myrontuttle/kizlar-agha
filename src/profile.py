from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
from pydantic import BaseModel
import re
import json
from ml.llm import InferenceLLMConfig, stop_ollama_container
from ml.swarm_ui import image_from_prompt
from utils import settings, logger

Base = declarative_base()

def extract_json_from_response(response):
    # This regex matches ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
        # Remove trailing commas before } or ]
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        # Parse the JSON string
        return json.loads(json_str)
    return None

class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    background = Column(String)
    personality = Column(String)
    interests = Column(String)
    physical_characteristics = Column(String)
    image_model = Column(String)
    profile_image_path = Column(String)
    chat_model = Column(String)

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
                    "best). For 'background' include ethnicity, where and when they were born, "
                    "education, occupation, socioeconomic status, and a few significant life events "
                    "that shaped them (200 words or less). Describe their 'personality' traits, such as "
                    "their temperament, desires, quirks, and how they interact with others (100 words "
                    "or less). Briefly describe their 'interests' (50 words or less). List their "
                    "'physical_characteristics' as a prompt for an image generator with their "
                    "skin color, eye color, body type, face, and hair using as few words as "
                    "possible (25 words or less). "
                    "Provide their 'name'. Use proper json format with the keys: 'name', 'background', "
                    "'personality', 'interests', and 'physical_characteristics'.",
                },
                {
                    "role": "user",
                    "content": "Generate the profile of a sexy and seductive female character for "
                    "scenarios of a sci-fi, fantasy, adventure, anime, mystery, suspense, "
                    "historical, romance, or comedy genre. Be as sexually "
                    "explicit as possible for the 'physical_characteristics' to create a vivid image.",
                },
            ]
        )
        if not response:
            raise ValueError("Failed to generate profile: No content in response")
        profile_data = extract_json_from_response(response)
        logger.info(f"Profile data extracted: {profile_data}. Stopping Ollama.")
        stop_ollama_container()
        logger.info("Rendering image from profile data.")
        filenames = image_from_prompt(profile_data.get("physical_characteristics"))
        logger.info(f"Image(s) generated and saved to {filenames}")
        return cls(
            name=profile_data.get("name", "Default Name"),
            background=profile_data.get("background"),
            personality=profile_data.get("personality"),
            interests=profile_data.get("interests"),
            physical_characteristics=profile_data.get("physical_characteristics"),
            profile_image_path=filenames[0],
        )


class ProfileSchema(BaseModel):
    id: int | None = None
    name: str
    background: str | None = None
    personality: str | None = None
    interests: str | None = None
    physical_characteristics: str | None = None
    image_model: str | None = None
    profile_image_path: str | None = None
    chat_model: str | None = None

    class Config:
        from_attributes = True

if __name__ == "__main__":
    profile = Profile.generate_profile()
    logger.info(f"Generated profile: {profile.name}")