import os
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from pydantic import BaseModel
from base import Base
import json
from ml.llm import InferenceLLMConfig, extract_json_from_response, remove_thinking
from utils import settings, logger

Base = declarative_base()


class ModelUsage(Base):
    __tablename__ = "model_usage"
    id = Column(Integer, primary_key=True)
    llm_model = Column(String, nullable=False)
    image_model = Column(String, nullable=True)
    status = Column(String, nullable=False, default="idle")

    def model_dump(self):
        return {
            "id": self.id,
            "llm_model": self.llm_model,
            "image_model": self.image_model,
            "status": self.status
        }

class ModelUsageSchema(BaseModel):
    id: int | None = None
    llm_model: str
    image_model: str | None = None
    status: str

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
    profile_image_description = Column(String)
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
            "profile_image_description": self.profile_image_description,
            "profile_image_path": self.profile_image_path,
            "chat_model": self.chat_model
        }

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


class ProfileSchema(BaseModel):
    id: int | None = None
    name: str
    background: str | None = None
    personality: str | None = None
    interests: str | None = None
    physical_characteristics: str | None = None
    image_model: str | None = None
    image_seed: str | None = None
    profile_image_description: str | None = None
    profile_image_path: str | None = None
    chat_model: str | None = None

    class Config:
        from_attributes = True


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

    def get_scene_summaries_as_array(self) -> list:
        """Get the plot points as an array."""
        if self.scene_summaries:
            # Replace curly brackets with square brackets for JSON compatibility
            if self.scene_summaries.startswith("{") and self.scene_summaries.endswith("}"):
                self.scene_summaries = self.scene_summaries.replace("{", "[").replace("}", "]")
            # Parse the JSON string into a Python list
            try:
                return json.loads(self.scene_summaries)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode scene summaries: {self.scene_summaries}")
                return []
        return []

    def get_scene_descriptions(self):
        """Get scene descriptions as json"""
        if self.scene_descriptions:
            # Replace curly brackets with square brackets for JSON compatibility
            if self.scene_descriptions.startswith("{") and self.scene_descriptions.endswith("}"):
                self.scene_descriptions = self.scene_descriptions.replace("{", "[").replace("}", "]")
            try:
                scene_descriptions = json.loads(self.scene_descriptions)
                return scene_descriptions
            except json.JSONDecodeError:
                logger.error("Invalid JSON in scene_descriptions: {self.scene_descriptions}")
                return []
        return []

    def delete_images(self):
        """Delete the scenario images from the filesystem."""
        if not self.images:
            logger.warning("No images to delete.")
            return
        image_paths = json.loads(self.images)
        for path in image_paths:
            try:
                os.remove(path)
                logger.info(f"Deleted image: {path}")
            except OSError as e:
                logger.error(f"Error deleting image {path}: {e}")
        self.images = None
        logger.info("All scenario images deleted and images field cleared.")


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