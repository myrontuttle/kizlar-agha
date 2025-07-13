from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from base import Base
from profile import Base, Profile, ProfileSchema
from model_selection import ModelSelection, ModelSelectionSchema

import os

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    # Base.metadata.drop_all(bind=engine)  # Only for dev!
    Base.metadata.create_all(bind=engine)
    from model_selection import Base as ModelSelectionBase
    ModelSelectionBase.metadata.create_all(bind=engine)

def get_profiles():
    with SessionLocal() as session:
        return session.query(Profile).all()

def get_profile(profile_id: int):
    with SessionLocal() as session:
        return session.query(Profile).filter_by(id=profile_id).first()

def save_profile(data):
    with SessionLocal() as session:
        if data.id:
            profile = session.query(Profile).filter_by(id=data.id).first()
            if profile:
                profile.name = data.name
                profile.background = data.background
                profile.personality = data.personality
                profile.interests = data.interests
                profile.physical_characteristics = data.physical_characteristics
                profile.image_model = data.image_model
                profile.image_seed = data.image_seed
                profile.profile_image_path = data.profile_image_path
                profile.chat_model = data.chat_model
        else:
            profile = Profile(
                name=data.name,
                background=data.background,
                personality=data.personality,
                interests=data.interests,
                physical_characteristics=data.physical_characteristics,
                image_model=data.image_model,
                image_seed=data.image_seed,
                profile_image_path=data.profile_image_path,
                chat_model=data.chat_model
            )
            session.add(profile)
        session.commit()
        # Return a copy or dict, not the ORM object
        return ProfileSchema.model_validate(profile)

def delete_profile(profile_id: int):
    with SessionLocal() as session:
        profile = session.query(Profile).filter_by(id=profile_id).first()
        if profile:
            session.delete(profile)
            session.commit()
            return True
        return False

def get_model_selection():
    with SessionLocal() as session:
        selection = session.query(ModelSelection).first()
        if selection:
            return ModelSelectionSchema.model_validate(selection)
        return None

def save_model_selection(data):
    with SessionLocal() as session:
        selection = session.query(ModelSelection).first()
        if selection:
            selection.llm_model = data.llm_model
            selection.embedding_model = data.embedding_model
            selection.image_model = data.image_model
        else:
            selection = ModelSelection(
                llm_model=data.llm_model,
                embedding_model=data.embedding_model,
                image_model=data.image_model,
            )
            session.add(selection)
        session.commit()
        return ModelSelectionSchema.model_validate(selection)