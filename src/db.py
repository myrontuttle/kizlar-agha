from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from profile import Base, Profile

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
    Base.metadata.create_all(bind=engine)

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
                profile.image_model = data.image_model
                profile.physical_characteristics = data.physical_characteristics
                profile.profile_image_path = data.profile_image_path
                profile.chat_model = data.chat_model
                profile.personality = data.personality
                profile.background = data.background
        else:
            profile = Profile(
                name=data.name,
                image_model=data.image_model,
                physical_characteristics=data.physical_characteristics,
                profile_image_path=data.profile_image_path,
                chat_model=data.chat_model,
                personality=data.personality,
                background=data.background
            )
            session.add(profile)
        session.commit()
        return profile