from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from profile import Base, Profile

import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@db:5432/postgres")

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
                profile.role = data.role
                profile.genre = data.genre
        else:
            profile = Profile(
                name=data.name,
                image_model=data.image_model,
                physical_characteristics=data.physical_characteristics,
                profile_image_path=data.profile_image_path,
                chat_model=data.chat_model,
                personality=data.personality,
                background=data.background,
                role=data.role,
                genre=data.genre
            )
            session.add(profile)
        session.commit()
        return profile