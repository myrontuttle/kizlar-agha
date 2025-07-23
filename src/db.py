from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, joinedload
from base import Base
from models import Base, ModelUsage, ModelUsageSchema, Profile, ProfileSchema, Scenario, ScenarioSchema, Message, MessageSchema
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
    from models import Base as ModelUsageBase
    ModelUsageBase.metadata.create_all(bind=engine)
    from models import Base as ProfileBase
    ProfileBase.metadata.create_all(bind=engine)
    from models import Base as ScenarioBase
    ScenarioBase.metadata.create_all(bind=engine)
    from models import Base as MessageBase
    MessageBase.metadata.create_all(bind=engine)

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
                profile.profile_image_description = data.profile_image_description
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
                profile_image_description=data.profile_image_description,
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

def get_model_usage():
    with SessionLocal() as session:
        usage = session.query(ModelUsage).first()
        if usage:
            return ModelUsageSchema.model_validate(usage)
        return None

def save_model_usage(data):
    with SessionLocal() as session:
        usage = session.query(ModelUsage).first()
        if usage:
            usage.llm_model = data.llm_model
            usage.image_model = data.image_model
            usage.status = data.status
        else:
            usage = ModelUsage(
                llm_model=data.llm_model,
                image_model=data.image_model,
                status=data.status
            )
            session.add(usage)
        session.commit()
        return ModelUsageSchema.model_validate(usage)

def get_scenarios():
    with SessionLocal() as session:
        return session.query(Scenario).all()

def get_scenarios_for_profile(profile_id: int):
    with SessionLocal() as session:
        return session.query(Scenario).filter_by(profile_id=profile_id).all()

def get_scenario(scenario_id: int):
    with SessionLocal() as session:
        scenario = session.query(Scenario)\
            .options(joinedload(Scenario.profile), joinedload(Scenario.messages))\
            .get(scenario_id)
        return scenario

def save_scenario(data):
    with SessionLocal() as session:
        if data.id:
            scenario = session.query(Scenario).filter_by(id=data.id).first()
            if scenario:
                scenario.profile_id = data.profile_id
                scenario.title = data.title
                scenario.summary = data.summary
                scenario.scene_summaries = data.scene_summaries
                scenario.invitation = data.invitation
                scenario.scene_descriptions = data.scene_descriptions
                scenario.images = data.images
        else:
            scenario = Scenario(
                profile_id=data.profile_id,
                title=data.title,
                summary=data.summary,
                scene_summaries=data.scene_summaries,
                invitation=data.invitation,
                scene_descriptions=data.scene_descriptions,
                images=data.images
            )
            session.add(scenario)
        session.commit()
        return ScenarioSchema.model_validate(scenario)

def delete_scenario(scenario_id: int):
    with SessionLocal() as session:
        scenario = session.query(Scenario).filter_by(id=scenario_id).first()
        if scenario:
            session.delete(scenario)
            session.commit()
            return True
        return False

def get_messages(scenario_id):
    """Get all messages for a scenario."""
    with SessionLocal() as session:
        messages = session.query(Message).filter_by(scenario_id=scenario_id).order_by(Message.order).all()
        return [MessageSchema.model_validate(m) for m in messages]

def save_message(data):
    """Save a message for a scenario."""
    with SessionLocal() as session:
        if data.id:
            message = session.query(Message).filter_by(id=data.id).first()
            if message:
                message.scenario_id = data.scenario_id
                message.role = data.role
                message.content = data.content
                message.order = data.order
        else:
            message = Message(
                scenario_id=data.scenario_id,
                role=data.role,
                content=data.content,
                order=data.order
            )
            session.add(message)
        session.commit()
        return MessageSchema.model_validate(message)

def delete_message(message_id: int):
    """Delete a message by its ID."""
    with SessionLocal() as session:
        message = session.query(Message).filter_by(id=message_id).first()
        if message:
            session.delete(message)
            session.commit()
            return True
        return False

def get_next_message_order(scenario_id):
    with SessionLocal() as session:
        last_message = (
            session.query(Message)
            .filter_by(scenario_id=scenario_id)
            .order_by(Message.order.desc())
            .first()
        )
        return (last_message.order + 1) if last_message else 0