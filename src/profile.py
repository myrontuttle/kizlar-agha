from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
from pydantic import BaseModel

Base = declarative_base()

class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    image_model = Column(String)
    physical_characteristics = Column(String)
    profile_image_path = Column(String)
    chat_model = Column(String)
    personality = Column(String)
    background = Column(String)


class ProfileSchema(BaseModel):
    id: int | None = None
    name: str
    image_model: str | None = None
    physical_characteristics: str | None = None
    profile_image_path: str | None = None
    chat_model: str | None = None
    personality: str | None = None
    background: str | None = None

    class Config:
        from_attributes = True