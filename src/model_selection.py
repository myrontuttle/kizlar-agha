from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
from pydantic import BaseModel
from base import Base

Base = declarative_base()

class ModelSelection(Base):
    __tablename__ = "model_selections"
    id = Column(Integer, primary_key=True)
    llm_model = Column(String, nullable=False)
    embedding_model = Column(String, nullable=True)
    image_model = Column(String, nullable=True)

    def model_dump(self):
        return {
            "id": self.id,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "image_model": self.image_model,
        }

class ModelSelectionSchema(BaseModel):
    id: int | None = None
    llm_model: str
    embedding_model: str | None = None
    image_model: str | None = None

    class Config:
        from_attributes = True