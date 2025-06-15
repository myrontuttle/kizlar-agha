from typing import Optional, Self

from loguru import logger as loguru_logger
from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.pretty import pretty_repr


class BaseEnvironmentVariables(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")


class InferenceEnvironmentVariables(BaseEnvironmentVariables):
    INFERENCE_BASE_URL: Optional[str] = "http://localhost:11434"
    INFERENCE_API_KEY: Optional[SecretStr] = "tt"
    INFERENCE_DEPLOYMENT_NAME: Optional[str] = "ollama_chat/qwen2.5:0.5b"

    def get_inference_env_vars(self):
        return {
            "INFERENCE_BASE_URL": self.INFERENCE_BASE_URL,
            "INFERENCE_API_KEY": self.INFERENCE_API_KEY,
            "INFERENCE_DEPLOYMENT_NAME": self.INFERENCE_DEPLOYMENT_NAME,
        }


class EmbeddingsEnvironmentVariables(BaseEnvironmentVariables):
    EMBEDDINGS_BASE_URL: Optional[str] = None
    EMBEDDINGS_API_KEY: Optional[SecretStr] = "tt"
    EMBEDDINGS_DEPLOYMENT_NAME: Optional[str] = None

    def get_embeddings_env_vars(self):
        return {
            "EMBEDDINGS_BASE_URL": self.EMBEDDINGS_BASE_URL,
            "EMBEDDINGS_API_KEY": self.EMBEDDINGS_API_KEY,
            "EMBEDDINGS_DEPLOYMENT_NAME": self.EMBEDDINGS_DEPLOYMENT_NAME,
        }


class EvaluatorEnvironmentVariables(BaseEnvironmentVariables):
    EVALUATOR_BASE_URL: Optional[str] = "http://localhost:11434"
    EVALUATOR_API_KEY: Optional[SecretStr] = "tt"
    EVALUATOR_DEPLOYMENT_NAME: Optional[str] = "ollama_chat/qwen2.5:0.5b"

    ENABLE_EVALUATION: bool = False

    def get_evaluator_env_vars(self):
        return {
            "EVALUATOR_BASE_URL": self.EVALUATOR_BASE_URL,
            "EVALUATOR_API_KEY": self.EVALUATOR_API_KEY,
            "EVALUATOR_DEPLOYMENT_NAME": self.EVALUATOR_DEPLOYMENT_NAME,
        }

    @model_validator(mode="after")
    def check_eval_api_keys(self: Self) -> Self:
        """Validate API keys based on the selected provider after model initialization."""
        if self.ENABLE_EVALUATION:
            eval_vars = self.get_evaluator_env_vars()
            if any(value is None for value in eval_vars.values()):
                # loguru_logger.opt(exception=True).error("Your error message")
                loguru_logger.error(
                    "\nEVALUATION environment variables must be provided when ENABLE_EVALUATION is True."
                    f"\n{pretty_repr(eval_vars)}"
                )
                raise ValueError(
                    "\nEVALUATION environment variables must be provided when ENABLE_EVALUATION is True."
                    f"\n{pretty_repr(eval_vars)}"
                )

        return self


class Settings(
    InferenceEnvironmentVariables,
    EmbeddingsEnvironmentVariables,
    EvaluatorEnvironmentVariables,
):
    """Settings class for the application.

    This class is automatically initialized with environment variables from the .env file.
    It inherits from the following classes and contains additional settings for streamlit
    - ChatEnvironmentVariables
    - EvaluationEnvironmentVariables

    """

    STREAMLIT_PORT: int = 8501
    DEV_MODE: bool = True

    def get_active_env_vars(self):
        env_vars = {
            "DEV_MODE": self.DEV_MODE,
            "STREAMLIT_PORT": self.STREAMLIT_PORT,
        }

        env_vars.update(self.get_inference_env_vars())
        env_vars.update(self.get_embeddings_env_vars())

        if self.ENABLE_EVALUATION:
            env_vars.update(self.get_evaluator_env_vars())

        return env_vars
