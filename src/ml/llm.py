import ast
from typing import Optional, Type

import instructor
import litellm
import requests
import docker
from langfuse.decorators import observe
from litellm import supports_response_schema, acompletion, completion, aembedding, embedding
from pydantic import BaseModel, SecretStr, ConfigDict, model_validator
from typing_extensions import Self


from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
)

from utils import settings, logger, docker_client

OLLAMA_CONTAINER = "ollama"

def start_ollama_container():
    """Start the Ollama container if not already running."""
    containers = docker_client.containers.list(all=True)
    if OLLAMA_CONTAINER not in [c.name for c in containers]:
        logger.info(f"{OLLAMA_CONTAINER} container not found.")
        for container in containers:
            logger.info(f"- {container.name}")
        return
    ollama_container = docker_client.containers.get(OLLAMA_CONTAINER)
    if ollama_container.status == "running":
        logger.info(f"{OLLAMA_CONTAINER} container is already running.")
        return
    else:
        try:
            docker_client.containers.get(OLLAMA_CONTAINER).start()
            logger.info("Ollama container started successfully.")
        except docker.errors.APIError as e:
            logger.error(f"Error starting {OLLAMA_CONTAINER} container: {e}")


def list_ollama_models():
    """List all models available in the Ollama container."""
    start_ollama_container()
    try:
        response = requests.get(
            f"{settings.INFERENCE_BASE_URL}/api/tags",
            headers={"Accept": "application/json"},
        )
        if response.status_code != 200:
            logger.error(f"Failed to list Ollama models: {response.status_code} - {response.text}")
            return []
        response = response.json()
        if isinstance(response, dict) and "models" in response:
            response = response["models"]
        elif isinstance(response, list):
            response = [model["name"] for model in response]
        else:
            logger.error("Unexpected response format from Ollama API.")
            return []
        logger.info(f"Available Ollama models: {response}")
        if not response:
            logger.warning("No models found in Ollama.")
            return []
        return response
    except Exception as e:
        logger.error(f"Error listing Ollama models: {e}")
        return []

def stop_ollama_container():
    """Stop the Ollama container if it is running."""
    containers = docker_client.containers.list(all=True)
    if OLLAMA_CONTAINER not in [c.name for c in containers]:
        logger.info(f"{OLLAMA_CONTAINER} container not found.")
        return
    ollama_container = docker_client.containers.get(OLLAMA_CONTAINER)
    if ollama_container.status == "running":
        try:
            ollama_container.stop()
            logger.info(f"{OLLAMA_CONTAINER} container stopped successfully.")
        except docker.errors.APIError as e:
            logger.error(f"Error stopping {OLLAMA_CONTAINER} container: {e}")
    else:
        logger.info(f"{OLLAMA_CONTAINER} container is not running.")

class InferenceLLMConfig(BaseModel):
    """Configuration for the inference model."""

    model_name: str
    base_url: str
    api_key: SecretStr
    model_config = ConfigDict(arbitrary_types_allowed=True)

    supports_response_schema: bool = False

    temperature: Optional[float] = None
    seed: int = 1729
    max_tokens: Optional[int] = None

    @model_validator(mode="after")
    def init_client(self) -> Self:
        """Initialize the LLM client."""
        start_ollama_container()
        if not self.model_name:
            models = list_ollama_models()
            self.model_name = models[0].name
        try:
            # check if the model supports structured output
            self.supports_response_schema = supports_response_schema(self.model_name.split("/")[-1])
            logger.debug(
                f"\nModel: {self.model_name} Supports response schema: {self.supports_response_schema}"
            )
        except Exception as e:
            # logger.exception(f"Error in initializing the LLM : {self}")
            logger.error(f"Error in initializing the LLM : {e}")
            raise e

        return self

    def load_model(self, prompt: str, schema: Type[BaseModel] = None, *args, **kwargs):
        pass

    @observe(as_type="generation")
    async def a_generate(self, prompt: str, schema: Type[BaseModel] = None, *args, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        return await self.a_generate_from_messages(
            messages=messages, schema=schema, *args, **kwargs
        )

    @observe(as_type="generation")
    @retry(
        wait=wait_fixed(60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (litellm.exceptions.RateLimitError, instructor.exceptions.InstructorRetryException)
        ),
    )
    async def a_generate_from_messages(
        self, messages: list, schema: Type[BaseModel] = None, *args, **kwargs
    ):
        # check if model supports structured output
        if schema:
            if self.supports_response_schema:
                res = await litellm.acompletion(
                    model=self.model_name,
                    api_key=self.api_key.get_secret_value(),
                    base_url=self.base_url,
                    messages=messages,
                    response_format=schema,
                )
                if res.choices[0].finish_reason == "content_filter":
                    raise ValueError(f"Response filtred by content filter")
                else:
                    dict_res = ast.literal_eval(res.choices[0].message.content)
                    return schema(**dict_res)

            else:
                client = instructor.from_litellm(acompletion, mode=instructor.Mode.JSON)
                res, raw_completion = await client.chat.completions.create_with_completion(
                    model=self.model_name,
                    api_key=self.api_key.get_secret_value(),
                    base_url=self.base_url,
                    messages=messages,
                    response_model=schema,
                )
                return res
        else:
            res = await litellm.acompletion(
                model=self.model_name,
                api_key=self.api_key.get_secret_value(),
                base_url=self.base_url,
                messages=messages,
            )
            return res.choices[0].message.content

    @observe(as_type="generation")
    def generate(self, prompt: str, schema: Type[BaseModel] = None, *args, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        return self.generate_from_messages(messages=messages, schema=schema, *args, **kwargs)

    @observe(as_type="generation")
    @retry(
        wait=wait_fixed(60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (litellm.exceptions.RateLimitError, instructor.exceptions.InstructorRetryException)
        ),
    )
    def generate_from_messages(
        self, messages: list, schema: Type[BaseModel] = None, *args, **kwargs
    ):
        try:
            # check if model supports structured output
            if schema:
                if self.supports_response_schema:
                    res = litellm.completion(
                        model=self.model_name,
                        api_key=self.api_key.get_secret_value(),
                        base_url=self.base_url,
                        messages=messages,
                        response_format=schema,
                    )
                    if res.choices[0].finish_reason == "content_filter":
                        raise ValueError(f"Response filtred by content filter")
                    else:
                        dict_res = ast.literal_eval(res.choices[0].message.content)
                        return schema(**dict_res)

                else:
                    client = instructor.from_litellm(completion, mode=instructor.Mode.JSON)
                    res, raw_completion = client.chat.completions.create_with_completion(
                        model=self.model_name,
                        api_key=self.api_key.get_secret_value(),
                        base_url=self.base_url,
                        messages=messages,
                        response_model=schema,
                    )
                    return res
            else:
                res = litellm.completion(
                    model=self.model_name,
                    api_key=self.api_key.get_secret_value(),
                    base_url=self.base_url,
                    messages=messages,
                )

                return res.choices[0].message.content
        except Exception as e:
            # todo handle cost if exception
            logger.error(f"Error in generating response from LLM: {e}")
            return None

    def get_model_name(self, *args, **kwargs) -> str:
        return self.model_name


class EmbeddingLLMConfig(InferenceLLMConfig):
    """Configuration for the embedding model."""

    model_name: str
    base_url: str
    api_key: SecretStr
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_model(self, prompt: str, schema: Type[BaseModel] = None, *args, **kwargs):
        pass

    def embed_text(self, text: str) -> list[float]:
        response = embedding(
            model=self.model_name,
            api_base=self.base_url,
            api_key=self.api_key.get_secret_value(),
            input=[text],
        )
        return response.data[0]["embedding"]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = embedding(
            model=self.model_name,
            api_base=self.base_url,
            api_key=self.api_key.get_secret_value(),
            input=texts,
        )
        return [data.embedding for data in response.data]

    async def a_embed_text(self, text: str) -> list[float]:
        response = await aembedding(
            model=self.model_name,
            api_base=self.base_url,
            api_key=self.api_key.get_secret_value(),
            input=[text],
        )
        return response.data[0]["embedding"]

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = await aembedding(
            model=self.model_name,
            api_base=self.base_url,
            api_key=self.api_key.get_secret_value(),
            input=texts,
        )
        return [data.embedding for data in response.data]

    def get_model_name(self):
        return self.model_name
