import ast
import os
import sys
import timeit
from pathlib import Path

from loguru import logger as loguru_logger
from pydantic import ValidationError

import docker

from settings_env import Settings

# Check if we run the code from the src directory
if Path("src").is_dir():
    loguru_logger.warning("Changing working directory to src")
    loguru_logger.warning(f" Current working dir is {Path.cwd()}")
    os.chdir("src")
elif Path("ml").is_dir():
    # loguru_logger.warning(f" Current working dir is {Path.cwd()}")
    pass
else:
    raise Exception(
        f"Project should always run from the src directory. But current working dir is {Path.cwd()}"
    )


def initialize():
    """Initialize the settings, logger and docker client.

    Reads the environment variables from the .env file defined in the Settings class.

    Returns:
        settings
        loguru_logger
    """
    settings = Settings()
    loguru_logger.remove()

    if settings.DEV_MODE:
        loguru_logger.add(sys.stderr, level="TRACE")
    else:
        loguru_logger.add(sys.stderr, level="INFO")

    docker_client = docker.DockerClient(base_url="unix://var/run/docker.sock")

    return settings, loguru_logger, docker_client


def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)

        end_time = timeit.default_timer()
        execution_time = round(end_time - start_time, 2)
        if "reason" in result:
            result["reason"] = f" Execution time: {execution_time}s | " + result["reason"]

        if "output" in result:
            result["output"] = f" Execution time: {execution_time}s | " + result["output"]
        logger.debug(f"Function {func.__name__} took {execution_time} seconds to execute.")

        return result

    return wrapper


def validation_error_message(error: ValidationError) -> ValidationError:
    for err in error.errors():
        del err["input"]
        del err["url"]

    return error


settings, logger, docker_client = initialize()
