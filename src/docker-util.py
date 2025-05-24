import docker

dockerClient = docker.DockerClient(url='unix://var/run/docker.sock')

def list_containers():
    """
    List all containers
    """
    containers = dockerClient.containers.list(all=True)
    return [container.name for container in containers]

def start_container(container_name):
    """
    Start a container
    """
    container = dockerClient.containers.get(container_name)
    container.start()
    return f"Container {container_name} started."

def stop_container(container_name):
    """
    Stop a container
    """
    container = dockerClient.containers.get(container_name)
    container.stop()
    return f"Container {container_name} stopped."
