import streamlit as st

from utils import docker_client

st.write("# Home Page")

st.write(
    """This application template showcases the versatility of Streamlit,
    allowing you to choose between using Streamlit."""
)

containers = docker_client.containers.list(all=True)

# Display each container's name and status
for container in containers:
    st.write(f"**{container.name}** — Status: `{container.status}`")
