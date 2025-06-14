import streamlit as st

from utils import docker_client

st.write("# Home Page")

st.write(
    """This application template showcases the versatility of Streamlit,
    allowing you to choose between using Streamlit."""
)

containers = docker_client.containers.list(all=True)

st.write([container.name for container in containers])
