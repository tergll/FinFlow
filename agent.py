import streamlit as st
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel


# Initialize API credentials
GEMINI_API_KEY = st.secrets["GOOGLE_AI_STUDIO"]

# Set up Gemini model
gemini_2o_model = GeminiModel('gemini-2.0-flash', api_key=GEMINI_API_KEY)

