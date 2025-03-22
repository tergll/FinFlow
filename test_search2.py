import streamlit as st
import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
import nest_asyncio
from google import genai
from google.genai import types

# Apply nest_asyncio to handle async operations in Streamlit
nest_asyncio.apply()

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API credentials
GEMINI_API_KEY = st.secrets["GOOGLE_AI_STUDIO"]
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY  # Set for genai library

# Set up Gemini model
gemini_model = GeminiModel('gemini-2.0-flash', api_key=GEMINI_API_KEY)

# Helper function to safely run async code
def safe_async_run(coroutine):
    """Safely runs async coroutines, handling event loop management."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)
    except RuntimeError as e:
        if "There is no current event loop in thread" in str(e):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coroutine)
        else:
            raise

#########################
# Google Search Integration
#########################

async def execute_google_search(query):
    """Execute a Google search using Gemini's Google Search tool."""
    try:
        # Initialize the Gemini client
        client = genai.Client(api_key=GEMINI_API_KEY)
        model = "gemini-2.0-flash"
        
        # Set up the query with the search tool
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=query)],
            ),
        ]
        
        tools = [types.Tool(google_search=types.GoogleSearch())]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            tools=tools,
            response_mime_type="text/plain",
        )
        
        # Execute the search and collect the results
        result_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            result_text += chunk.text
        
        return result_text
    except Exception as e:
        logger.error(f"Error executing Google search: {e}")
        return f"Error executing search: {str(e)}"

#########################
# Research Planner Models
#########################

class ResearchStep(BaseModel):
    """A single step in the research plan"""
    step_number: int = Field(..., description="The sequence number of this step")
    step_name: str = Field(..., description="A concise name for this step")
    description: str = Field(..., description="Detailed description of what to do in this step")
    sources: List[str] = Field(..., description="Information sources to use (Google, YouTube, documents, etc.)")
    expected_outcome: str = Field(..., description="What information this step should provide")

class ResearchPlan(BaseModel):
    """A comprehensive research plan"""
    user_intent: str = Field(..., description="Analysis of the user's intent based on their query")
    key_topics: List[str] = Field(..., description="List of key topics that need to be researched")
    research_steps: List[ResearchStep] = Field(..., description="Step-by-step plan for gathering information")
    final_deliverable: str = Field(..., description="Description of what the final answer should include")

class ResearchPlannerParams(BaseModel):
    """Parameters for the research planner"""
    query: str = Field(..., description="The user's research query")

#########################
# Research Planner Agent
#########################

research_planner_agent = Agent(
    model=gemini_model,
    deps_type=ResearchPlannerParams,
    result_type=ResearchPlan,
    system_prompt="""You are an expert research planner who helps analysts develop structured plans to answer research questions.

When given a query, your task is to:

1. Identify the user's primary intent and the type of information they're looking for
2. Extract the key topics that need to be investigated
3. Create a detailed, step-by-step research plan that a junior analyst could follow
4. Specify exactly what sources should be consulted for each step (YouTube, Google News, Yahoo Finance, specific documents, etc.)

Your research plan should be thorough yet practical, helping the analyst efficiently gather all information needed to provide a comprehensive answer.

Present your analysis in a structured format including:
- User intent analysis
- Key topics to research
- Numbered research steps with clear descriptions
- Expected outcome for each step
- Description of the final deliverable

Consider what a new analyst would need to know to conduct this research effectively.
"""
)

# Streamlit UI Functions

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "research_plan" not in st.session_state:
        st.session_state["research_plan"] = None
    if "trigger_research" not in st.session_state:
        st.session_state["trigger_research"] = False
    
    # Initialize search-related state variables
    if "search_results" not in st.session_state:
        st.session_state["search_results"] = {}

def display_chat_messages():
    """Display chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    """Handle user input from chat."""
    if prompt := st.chat_input("What would you like to research?"):
        logger.info(f"User input received: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["trigger_research"] = True
        return prompt
    return None

def setup_sidebar():
    """Setup the sidebar with research sources and options."""
    st.write("Available sources for gathering information:")
    
    sources = {
        "Web Search": True,
        "YouTube": True,
        "Google News": True,
        "Yahoo Finance": True,
        "Documents on Hand": True
    }
    
    # Create toggles for each source
    for source, default in sources.items():
        sources[source] = st.checkbox(source, value=default)
    
    # Source descriptions
    with st.expander("Source Information"):
        st.write("**Web Search**: General information from the web")
        st.write("**YouTube**: Video content and tutorials")
        st.write("**Google News**: Recent news articles")
        st.write("**Yahoo Finance**: Financial data and business news")
        st.write("**Documents on Hand**: Files and documents uploaded by the user")
    
    return sources

async def generate_research_plan(query):
    """Generate a research plan using the Gemini agent."""
    try:
        params = ResearchPlannerParams(query=query)
        response = await research_planner_agent.run(
            user_prompt=f"Create a detailed research plan for the following query: {query}",
            deps=params
        )
        return response.data
    except Exception as e:
        logger.error(f"Error generating research plan: {e}")
        return None

def display_research_plan(research_plan):
    """Display the research plan in a structured format with search buttons."""
    if not research_plan:
        st.error("Failed to generate a research plan.")
        return
    
    st.header("Research Analysis & Plan")
    
    # User Intent
    st.subheader("📋 User Intent")
    st.write(research_plan.user_intent)
    
    # Key Topics
    st.subheader("🔑 Key Topics to Research")
    for i, topic in enumerate(research_plan.key_topics, 1):
        st.write(f"{i}. {topic}")
    
    # Research Steps
    st.subheader("🔍 Step-by-Step Research Plan")
    
    for step in research_plan.research_steps:
        with st.expander(f"Step {step.step_number}: {step.step_name}", expanded=True):
            st.write("**Description:**")
            st.write(step.description)
            
            st.write("**Sources to Use:**")
            for source in step.sources:
                st.write(f"- {source}")
                
            st.write("**Expected Outcome:**")
            st.write(step.expected_outcome)
            
            # Add search buttons for this step
            if any(s.lower() in [src.lower() for src in step.sources] for s in ["google", "web search", "search"]):
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_query = st.text_input(
                        "Search query:",
                        value=f"{step.step_name}: {step.description}",
                        key=f"query_{step.step_number}"
                    )
                with col2:
                    if st.button("Search", key=f"search_btn_{step.step_number}"):
                        with st.spinner("Searching..."):
                            results = safe_async_run(execute_google_search(search_query))
                            st.session_state[f"search_results_{step.step_number}"] = results
    
    # Display search results section
    st.subheader("🔎 Research Results")
    
    has_results = False
    for step in research_plan.research_steps:
        result_key = f"search_results_{step.step_number}"
        if result_key in st.session_state and st.session_state[result_key]:
            has_results = True
            with st.expander(f"Search Results for Step {step.step_number}: {step.step_name}", expanded=True):
                st.markdown(st.session_state[result_key])
    
    if not has_results:
        st.info("No research has been conducted yet. Click the search buttons in the research plan steps to gather information.")
    
    # Final Deliverable
    st.subheader("📊 Final Deliverable")
    st.write(research_plan.final_deliverable)

async def main():
    """Main function for the Streamlit app."""
    with st.sidebar:
        st.title("📚 Research Assistant")
        st.markdown("*Helping analysts plan and conduct thorough research*")
        
        # Initialize session state
        initialize_session_state()
        
        # Setup research sources
        st.subheader("Research Sources")
        sources = setup_sidebar()
        
        # Display chat interface
        st.subheader("Chat Interface")
        display_chat_messages()
        prompt = handle_user_input()
    
    # Process research if triggered
    if st.session_state["trigger_research"]:
        st.session_state["trigger_research"] = False
        
        with st.status("Analyzing query and creating research plan...") as status:
            query = st.session_state.messages[-1]["content"]
            
            # Generate research plan
            research_plan = await generate_research_plan(query)
            st.session_state["research_plan"] = research_plan
            
            status.update(label="Research plan generated!", state="complete")
            
            # Add response to chat history
            response = "I've analyzed your research query and created a detailed plan to help you gather the information you need."
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.sidebar:
                with st.chat_message("assistant"):
                    st.markdown(response)
    
    # Display the research plan
    if st.session_state.get("research_plan"):
        display_research_plan(st.session_state["research_plan"])

if __name__ == "__main__":
    safe_async_run(main())