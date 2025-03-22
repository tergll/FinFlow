import streamlit as st
import asyncio
import logging
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
import nest_asyncio

# Apply nest_asyncio to handle async operations in Streamlit
nest_asyncio.apply()

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API credentials
GEMINI_API_KEY = st.secrets["GOOGLE_AI_STUDIO"]
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", "")

# Set up Gemini model
gemini_model = GeminiModel('gemini-2.0-flash', api_key=GEMINI_API_KEY)

# Helper function to safely run async code
def safe_async_run(coroutine):
    """Safely runs async coroutines, handling event loop management."""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        
        # Check if the loop is closed
        if loop.is_closed():
            # Create a new event loop if closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the coroutine
        return loop.run_until_complete(coroutine)
    except RuntimeError as e:
        # Handle various runtime errors
        if "There is no current event loop in thread" in str(e):
            # Create a new event loop if there is none
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coroutine)
        else:
            # Re-raise other RuntimeErrors
            raise

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

class WebSearchParameters(BaseModel):
    """Input parameters for web search"""
    search_query: str = Field(min_length=3, description="The search query")
    max_result_count: int = Field(default=3, ge=1, le=10, description="Maximum number of results to return")
    search_date: str = Field(description="Date when search is performed")
    include_images: bool = Field(default=False, description="Whether to include image results")
    search_depth: str = Field(default="advanced", description="Search depth (basic/advanced)")

class WebSearchResultItem(BaseModel):
    """Individual web search result with metadata"""
    result_title: str = Field(description="Title of the search result")
    result_content: str = Field(description="Main content or summary of the result")
    result_url: str = Field(description="URL of the source")
    result_type: str = Field(description="Type of the source (e.g., Website, News, Academic)")
    result_score: float = Field(ge=0.0, le=1.0, description="Relevance score of the result (0.0 to 1.0)")
    result_date: Optional[str] = Field(None, description="Publication or last updated date of the result")
    query_timestamp: Optional[str] = Field(default=None, description="Query Timestamp")
    search_query: Optional[str] = Field(default=None, description="Search query used to find this result")

class WebSearchResponse(BaseModel):
    """Complete web search response including analysis"""
    search_summary: str = Field(min_length=50, description="AI-generated summary of all search results")
    search_findings: List[str] = Field(min_items=1, description="List of key findings from the search results")
    search_results: List[WebSearchResultItem] = Field(min_items=1, description="List of relevant search results")
    follow_up_queries: List[str] = Field(min_items=1, description="Suggested follow-up queries for more information")
    search_timestamp: str = Field(description="Timestamp when the search was performed")

# Streamlit UI Functions

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "research_plan" not in st.session_state:
        st.session_state["research_plan"] = None
    if "trigger_research" not in st.session_state:
        st.session_state["trigger_research"] = False

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
    st.sidebar.title("Research Sources")
    
    st.sidebar.write("Available sources for gathering information:")
    
    sources = {
        "Web Search": True,
        "YouTube": True,
        "Google News": True,
        "Yahoo Finance": True,
        "Documents on Hand": True
    }
    
    # Create toggles for each source
    for source, default in sources.items():
        sources[source] = st.sidebar.checkbox(source, value=default)
    
    # Source descriptions
    with st.sidebar.expander("Source Information"):
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
    """Display the research plan in a structured format."""
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
    
    # Final Deliverable
    st.subheader("📊 Final Deliverable")
    st.write(research_plan.final_deliverable)

async def main():
    """Main function for the Streamlit app."""
    with st.sidebar:
        
        st.title("📚 Research Assistant")
        st.markdown("*Helping analysts plan and conduct thorough research*")
        
        # Initialize session state and sidebar
        initialize_session_state()
        sources = setup_sidebar()
        
        # Display existing chat messages
        display_chat_messages()
        
        # Handle new user input
        prompt = handle_user_input()
        
        # Process research if triggered
        if st.session_state["trigger_research"]:
            st.session_state["trigger_research"] = False
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Analyzing your query and creating a research plan...")
                
                query = st.session_state.messages[-1]["content"]
                
                # Generate research plan
                research_plan = await generate_research_plan(query)
                st.session_state["research_plan"] = research_plan
                
                # Display response
                response = "I've analyzed your research query and created a detailed plan to help you gather the information you need. See the plan below."
                message_placeholder.markdown(response)
                
                # Add response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
    # Display the research plan below the chat
    if st.session_state["research_plan"]:
        display_research_plan(st.session_state["research_plan"])

if __name__ == "__main__":
    safe_async_run(main())