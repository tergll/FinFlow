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
from voice import get_audio_bytes

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

class SearchResult(BaseModel):
    """Results from a search operation"""
    step_number: int = Field(..., description="The step number this search is for")
    step_name: str = Field(..., description="The name of the step")
    search_query: str = Field(..., description="The query used for searching")
    results: str = Field(..., description="The search results")
    timestamp: str = Field(..., description="When the search was conducted")

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

#########################
# Research Execution Agent
#########################

class ResearchExecutionParams(BaseModel):
    """Parameters for executing research"""
    plan: ResearchPlan = Field(..., description="The research plan to execute")
    selected_sources: Dict[str, bool] = Field(..., description="Selected sources to use")

class ResearchExecutionResult(BaseModel):
    """Results from executing research"""
    search_results: List[SearchResult] = Field(..., description="List of search results for each step")
    summary: str = Field(..., description="Summary of the research findings")

research_execution_agent = Agent(
    model=gemini_model,
    deps_type=ResearchExecutionParams,
    result_type=ResearchExecutionResult,
    system_prompt="""You are an expert research executor who follows research plans to gather information.

Your task is to:

1. Analyze the provided research plan
2. For each step in the plan, generate appropriate search queries based on the step description
3. Execute searches using the specified sources
4. Compile and organize the search results
5. Provide a summary of the overall findings

Focus on being thorough and accurate while following the research plan closely.
"""
)

#########################
# Research Summary Agent
#########################

class ResearchSummaryParams(BaseModel):
    """Parameters for generating a research summary"""
    research_plan: ResearchPlan = Field(..., description="The research plan that was executed")
    search_results: List[SearchResult] = Field(..., description="The search results obtained during research")

class ResearchSummaryResult(BaseModel):
    """Result of the research summary generation"""
    summary: str = Field(..., description="A comprehensive summary of the research findings")

research_summary_agent = Agent(
    model=gemini_model,
    deps_type=ResearchSummaryParams,
    result_type=ResearchSummaryResult,
    system_prompt="""You are an expert research analyst who creates concise yet comprehensive summaries of research findings.

Your task is to:

1. Analyze the provided research plan and search results
2. Evaluate how well the findings address the research objectives and key topics
3. Identify any gaps that may need further investigation
4. Create a well-structured summary that highlights the most important insights

Focus on being objective and analytical while providing a clear overview of what was discovered.
"""
)

# Streamlit UI Functions

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "research_plan" not in st.session_state:
        st.session_state["research_plan"] = None
    if "research_results" not in st.session_state:
        st.session_state["research_results"] = None
    if "trigger_research" not in st.session_state:
        st.session_state["trigger_research"] = False
    if "selected_sources" not in st.session_state:
        st.session_state["selected_sources"] = {
            "Web Search": True,
            "YouTube": True,
            "Google News": True,
            "Yahoo Finance": True,
            "Documents on Hand": True
        }

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
    
    sources = {}
    for source, default in st.session_state["selected_sources"].items():
        sources[source] = st.checkbox(source, value=default)
    
    # Update session state with selected sources
    st.session_state["selected_sources"] = sources
    
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

async def execute_research(research_plan, selected_sources):
    """Execute research based on the plan and selected sources."""
    search_results = []
    for step in research_plan.research_steps:
        # Only process steps that use sources that are selected
        step_sources = [s.lower() for s in step.sources]
        
        should_search = False
        search_type = ""
        
        # Determine if this step should use search and what type
        if any(s in ["google", "web search", "search"] for s in step_sources) and selected_sources.get("Web Search", False):
            should_search = True
            search_type = "Web Search"
        elif "youtube" in step_sources and selected_sources.get("YouTube", False):
            should_search = True
            search_type = "YouTube"
        elif any(s in ["news", "google news"] for s in step_sources) and selected_sources.get("Google News", False):
            should_search = True
            search_type = "Google News"
        elif any(s in ["finance", "yahoo finance"] for s in step_sources) and selected_sources.get("Yahoo Finance", False):
            should_search = True
            search_type = "Yahoo Finance"
        
        if should_search:
            # Generate a search query based on the step
            search_query = f"{search_type}: {step.step_name} - {step.description}"
            
            # Execute the search
            results = await execute_google_search(search_query)
            
            # Store the results
            search_results.append(SearchResult(
                step_number=step.step_number,
                step_name=step.step_name,
                search_query=search_query,
                results=results,
                timestamp=datetime.now().isoformat()
            ))
    
    # Generate a summary of all findings
    summary = await generate_research_summary(research_plan, search_results)
    
    return ResearchExecutionResult(
        search_results=search_results,
        summary=summary
    )

async def generate_research_summary(research_plan, search_results):
    """Generate a summary of the research findings using the research summary agent."""
    try:
        # Create a summary of the research findings using the agent
        summary_params = ResearchSummaryParams(
            research_plan=research_plan,
            search_results=search_results
        )
        
        result = await research_summary_agent.run(summary_params)
        
        return result.summary
    except Exception as e:
        logger.error(f"Error generating research summary: {e}")
        return "Could not generate a summary of the research findings."

def display_audio_player(text: str):
    """Display an audio player for the given text"""
    audio_bytes = get_audio_bytes(text)
    if audio_bytes:
        st.audio(audio_bytes, format='audio/mp3')
    else:
        st.error("Failed to generate audio")

def display_research_plan_and_results(research_plan, research_results):
    """Display the research plan and results in a structured format."""
    if not research_plan:
        st.error("Failed to generate a research plan.")
        return
    
    # Display research plan
    st.header("Research Analysis & Plan")
    
    # User Intent
    st.subheader("📋 User Intent")
    st.write(research_plan.user_intent)
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("🔊", key="intent_audio"):
            display_audio_player(research_plan.user_intent)
    
    # Key Topics
    st.subheader("🔑 Key Topics to Research")
    for i, topic in enumerate(research_plan.key_topics, 1):
        st.write(f"{i}. {topic}")
    
    # Research Steps with Results
    st.subheader("🔍 Research Steps & Findings")
    
    if research_results and research_results.search_results:
        # Create a dictionary to easily access results by step number
        results_by_step = {r.step_number: r for r in research_results.search_results}
        
        for step in research_plan.research_steps:
            with st.expander(f"Step {step.step_number}: {step.step_name}", expanded=True):
                col1, col2 = st.columns([10, 1])
                with col1:
                    st.write("**Description:**")
                    st.write(step.description)
                with col2:
                    if st.button("🔊", key=f"step_{step.step_number}_audio"):
                        display_audio_player(step.description)
                
                st.write("**Sources Used:**")
                for source in step.sources:
                    st.write(f"- {source}")
                
                st.write("**Expected Outcome:**")
                st.write(step.expected_outcome)
                
                # Display search results if available for this step
                if step.step_number in results_by_step:
                    result = results_by_step[step.step_number]
                    st.write("**🔎 Research Findings:**")
                    col1, col2 = st.columns([10, 1])
                    with col1:
                        st.markdown(result.results)
                    with col2:
                        if st.button("🔊", key=f"results_{step.step_number}_audio"):
                            display_audio_player(result.results)
                else:
                    st.info("No research findings available for this step.")
        
        # Display summary with audio button
        st.subheader("📊 Research Summary")
        col1, col2 = st.columns([10, 1])
        with col1:
            st.write(research_results.summary)
        with col2:
            if st.button("🔊", key="summary_audio"):
                display_audio_player(research_results.summary)
    else:
        # If no results yet, show research plan only
        for step in research_plan.research_steps:
            with st.expander(f"Step {step.step_number}: {step.step_name}", expanded=True):
                col1, col2 = st.columns([10, 1])
                with col1:
                    st.write("**Description:**")
                    st.write(step.description)
                with col2:
                    if st.button("🔊", key=f"step_{step.step_number}_audio"):
                        display_audio_player(step.description)
                
                st.write("**Sources to Use:**")
                for source in step.sources:
                    st.write(f"- {source}")
                
                st.write("**Expected Outcome:**")
                st.write(step.expected_outcome)
        
        st.info("Research is currently in progress. Results will appear here once complete.")
    
    # Final Deliverable with audio button
    st.subheader("📑 Final Deliverable")
    col1, col2 = st.columns([10, 1])
    with col1:
        st.write(research_plan.final_deliverable)
    with col2:
        if st.button("🔊", key="deliverable_audio"):
            display_audio_player(research_plan.final_deliverable)

async def main():
    """Main function for the Streamlit app."""
    with st.sidebar:
        st.title("📚 Research Assistant")
        st.markdown("*Helping analysts plan and conduct thorough research*")
        
        # Initialize session state
        initialize_session_state()
        
        # Setup research sources
        st.subheader("Research Sources")
        selected_sources = setup_sidebar()
        
        # Display chat interface
        st.subheader("Chat Interface")
        display_chat_messages()
        
        # Input field for new queries
        prompt = handle_user_input()
    
    # Main content area
    if st.session_state["trigger_research"]:
        st.session_state["trigger_research"] = False
        
        # Process the research query
        with st.status("Processing your research query...", expanded=True) as status:
            # Step 1: Generate research plan
            status.update(label="Generating research plan...")
            query = st.session_state.messages[-1]["content"]
            research_plan = await generate_research_plan(query)
            st.session_state["research_plan"] = research_plan
            
            # Step 2: Execute research automatically
            if research_plan:
                status.update(label="Executing research automatically...")
                research_results = await execute_research(research_plan, st.session_state["selected_sources"])
                st.session_state["research_results"] = research_results
                
                # Add response to chat history
                response = "I've analyzed your query, created a research plan, and automatically conducted the research. You can see the complete findings below."
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                with st.sidebar:
                    with st.chat_message("assistant"):
                        st.markdown(response)
                
                status.update(label="Research completed!", state="complete")
            else:
                status.update(label="Failed to generate research plan", state="error")
    
    # Display the research plan and results
    if st.session_state.get("research_plan"):
        display_research_plan_and_results(
            st.session_state["research_plan"],
            st.session_state.get("research_results")
        )

if __name__ == "__main__":
    safe_async_run(main())