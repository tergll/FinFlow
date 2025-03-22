import streamlit as st
from typing import List, Optional, Dict, Any, Tuple
from tavily_client import AsyncTavilyClient
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
import asyncio
import os
import sys
import logging


import nest_asyncio
nest_asyncio.apply()

def safe_async_run(coroutine):
    """
    Safely runs async coroutines, handling event loop management.
    
    Args:
        coroutine: The async coroutine to run
        
    Returns:
        The result of the coroutine
    """
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


# Initialize Tavily client
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

# Logger setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# Initialize API credentials
GEMINI_API_KEY = st.secrets["GOOGLE_AI_STUDIO"]

# Set up Gemini model
gemini_2o_model = GeminiModel('gemini-2.0-flash', api_key=GEMINI_API_KEY)


class CompanyProfileData(BaseModel):
    """Detailed company profile information."""
    name: str = Field(..., description="Company name")
    founded: str = Field("", description="Year the company was founded")
    headquarters: str = Field("", description="Location of company headquarters")
    website: str = Field("", description="Company's official website URL")
    description: str = Field("", description="Comprehensive company description")
    business_model: str = Field("", description="Business model (e.g., B2B SaaS, open-source)")
    company_size: str = Field("", description="Approximate number of employees")
    total_funding: str = Field("", description="Total funding raised by the company")
    latest_funding_round: str = Field("", description="Details about the most recent funding round")
    industry_tags: List[str] = Field(default_factory=list, description="Industry categories the company belongs to")
    is_ai_focused: bool = Field(False, description="Whether the company primarily focuses on AI")
    is_startup: bool = Field(False, description="Whether the company is considered a startup")
    has_open_source_products: bool = Field(False, description="Whether the company offers open-source products")
    

company_profile_agent = Agent(
    model=gemini_2o_model,
    system_prompt="""You are a specialized agent focusing exclusively on extracting foundational company information.
    
    Your task is to thoroughly analyze the provided content about a company and extract detailed profile information including:
    
    1. Company name
    2. Year founded (specific year, not "X years ago")
    3. Headquarters location (city and country)
    4. Official website URL
    5. Comprehensive company description
    6. Business model (B2B SaaS, open-source + enterprise, etc.)
    7. Company size (employees)
    8. Total funding raised (with currency symbol)
    9. Latest funding round details
    10. Industry categories (as a list of tags)
    
    Also set boolean indicators for:
    - Whether the company is primarily AI-focused
    - Whether it's considered a startup
    - Whether it offers open-source products
    
    Be extremely thorough in searching for specific factual details like founding year and headquarters location. These are critical fields that should not be left empty if the information exists anywhere in the content. Look for phrases like "founded in", "established in", "based in", "headquartered in" throughout the entire content.
    
    If information is genuinely not available in the provided content, use empty strings for text fields or default values for boolean fields instead of "Unknown" or placeholder text.
    """,
    result_type=CompanyProfileData
)


class WebSearchResponse(BaseModel):
    """Complete web search response including analysis"""
    search_summary: str = Field(min_length=50, description="AI-generated summary of all search results")
    search_findings: List[str] = Field(min_items=1, description="List of key findings from the search results")
    search_results: List[WebSearchResultItem] = Field(min_items=1, description="List of relevant search results")
    follow_up_queries: List[str] = Field(min_items=1, description="Suggested follow-up queries for more information")
    search_timestamp: str = Field(description="Timestamp when the search was performed")


class WebSearchParameters(BaseModel):
    """Input parameters for web search"""
    search_query: str = Field(min_length=3, description="The search query")
    max_result_count: int = Field(default=3, ge=1, le=10, description="Maximum number of results to return")
    search_date: str = Field(description="Date when search is performed")
    include_images: bool = Field(default=False, description="Whether to include image results")
    search_depth: str = Field(default="advanced", description="Search depth (basic/advanced)")


# Web Search Agent Models and Types
class WebSearchResultItem(BaseModel):
    """Individual web search result with metadata"""
    result_title: str = Field(description="Title of the search result")
    result_content: str = Field(description="Main content or summary of the result")
    result_url: str = Field(description="URL of the source")
    result_type: str = Field(description="Type of the source (e.g., Website, News, Academic)")
    result_score: float = Field(ge=0.0, le=1.0, description="Relevance score of the result (0.0 to 1.0)")
    result_date: Optional[str] = Field(None, description="Publication or last updated date of the result")
    query_timestamp: Optional[str] = Field(default=None, description="Query Timestamp")
    search_query: Optional[str] = Field(default=None, description="Search query used to find this result")  # Added field

# Web Search Agent
web_search_agent = Agent(
    model=gemini_2o_model,
    deps_type=WebSearchParameters,
    result_type=WebSearchResponse,
    system_prompt=(
        "You are a web search specialist focused on accurate information retrieval and analysis.\n"
        "1. Process search results and generate a concise summary.\n"
        "2. Extract specific, actionable key findings.\n"
        "3. Evaluate and rank results by relevance.\n"
        "4. Generate targeted follow-up queries for deeper research.\n"
        "Ensure all outputs strictly follow the specified schema."
    )
)


@web_search_agent.tool
async def execute_web_search(search_context: RunContext[WebSearchParameters]) -> dict:
    """Execute web search using Tavily API with error handling."""
    start_time = datetime.now()
    try:
        search_query = search_context.deps.search_query.strip()
        if not search_query:
            raise ValueError("Search query cannot be empty")

        search_results_raw = await tavily_client.search(
            query=search_query,
            max_results=search_context.deps.max_result_count,
            search_depth=search_context.deps.search_depth,
            include_answer=True,
            include_images=search_context.deps.include_images,
            include_raw_content=True
        )

        if not search_results_raw or not search_results_raw.get('results'):
            logger.warning(f"No results found for query: {search_query}")
            return {"search_results": []}

        search_results = search_results_raw.get('results', [])
        processed_results = []
        for result in search_results:
            processed_results.append(
                WebSearchResultItem(
                    result_title=result.get("title", "Untitled"),
                    result_content=result.get("content", "No content available"),
                    result_url=result.get("url", ""),
                    result_type=result.get("type", "Website"),
                    result_score=float(result.get("score", 0.0)),
                    result_date=result.get("published_date", None),
                    query_timestamp=start_time.isoformat(),
                    search_query=search_query  # Add the search query to each result
                )
            )

        # Store raw results separately but don't return them in the WebSearchResponse
        # This avoids the Gemini API error while still preserving the data if needed
        st.session_state['last_raw_search_results'] = search_results_raw

        return {
            "search_summary": search_results_raw.get("answer", "No summary available."),
            "search_findings": [],
            "search_results": processed_results,
            "follow_up_queries": [],
            "search_timestamp": start_time.isoformat()
            # raw_search_results field removed
        }

    except Exception as e:
        error_message = f"Web search error: {str(e)}"
        logger.error(error_message)
        return {"search_results": [], "error": error_message, "search_timestamp": start_time.isoformat()}

async def process_web_search_results(search_query: str) -> Optional[WebSearchResponse]:
    """Process web search results using the agent."""
    try:
        search_params = WebSearchParameters(
            search_query=search_query,
            max_result_count=7,
            search_date=datetime.now().strftime("%Y-%m-%d"),
            include_images=False,
            search_depth="advanced"
        )
        async with sem:
            response = await web_search_agent.run(search_query, deps=search_params)
            return response.data
    except Exception as e:
        logger.error(f"Error processing web search for '{search_query}': {e}")
        return None