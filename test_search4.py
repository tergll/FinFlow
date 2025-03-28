import streamlit as st
import asyncio
import logging
import os
import sys
import hashlib
import io
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
import nest_asyncio
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import httpx
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from elevenlabs.core.api_error import ApiError

# Third-party client imports - conditionally import
try:
    from apify_client import ApifyClient
    APIFY_AVAILABLE = True
except ImportError:
    APIFY_AVAILABLE = False

# Apply nest_asyncio to handle async operations in Streamlit
nest_asyncio.apply()

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API credentials
GEMINI_API_KEY = st.secrets.get("GOOGLE_AI_STUDIO", "")
APIFY_API_KEY = st.secrets.get("APIFY_API_KEY", "")
ELEVEN_LABS_API_KEY = st.secrets.get("ELEVEN_LABS_API_KEY", "")

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

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
# ElevenLabs Integration
#########################

def check_elevenlabs_api_key() -> bool:
    """Check if the ElevenLabs API key exists in the environment variables"""
    api_key = st.secrets.get("ELEVEN_LABS_API_KEY", "")
    if not api_key:
        st.error("Error: ELEVEN_LABS_API_KEY not found in secrets")
        return False
    return True

def text_to_speech(text: str, voice_id: str = "JBFqnCBsd6RMkjVDRZzb", max_retries: int = 3) -> Optional[bytes]:
    """Convert text to speech using ElevenLabs API"""
    if not check_elevenlabs_api_key():
        return None
        
    client = ElevenLabs(
        api_key=ELEVEN_LABS_API_KEY,
    )
    
    for attempt in range(max_retries):
        try:
            audio = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
            
            # Handle different return types
            if hasattr(audio, '__iter__') and not isinstance(audio, (bytes, str)):
                try:
                    # It's a generator or iterable, collect all chunks
                    audio_bytes = b''
                    for chunk in audio:
                        if isinstance(chunk, bytes):
                            audio_bytes += chunk
                    return audio_bytes
                except Exception as e:
                    st.error(f"Error processing audio chunks: {str(e)}")
                    return None
            elif hasattr(audio, 'read'):
                # It's a file-like object
                return audio.read()
            elif isinstance(audio, bytes):
                # It's already bytes
                return audio
            else:
                st.error(f"Unexpected audio data type: {type(audio)}")
                return None
        except ApiError as e:
            st.error(f"API Error: {str(e)}")
            return None
        except httpx.RemoteProtocolError as e:
            if attempt == max_retries - 1:
                st.error(f"Failed after {max_retries} attempts: {str(e)}")
                return None
            time.sleep(2)
    
    return None
#########################
# YouTube Search Functions
#########################

def generate_unique_key(text):
    """Generate a unique key for Streamlit elements based on text content."""
    if isinstance(text, str):
        # Create a hash of the string
        hash_object = hashlib.md5(text.encode())
        return hash_object.hexdigest()[:10]
    else:
        # For non-string objects, use a timestamp
        return datetime.now().strftime("%H%M%S%f")[:10]

# Add to your imports section
try:
    from apify_client import ApifyClient
    APIFY_AVAILABLE = True
except ImportError:
    APIFY_AVAILABLE = False

# Update the search_youtube_videos function
def search_youtube_videos(query, max_results=10):
    """Search YouTube videos from the past 2 days using Apify"""
    if not APIFY_AVAILABLE:
        logger.error("Apify client not available. Please install with: pip install apify-client")
        return {
            "success": False,
            "error": "Apify client not installed. Run: pip install apify-client",
            "videos": []
        }
    
    if not APIFY_API_KEY:
        logger.error("APIFY_API_KEY not set in environment or secrets")
        return {
            "success": False,
            "error": "Apify API key not configured. Please add it to your secrets.",
            "videos": []
        }
    
    try:
        client = ApifyClient(APIFY_API_KEY)
        
        # Calculate date 2 days ago
        two_days_ago = (datetime.utcnow() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        # Prepare the Actor input
        run_input = {
            "searchQueries": [query],
            "maxResults": max_results,
            "maxResultsShorts": 0,
            "maxResultStreams": 0,
            "postsFromDate": two_days_ago
        }
        
        logger.info(f"Starting YouTube search via Apify for: {query}")
        
        # Run the Actor and wait for it to finish
        run = client.actor("streamers/youtube-scraper").call(run_input=run_input)
        
        # Fetch results
        videos = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            if 'title' not in item or 'url' not in item:
                continue
                
            video = {
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'channel': item.get('channelName', ''),
                'description': item.get('description', ''),
                'publishedAt': item.get('uploadDate', ''),
                'views': item.get('viewCount', 0),
                'duration': item.get('duration', ''),
                'likes': item.get('likeCount', 0)
            }
            videos.append(video)
        
        # Filter by relevance
        if videos:
            filtered_videos = filter_by_relevance(videos, query, min(5, len(videos)))
            return {
                "success": True,
                "videos": filtered_videos
            }
        else:
            return {
                "success": True,
                "videos": [],
                "message": "No videos found matching your query."
            }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"An error occurred during YouTube search: {error_msg}")
        return {
            "success": False,
            "error": f"An error occurred during YouTube search: {error_msg}",
            "videos": []
        }

def filter_by_relevance(videos, query, top_n=5):
    """Filter videos based on title relevance using TF-IDF and cosine similarity"""
    if not videos:
        return []
        
    # Prepare texts for comparison
    titles = [video['title'] for video in videos]
    texts = titles + [query]
    
    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate similarity between query and each title
    query_vector = tfidf_matrix[-1]
    title_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(title_vectors, query_vector)
    
    # Get top N most relevant videos
    top_indices = np.argsort(similarities.flatten())[-top_n:][::-1]
    return [videos[i] for i in top_indices]

#########################
# Data Models
#########################

class YouTubeVideo(BaseModel):
    """YouTube video information"""
    title: str = Field(..., description="Title of the video")
    url: str = Field(..., description="URL of the video") 
    channel: str = Field("", description="Channel name")
    description: str = Field("", description="Video description")
    publishedAt: str = Field("", description="Publication date")
    views: int = Field(0, description="View count")
    duration: str = Field("", description="Video duration")
    likes: int = Field(0, description="Like count")

class YouTubeAnalysis(BaseModel):
    """Analysis of a YouTube video"""
    title: str = Field(..., description="Title of the video")
    summary: str = Field(..., description="Summary of the video content")
    key_findings: List[Dict[str, str]] = Field(..., description="Key findings with timestamps")
    transcript_excerpts: List[Dict[str, str]] = Field(..., description="Selected transcript excerpts with timestamps")
    video_url: str = Field(..., description="URL of the YouTube video")

class WebSearchResult(BaseModel):
    """Results from a web search"""
    title: str = Field(..., description="Title of the search result")
    content: str = Field(..., description="Content snippet from the search result")
    url: str = Field(..., description="URL of the search result")
    
class ResearchStepResult(BaseModel):
    """Results from a single research step"""
    step_number: int = Field(..., description="The step number")
    step_name: str = Field(..., description="Name of the research step")
    description: str = Field(..., description="Description of what was done")
    source_used: str = Field(..., description="Source used for this step")
    search_query: Optional[str] = Field(None, description="Search query used, if applicable")
    findings: str = Field(..., description="Summary of findings from this step")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="Raw search results")

class ResearchStep(BaseModel):
    """A single step in the research process"""
    step_number: int
    step_name: str
    description: str
    source_used: str
    search_query: Optional[str] = None
    findings: str
    
    # Using a simpler structure for results
    result_titles: List[str] = Field(default_factory=list)
    result_contents: List[str] = Field(default_factory=list)
    result_urls: List[str] = Field(default_factory=list)

class ResearchResult(BaseModel):
    """Complete research results"""
    user_query: str
    user_intent: str
    key_topics: List[str]
    research_steps: List[ResearchStep]
    summary: str
    conclusion: str
    further_research: Optional[List[str]] = None
class ResearchDeps(BaseModel):
    """Dependencies for research operations"""
    gemini_api_key: str = Field(..., description="API key for Gemini")
    available_sources: Dict[str, bool] = Field(..., description="Available research sources")
    youtube_urls: Optional[List[str]] = Field(None, description="YouTube URLs to analyze")
    
    class Config:
        arbitrary_types_allowed = True

# Simplified data models
class SimpleResearchStep(BaseModel):
    """A simplified research step model"""
    step_number: int
    step_name: str
    description: str
    source_used: str
    search_query: Optional[str] = None
    findings: str
    # Flat arrays instead of nested objects
    result_titles: List[str] = Field(default_factory=list)
    result_contents: List[str] = Field(default_factory=list)
    result_urls: List[str] = Field(default_factory=list)

class SimpleResearchResult(BaseModel):
    """A simplified research result model"""
    user_query: str
    user_intent: str
    key_topics: List[str]
    research_steps: List[SimpleResearchStep] = Field(default_factory=list)
    summary: str
    conclusion: str
    further_research: List[str] = Field(default_factory=list)

class NarrativeInputSimple(BaseModel):
    """Simplified input for narrative synthesis"""
    query: str = Field(..., description="The original research query")
    summary: str = Field(..., description="Research summary")
    key_points: List[str] = Field(..., description="Key research points as simple strings")
    style_template: str = Field(..., description="Template text for narrative style")

class NarrativeOutputSimple(BaseModel):
    """Simplified output for narrative synthesis"""
    narrative_report: str = Field(..., description="The narrative synthesis report")
    key_points: List[str] = Field(
        default_factory=list, 
        description="Key points from the research as simple strings"
    )
    
#########################
# Research Assistant Agent
#########################

research_assistant = Agent(
    model=gemini_model,
    deps_type=ResearchDeps,
    result_type=ResearchResult,
    system_prompt="""You are an expert research assistant who helps analysts find information efficiently.

When given a research query, you will:
1. Analyze the user's intent and research needs
2. Extract key topics that need to be investigated
3. Create a research plan with clearly defined steps
4. Execute the research by calling the appropriate tools
5. Synthesize findings into a comprehensive analysis

You have the following tools available:
- web_search: Search the web for general information
- youtube_search: Search for relevant YouTube videos
- youtube_analysis: Analyze YouTube videos for relevant content

For each research query, use the appropriate tools in sequence based on what information is needed.
Be thorough but efficient, focusing on high-quality sources.

Your final response should include:
- Analysis of the user's intent
- Key topics that were researched
- Step-by-step results of your research process
- A comprehensive summary of findings
- A clear conclusion
- Suggestions for further research if needed

When including YouTube analysis results, present them in a narrative style with timestamps, similar to an interview transcript.
"""
)

#########################
# Narrative Synthesis Agent
#########################

narrative_synthesis_agent = Agent(
    model=gemini_model,
    deps_type=NarrativeInputSimple,
    result_type=NarrativeOutputSimple,
    system_prompt="""You are a specialized agent that synthesizes research findings into a cohesive narrative.

Your task is to:
1. Review the research query, summary, and key points
2. Follow the provided style template
3. Create a conversational narrative that includes timestamps
4. Structure the report as if it were a transcript of an interview

Make the narrative engaging, clear, and easy to follow, while maintaining the conversational style of the template.
"""
)

class SpeechScriptInputModel(BaseModel):
    """Input model for speech script generation"""
    research_query: str = Field(..., description="The original research query")
    key_findings: List[str] = Field(..., description="Key findings from research")
    main_points: List[str] = Field(..., description="Main points to cover in speech")
    source_highlights: List[str] = Field(..., description="Highlights from key sources")
    youtube_insights: List[str] = Field(default_factory=list, description="Insights from YouTube analysis")
    target_duration: str = Field("5-7 minutes", description="Target speech duration")

class SpeechScriptOutputModel(BaseModel):
    """Output model for speech script generation"""
    title: str = Field(..., description="Title of the speech")
    introduction: str = Field(..., description="Introduction section")
    body_sections: List[str] = Field(..., description="Main body sections")
    conclusion: str = Field(..., description="Conclusion section")
    delivery_notes: List[str] = Field(..., description="Notes on delivery style")
    full_script: str = Field(..., description="Complete formatted script")

speech_script_agent = Agent(
    model=gemini_model,
    deps_type=SpeechScriptInputModel,
    result_type=SpeechScriptOutputModel,
    system_prompt="""You are an expert speech writer who creates polished, professional delivery scripts based on research findings.

Your task is to:
1. Create a compelling, structured speech script that follows professional presentation standards
2. Incorporate key research findings into a logical narrative flow
3. Include proper pacing, emphasis points, and delivery notes
4. Format the script for easy reading during presentation with clear sections
5. Include timestamps and speaker notes where appropriate
6. Maintain a professional, authoritative tone suitable for business or academic contexts

The final script should:
- Begin with a strong hook and clear introduction of the topic
- Structure information in a logical sequence with smooth transitions
- Emphasize the most important findings with supporting evidence
- Include brief reference to sources where appropriate
- End with a clear conclusion and actionable takeaways
- Include formatting that indicates pacing, emphasis, and delivery style

Format the script with clear sections, timestamps in [MM:SS] format, and delivery notes in {italics} or [DELIVERY NOTES] where appropriate.
"""
)

async def generate_speech_script(research_results, youtube_analyses=None):
    """Generate a professional speech script based on research findings and YouTube analysis."""
    try:
        # Extract key information from research results
        if not research_results:
            return {
                "error": "No research results available",
                "message": "Please complete research before generating a speech script."
            }
            
        # Extract key information from research results
        key_findings = []
        main_points = []
        source_highlights = []
        
        # Extract from research steps
        if hasattr(research_results, "research_steps"):
            for step in research_results.research_steps:
                if hasattr(step, "findings") and step.findings:
                    key_findings.append(step.findings)
                
                # Extract sources
                if hasattr(step, "source_used") and step.source_used:
                    source_info = f"From {step.source_used}"
                    if hasattr(step, "search_query") and step.search_query:
                        source_info += f": {step.search_query}"
                    source_highlights.append(source_info)
        
        # Extract key topics
        if hasattr(research_results, "key_topics"):
            main_points.extend(research_results.key_topics)
            
        # Add summary and conclusion
        if hasattr(research_results, "summary") and research_results.summary:
            main_points.append(research_results.summary)
            
        if hasattr(research_results, "conclusion") and research_results.conclusion:
            key_findings.append(research_results.conclusion)
            
        # Extract YouTube insights
        youtube_insights = []
        if youtube_analyses:
            if isinstance(youtube_analyses, list):
                for analysis in youtube_analyses:
                    if isinstance(analysis, str):
                        youtube_insights.append(f"From YouTube analysis: {analysis[:200]}...")
                    elif isinstance(analysis, dict) and "summary" in analysis:
                        youtube_insights.append(f"From YouTube video: {analysis['summary'][:200]}...")
            elif isinstance(youtube_analyses, dict) and "video_analyses" in youtube_analyses:
                for video in youtube_analyses["video_analyses"]:
                    if "summary" in video:
                        youtube_insights.append(f"From YouTube video: {video['summary'][:200]}...")
        
        # Create input model
        script_input = SpeechScriptInputModel(
            research_query=getattr(research_results, "user_query", "Research topic"),
            key_findings=key_findings[:5] if len(key_findings) > 5 else key_findings,
            main_points=main_points[:5] if len(main_points) > 5 else main_points,
            source_highlights=source_highlights[:5] if len(source_highlights) > 5 else source_highlights,
            youtube_insights=youtube_insights[:3] if len(youtube_insights) > 3 else youtube_insights
        )
        
        # Generate speech script
        result = await speech_script_agent.run(
            user_prompt="Create a professional speech delivery script based on these research findings. Format it with clear sections, proper pacing, and delivery notes.",
            deps=script_input
        )
        
        if result and hasattr(result, 'data'):
            return result.data
        else:
            return {
                "error": "Failed to generate speech script",
                "message": "An error occurred while generating the speech script."
            }
            
    except Exception as e:
        logger.error(f"Error generating speech script: {str(e)}")
        return {
            "error": f"Error generating speech script: {str(e)}",
            "message": "An error occurred while generating the speech script. Please try again."
        }

def display_speech_script(script):
    """Display formatted speech script in the Streamlit UI."""
    if isinstance(script, str):
        st.error(script)
        return
        
    if isinstance(script, dict) and "error" in script:
        st.error(script["error"])
        if "message" in script:
            st.write(script["message"])
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Full Script", "Script Sections", "Audio"])
    
    with tab1:
        # Display speech title
        st.header(f"📢 {script.title}")
        
        # Display delivery notes
        with st.expander("Delivery Notes", expanded=False):
            for note in script.delivery_notes:
                st.write(f"• {note}")
                
        # Format the full script with proper markdown
        formatted_script = script.full_script
        
        # Convert [MM:SS] timestamps to bold
        import re
        formatted_script = re.sub(r'\[(\d+:\d+)\]', r'**[\1]**', formatted_script)
        
        # Convert {delivery notes} to italics
        formatted_script = re.sub(r'\{([^}]+)\}', r'*{\1}*', formatted_script)
        
        # Convert [DELIVERY NOTES] to highlighted text
        formatted_script = re.sub(r'\[([A-Z\s]+)\]', r'<span style="background-color: #f0f0f0; padding: 2px 5px; border-radius: 3px;">\1</span>', formatted_script)
        
        st.markdown(formatted_script, unsafe_allow_html=True)
    
    with tab2:
        # Introduction
        st.subheader("Introduction")
        st.markdown(script.introduction)
        
        # Body sections
        st.subheader("Main Content")
        for i, section in enumerate(script.body_sections, 1):
            with st.expander(f"Section {i}", expanded=i==1):
                st.markdown(section)
        
        # Conclusion
        st.subheader("Conclusion")
        st.markdown(script.conclusion)
    
    with tab3:
        st.header(f"📢 Audio Generation for: {script.title}")
        
        # Voice selection dropdown
        voice_options = {
            "Rachel (Female)": "JBFqnCBsd6RMkjVDRZzb",
            "Adam (Male)": "pNInz6obpgDQGcFmaJgB",
            "Antoni (Male)": "ErXwobaYiN019PkySvjV",
            "Bella (Female)": "EXAVITQu4vr4xnSDxMaL",
            "Elli (Female)": "MF3mGyEYCl7XYWbV9V6O"
        }
        
        selected_voice = st.selectbox("Select Voice:", options=list(voice_options.keys()))
        
        # Show API key status
        with st.expander("ElevenLabs API Status"):
            api_status = check_elevenlabs_api_key()
            st.write("API Key Status:", "Available ✅" if api_status else "Not Available ❌")
            if not api_status:
                st.warning("Please add your ElevenLabs API key to secrets to use voice generation.")
        
        # Section selection
        section_options = ["Full Speech", "Introduction Only", "Conclusion Only"]
        section_options.extend([f"Section {i+1}" for i in range(len(script.body_sections))])
        
        selected_section = st.selectbox("Choose section to generate:", options=section_options)
        
        # Determine text based on selection
        text_to_convert = ""
        if selected_section == "Full Speech":
            # Clean the script of formatting markers for speech generation
            clean_script = re.sub(r'\[\d+:\d+\]|\{[^}]+\}|\[[A-Z\s]+\]', '', script.full_script)
            text_to_convert = clean_script
        elif selected_section == "Introduction Only":
            clean_intro = re.sub(r'\[\d+:\d+\]|\{[^}]+\}|\[[A-Z\s]+\]', '', script.introduction)
            text_to_convert = clean_intro
        elif selected_section == "Conclusion Only":
            clean_conclusion = re.sub(r'\[\d+:\d+\]|\{[^}]+\}|\[[A-Z\s]+\]', '', script.conclusion)
            text_to_convert = clean_conclusion
        else:
            section_index = int(selected_section.split(" ")[1]) - 1
            if 0 <= section_index < len(script.body_sections):
                clean_section = re.sub(r'\[\d+:\d+\]|\{[^}]+\}|\[[A-Z\s]+\]', '', script.body_sections[section_index])
                text_to_convert = clean_section
        
        # Preview selected text
        with st.expander("Preview text to be converted", expanded=False):
            st.write(text_to_convert)
        
        # Generate audio button
        if st.button("Generate Audio"):
            if not text_to_convert:
                st.warning("No text selected for conversion.")
            else:
                with st.spinner("Generating audio..."):
                    voice_id = voice_options[selected_voice]
                    audio_data = text_to_speech(text_to_convert, voice_id=voice_id)
                    
                    if audio_data:
                        st.success(f"Audio generated successfully! Size: {len(audio_data)/1024:.1f} KB")
                        
                        # Display audio player
                        st.audio(io.BytesIO(audio_data), format="audio/mp3")
                        
                        # Add download button
                        filename = f"{script.title.replace(' ', '_')}_{selected_section.replace(' ', '_')}.mp3"
                        st.download_button(
                            label="Download Audio",
                            data=audio_data,
                            file_name=filename,
                            mime="audio/mp3"
                        )
                    else:
                        st.error("Failed to generate audio. Please check your API key and try again.")
    
    # Add download button for the script text
    script_text = f"# {script.title}\n\n"
    script_text += "## Delivery Notes\n"
    for note in script.delivery_notes:
        script_text += f"- {note}\n"
    script_text += "\n## Full Script\n\n"
    script_text += script.full_script
    
    st.download_button(
        label="Download Speech Script",
        data=script_text,
        file_name="speech_script.md",
        mime="text/markdown",
    )

#########################
# Streamlit UI Functions
#########################

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "research_results" not in st.session_state:
        st.session_state["research_results"] = None
    if "youtube_analyses" not in st.session_state:
        st.session_state["youtube_analyses"] = []
    if "youtube_url" not in st.session_state:
        st.session_state["youtube_url"] = ""
    if "youtube_template" not in st.session_state:
        st.session_state["youtube_template"] = ""
    if "youtube_search_results" not in st.session_state:
        st.session_state["youtube_search_results"] = {"success": False, "videos": []}
    if "trigger_research" not in st.session_state:
        st.session_state["trigger_research"] = False
    if "trigger_youtube_analysis" not in st.session_state:
        st.session_state["trigger_youtube_analysis"] = False
    if "trigger_youtube_search" not in st.session_state:
        st.session_state["trigger_youtube_search"] = False
    if "available_sources" not in st.session_state:
        st.session_state["available_sources"] = {
            "Web Search": True,
            "YouTube": True
        }
    if "narrative_synthesis" not in st.session_state:
        st.session_state["narrative_synthesis"] = None
    if "youtube_analysis_query" not in st.session_state:
        st.session_state["youtube_analysis_query"] = ""
    if "youtube_search_query" not in st.session_state:
        st.session_state["youtube_search_query"] = ""
    # Speech script and audio
    if "speech_script" not in st.session_state:
        st.session_state["speech_script"] = None
    if "trigger_speech_generation" not in st.session_state:
        st.session_state["trigger_speech_generation"] = False
    if "speech_audio" not in st.session_state:
        st.session_state["speech_audio"] = None

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
    st.title("📚 Research Assistant")
    st.markdown("*Helping analysts plan and conduct thorough research*")
    
    st.subheader("Research Sources")
    st.write("Available sources for gathering information:")
    
    sources = {}
    for source, default in st.session_state["available_sources"].items():
        sources[source] = st.checkbox(source, value=default)
    
    # Update session state with selected sources
    st.session_state["available_sources"] = sources
    
    # YouTube URL input
    st.subheader("YouTube Integration")
    col1, col2 = st.columns(2)
    
    with col1:
        youtube_search_query = st.text_input("Search YouTube:", key="youtube_search_query_input", 
                                           placeholder="Enter search term")
        if st.button("Search YouTube"):
            if youtube_search_query:
                st.session_state["youtube_search_query"] = youtube_search_query
                st.session_state["trigger_youtube_search"] = True
    
    with col2:
        youtube_url = st.text_input("Or enter URL:", key="youtube_url_input", 
                                   placeholder="https://youtu.be/...")
        youtube_query = st.text_input("Analysis focus:", key="youtube_query_input", 
                                     placeholder="Optional topic focus")
        if st.button("Analyze Video"):
            if youtube_url:
                st.session_state["youtube_url"] = youtube_url
                st.session_state["trigger_youtube_analysis"] = True
                st.session_state["youtube_analysis_query"] = youtube_query
    
    # Add speech generation section
    st.subheader("Speech Generation")
    
    if st.button("Generate Speech Script"):
        st.session_state["trigger_speech_generation"] = True
    
    # Add ElevenLabs API section
    st.subheader("ElevenLabs Integration")
    elevenlabs_api_key = st.text_input("ElevenLabs API Key:", type="password", 
                                      help="Get your API key from app.elevenlabs.io")
    
    if elevenlabs_api_key and st.button("Save API Key"):
        # In a real app, you would save this to secrets
        # Here we just update our session
        os.environ["ELEVEN_LABS_API_KEY"] = elevenlabs_api_key
        st.success("API key saved for this session!")
    
    return sources


async def execute_youtube_search(query):
    """Execute YouTube search using Apify."""
    if not APIFY_AVAILABLE or not APIFY_API_KEY:
        return {
            "success": False,
            "error": "Apify client not available or API key not set",
            "videos": []
        }
    
    results = search_youtube_videos(query)
    return results

async def execute_research(query, available_sources, youtube_urls=None):
    """Execute research using the research assistant agent."""
    try:
        # Set up dependencies for the research assistant
        deps = ResearchDeps(
            gemini_api_key=GEMINI_API_KEY,
            available_sources=available_sources,
            youtube_urls=youtube_urls if youtube_urls else []
        )
        
        # Log the setup
        logger.info(f"Executing research for query: {query}")
        logger.info(f"Available sources: {available_sources}")
        logger.info(f"YouTube URLs provided: {youtube_urls}")
        
        # Run the research assistant
        result = await research_assistant.run(
            user_prompt=f"Research the following query thoroughly: {query}",
            deps=deps
        )
        
        # Convert to a serializable format if needed
        if result and hasattr(result, 'data'):
            research_data = result.data
            
            # Create a SimpleResearchResult from the ResearchResult model
            simple_result = SimpleResearchResult(
                user_query=research_data.user_query,
                user_intent=research_data.user_intent,
                key_topics=research_data.key_topics,
                summary=research_data.summary,
                conclusion=research_data.conclusion,
                further_research=research_data.further_research if research_data.further_research else []
            )
            
            # Convert research steps to SimpleResearchStep objects
            for step in research_data.research_steps:
                simple_step = SimpleResearchStep(
                    step_number=step.step_number,
                    step_name=step.step_name,
                    description=step.description,
                    source_used=step.source_used,
                    search_query=step.search_query,
                    findings=step.findings,
                    result_titles=step.result_titles,
                    result_contents=step.result_contents,
                    result_urls=step.result_urls
                )
                simple_result.research_steps.append(simple_step)
            
            return simple_result
        else:
            logger.error("Research result is empty or invalid")
            return "Error: Research result is empty or invalid"
    except Exception as e:
        logger.error(f"Error executing research: {str(e)}")
        return f"Error executing research: {str(e)}"

def display_youtube_analysis(analysis):
    """Display YouTube analysis results."""
    if isinstance(analysis, str):
        # Display raw text or error message
        st.markdown(analysis)
        
        # Add a button to use this as a template with a unique key
        unique_key = generate_unique_key(analysis)
        if st.button("Use as Template for Narrative Style", key=f"template_btn_{unique_key}"):
            st.session_state["youtube_template"] = analysis
            st.success("Template saved for narrative synthesis!")
        return
    
    # For structured analysis (if implemented)
    st.subheader("YouTube Video Analysis")
    st.markdown(analysis)

def display_youtube_search_results(results):
    """Display YouTube search results."""
    if not results.get("success", False):
        # st.error(results.get("error", "An error occurred during YouTube search."))
        return
    
    videos = results.get("videos", [])
    if not videos:
        st.warning("No videos found matching your search query.")
        return
    
    st.subheader(f"Found {len(videos)} relevant YouTube videos")
    
    for i, video in enumerate(videos):
        with st.expander(f"{i+1}. {video['title']}", expanded=i==0):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Channel:** {video['channel']}")
                st.write(f"**Published:** {video['publishedAt']}")
                st.write(f"**Duration:** {video['duration']}")
                st.write(f"**Views:** {video['views']:,}")
                st.write(f"**Likes:** {video['likes']:,}")
                
                # Button to analyze this video
                if st.button("Analyze This Video", key=f"analyze_btn_{i}"):
                    st.session_state["youtube_url"] = video['url']
                    st.session_state["trigger_youtube_analysis"] = True
            
            with col2:
                st.write("**Description:**")
                st.write(video['description'][:300] + "..." if len(video['description']) > 300 else video['description'])
                st.write(f"**URL:** [{video['url']}]({video['url']})")

def display_research_results(results):
    """Display formatted research results."""
    if isinstance(results, str):
        # Error message
        st.error(results)
        return
    
    # User Intent Analysis
    st.header("Research Analysis")
    st.subheader("📋 User Intent")
    st.write(results.user_intent)
    
    # Key Topics
    st.subheader("🔑 Key Topics Researched")
    for i, topic in enumerate(results.key_topics, 1):
        st.write(f"{i}. {topic}")
    
    # Research Steps & Findings
    st.header("🔍 Research Process & Findings")
    for step in results.research_steps:
        with st.expander(f"Step {step.step_number}: {step.step_name}", expanded=True):
            st.write("**Description:**")
            st.write(step.description)
            
            st.write("**Source Used:**")
            st.write(step.source_used)
            
            if step.search_query:
                st.write("**Search Query:**")
                st.write(step.search_query)
            
            st.write("**Findings:**")
            st.write(step.findings)
            
            # Display results if available
            if step.result_titles:
                st.write("**Source Results:**")
                for i, (title, content, url) in enumerate(
                    zip(step.result_titles, step.result_contents, step.result_urls)
                ):
                    st.write(f"**{i+1}. [{title}]({url})**")
                    st.write(content)
                    st.write("---")
    
    # Summary & Conclusion
    st.header("📊 Research Summary")
    st.write(results.summary)
    
    st.header("🎯 Conclusion")
    st.write(results.conclusion)
    
    # Further Research
    if results.further_research:
        st.header("🔮 Suggestions for Further Research")
        for i, suggestion in enumerate(results.further_research, 1):
            st.write(f"{i}. {suggestion}")

def display_narrative_synthesis(synthesis):
    """Display narrative synthesis of research findings."""
    if isinstance(synthesis, str):
        # Error message
        st.error(synthesis)
        return
    
    # Check if the synthesis is a dictionary (error response)
    if isinstance(synthesis, dict) and "error" in synthesis:
        st.error(synthesis["error"])
        if "narrative_report" in synthesis:
            st.markdown(synthesis["narrative_report"])
        return
    
    st.header("📝 Narrative Research Report")
    
    # Display the narrative report
    if hasattr(synthesis, "narrative_report"):
        st.markdown(synthesis.narrative_report)
    
    # Display key points
    with st.expander("Key Points", expanded=False):
        if hasattr(synthesis, "key_points") and synthesis.key_points:
            for point in synthesis.key_points:
                st.write(point)
        else:
            st.write("No key points available in the synthesis output.")


#########################
# Narrative Synthesis Function
#########################

async def create_narrative_synthesis(research_findings, style_template):
    """Create a narrative synthesis of research findings using the provided template style."""
    try:
        # Extract key points - keep everything as simple strings
        key_points = []
        
        # Handle different types of input
        if hasattr(research_findings, "research_steps"):
            for step in research_findings.research_steps:
                key_points.append(step.findings)
        
        if hasattr(research_findings, "key_topics"):
            for topic in research_findings.key_topics:
                key_points.append(topic)
        
        # Create simplified input
        synthesis_input = NarrativeInputSimple(
            query=getattr(research_findings, "user_query", "Unknown query"),
            summary=getattr(research_findings, "summary", "No summary available"),
            key_points=key_points,
            style_template=style_template
        )
        
        # Run the narrative synthesis agent
        result = await narrative_synthesis_agent.run(
            user_prompt="Create a narrative synthesis of these research findings in the style of the provided template.",
            deps=synthesis_input
        )
        
        # Get the result data
        if result and hasattr(result, 'data'):
            return result.data
        else:
            return {
                "error": "Failed to generate narrative synthesis",
                "narrative_report": "An error occurred during narrative synthesis."
            }
            
    except Exception as e:
        logger.error(f"Error creating narrative synthesis: {str(e)}")
        return {
            "error": f"Error creating narrative synthesis: {str(e)}",
            "narrative_report": "An error occurred during narrative synthesis. Please try again."
        }



#########################
# Direct YouTube Analysis Function
#########################

async def analyze_youtube_url(youtube_url, query=""):
    """Analyze a single YouTube URL using Gemini."""
    try:
        # Initialize the Gemini client
        client = genai.Client(api_key=GEMINI_API_KEY)
        model = "gemini-2.0-flash"
        
        prompt = """Analyze this YouTube video and provide:
        1. Title
        2. Summary of the content
        3. Key findings with timestamps [MM:SS]
        4. Transcript excerpts with timestamps [MM:SS] of the most relevant sections
        
        Format the output like an interview transcript with clear timestamps for each section."""
        
        if query:
            prompt += f"\n\nFocus your analysis on topics related to: {query}"
        
        # Set up the content with the YouTube URL and query
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=youtube_url,
                        mime_type="video/*",
                    ),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="text/plain",
        )
        
        # Execute the analysis
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        return response.text
    except Exception as e:
        logger.error(f"Error analyzing YouTube URL: {str(e)}")
        return f"Error analyzing YouTube URL: {str(e)}"
    
async def main():
    """Main function for the Streamlit app."""
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar
    with st.sidebar:
        available_sources = setup_sidebar()
        
        # Chat interface for input
        st.subheader("Chat Interface")
        display_chat_messages()
        prompt = handle_user_input()
    
    # Main content area - title and description
    st.title("Research Assistant")
    st.write("Enter your research query in the chat and I'll automatically gather information using the selected sources.")
    
    # Process YouTube search if triggered
    if st.session_state["trigger_youtube_search"]:
        st.session_state["trigger_youtube_search"] = False
        
        with st.spinner("Searching YouTube videos..."):
            search_query = st.session_state["youtube_search_query"]
            search_results = await execute_youtube_search(search_query)
            st.session_state["youtube_search_results"] = search_results
    
    # Display YouTube search results if available
    if st.session_state.get("youtube_search_results"):
        display_youtube_search_results(st.session_state["youtube_search_results"])
    
    # Process YouTube analysis if triggered
    if st.session_state["trigger_youtube_analysis"]:
        st.session_state["trigger_youtube_analysis"] = False
        
        with st.spinner("Analyzing YouTube video..."):
            youtube_url = st.session_state["youtube_url"]
            youtube_query = st.session_state.get("youtube_analysis_query", "")  # Use correct key
            
            analysis = await analyze_youtube_url(youtube_url, youtube_query)
            st.session_state["youtube_analyses"].append(analysis)
            st.session_state["youtube_template"] = analysis
            
            # Display success message in sidebar
            with st.sidebar:
                st.success(f"YouTube video analyzed successfully!")
    
    # Display YouTube analyses if available
    if st.session_state["youtube_analyses"]:
        st.subheader("YouTube Video Analyses")
        for i, analysis in enumerate(st.session_state["youtube_analyses"]):
            with st.expander(f"Video Analysis {i+1}", expanded=i == len(st.session_state["youtube_analyses"])-1):
                display_youtube_analysis(analysis)
    
    # Process research if triggered
    if st.session_state["trigger_research"]:
        st.session_state["trigger_research"] = False
        
        # Process the research query
        with st.status("Processing your research query...", expanded=True) as status:
            query = st.session_state.messages[-1]["content"]
            
            # Get any YouTube URLs to include
            youtube_urls = []
            if st.session_state.get("youtube_search_results") and st.session_state["youtube_search_results"].get("success", False):
                # Use the top video from search results
                top_video = st.session_state["youtube_search_results"]["videos"][0]
                youtube_urls.append(top_video["url"])
            
            if st.session_state["youtube_url"]:
                youtube_urls.append(st.session_state["youtube_url"])
            
            status.update(label="Executing automated research...")
            research_results = await execute_research(query, st.session_state["available_sources"], youtube_urls)
            st.session_state["research_results"] = research_results
            
            # Create narrative synthesis if template exists
            if st.session_state["youtube_template"] and not isinstance(research_results, str):
                status.update(label="Creating narrative synthesis...")
                synthesis = await create_narrative_synthesis(research_results, st.session_state["youtube_template"])
                st.session_state["narrative_synthesis"] = synthesis
            
            # Add response to chat history
            if isinstance(research_results, str):
                # Error occurred
                response = f"Error during research: {research_results}"
            else:
                response = "I've conducted research on your query. Here are my findings:"
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            with st.sidebar:
                with st.chat_message("assistant"):
                    st.markdown(response)
            
            status.update(label="Research completed!", state="complete")

    # Process speech generation if triggered
    if st.session_state["trigger_speech_generation"]:
        st.session_state["trigger_speech_generation"] = False
        
        with st.status("Generating speech script...", expanded=True) as status:
            # Generate speech script
            speech_script = await generate_speech_script(
                st.session_state.get("research_results"), 
                st.session_state.get("youtube_analyses", [])
            )
            st.session_state["speech_script"] = speech_script
            
            status.update(label="Speech script generated!", state="complete")
    
    # Display speech script if available
    if st.session_state.get("speech_script"):
        st.markdown("---")
        display_speech_script(st.session_state["speech_script"])

    
    # Display the research results
    if st.session_state.get("research_results"):
        display_research_results(st.session_state["research_results"])
    
    # Narrative Synthesis Section  
    if st.session_state.get("narrative_synthesis"):
        st.markdown("---")
        display_narrative_synthesis(st.session_state["narrative_synthesis"])

if __name__ == "__main__":
    safe_async_run(main())