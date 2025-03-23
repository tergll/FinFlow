import streamlit as st
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from elevenlabs.core.api_error import ApiError
import os
import time
import httpx
from typing import Optional
import io

# Load environment variables
load_dotenv()

def check_api_key() -> bool:
    """Check if the ElevenLabs API key exists in the environment variables"""
    api_key = st.secrets["ELEVEN_LABS_API_KEY"]
    if not api_key:
        st.error("Error: ELEVEN_LABS_API_KEY not found in .env file")
        return False
    return True

def text_to_speech(text: str, max_retries: int = 3) -> Optional[bytes]:
    """Convert text to speech using ElevenLabs API"""
    if not check_api_key():
        return None
        
    client = ElevenLabs(
        api_key=st.secrets["ELEVEN_LABS_API_KEY"],
    )
    
    for attempt in range(max_retries):
        try:
            audio = client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
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

# Streamlit app
def main():
    st.title("Text to Speech with ElevenLabs")
    
    # App description
    st.markdown("""
    This application converts text to speech using the ElevenLabs API.
    Enter your text below and click the 'Generate Audio' button.
    """)
    
    # Text input
    text_input = st.text_area("Enter text to convert to speech:", 
                             height=150,
                             placeholder="Type your text here...")
    
    # Voice selection (you can add more voices here)
    voice_options = {
        "Rachel": "JBFqnCBsd6RMkjVDRZzb",
        # Add more voices as needed
    }
    
    voice = st.selectbox("Select voice:", options=list(voice_options.keys()))
    
    # Debug section (can be removed in production)
    with st.expander("Debug Information"):
        st.write("API Key Status:", "Available" if check_api_key() else "Not Available")
    
    # Generate button
    if st.button("Generate Audio"):
        if not text_input:
            st.warning("Please enter some text.")
        else:
            with st.spinner("Generating audio..."):
                st.info("Sending request to ElevenLabs API...")
                audio_data = text_to_speech(text_input)
                
                if audio_data:
                    st.success(f"Audio generated successfully! Size: {len(audio_data)} bytes")
                    # Display audio player
                    st.audio(io.BytesIO(audio_data), format="audio/mp3")
                    
                    # Add download button
                    st.download_button(
                        label="Download Audio",
                        data=audio_data,
                        file_name="speech.mp3",
                        mime="audio/mp3"
                    )
                else:
                    st.error("Failed to generate audio. Please check your API key and try again.")

if __name__ == "__main__":
    main()