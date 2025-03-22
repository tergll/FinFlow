from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from elevenlabs.core.api_error import ApiError
import os
import time
import httpx
from typing import Optional

load_dotenv()

def check_api_key() -> bool:
    api_key = os.getenv("ELEVEN_LABS_API_KEY")
    if not api_key:
        print("Error: ELEVEN_LABS_API_KEY not found in .env file")
        return False
    return True

def text_to_speech(text: str, max_retries: int = 3) -> Optional[bytes]:
    if not check_api_key():
        return None
        
    client = ElevenLabs(
        api_key=os.getenv("ELEVEN_LABS_API_KEY"),
    )
    
    for attempt in range(max_retries):
        try:
            audio = client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
            return audio
        except ApiError as e:
            print(f"API Error: {str(e)}")
            return None
        except httpx.RemoteProtocolError as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return None
            print(f"Attempt {attempt + 1} failed, retrying in 2 seconds...")
            time.sleep(2)
    
    return None

# Example usage
if __name__ == "__main__":
    try:
        audio = text_to_speech("The first move is what sets everything in motion.")
        if audio:
            play(audio)
        else:
            print("Failed to generate audio")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
