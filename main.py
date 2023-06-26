import os
from soniox.speech_service import SpeechClient
from soniox.transcribe_live import transcribe_microphone
from functionalities.sales_gpt import SalesGPT
from langchain.chat_models import ChatOpenAI
from elevenlabs import stream
import time
from io import BytesIO
import requests
import cProfile
import io 
import pstats
import langchain
from langchain.cache import InMemoryCache
from dotenv import load_dotenv
from helpers import * 
# load environment variables
load_dotenv(dotenv_path="configs/.env", override=True, verbose=True)

# Caching the responses in LLM memory
langchain.llm_cache = InMemoryCache()

# importing the environment variables in a more safe way 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

# define global variables
agent_is_speaking = False


def play_agent_response(text: str, voice_id: str = "pNInz6obpgDQGcFmaJgB", model_id: str = "eleven_monolingual_v1", optimize_streaming_latency: int = 4):
    global agent_is_speaking

    print(f"Playing agent response: {text}")
    agent_is_speaking = True

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?optimize_streaming_latency=4"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "0ddc8db042045085b262085b0acc096a"
    }

    data = {
        "text": text,
        "model_id": model_id,
        "optimize_streaming_latency": optimize_streaming_latency
    }

    response = requests.post(url, headers=headers, json=data, stream=True)

    # Ensure the response is valid
    if response.status_code == 200:
        # Create a BytesIO buffer to store audio data
        audio_data = BytesIO()
        for chunk in response.iter_content(chunk_size=512):
            if chunk:
                audio_data.write(chunk)

        # Reset the position in the buffer
        audio_data.seek(0)

        # Add silence to the beginning of the audio
        audio_data = add_silence_to_wav(convert_mp3_to_wav(audio_data.read()))

        # Stream the buffered audio data
        stream(BytesIO(audio_data))
    else:
        print(f"Error streaming audio: {response.status_code} {response.text}")

    agent_is_speaking = False




def main():
    # Initialize conversation
    llm = ChatOpenAI(temperature=0.9)
    sales_agent = SalesGPT.from_llm(llm)

    # Initialize SpeechClient and transcriber
    with SpeechClient() as client:

        count = 0
        max_num_turns = 4

        while count != max_num_turns:
            # Agent speaks
            count += 1
            sales_agent.step()
            agent_response = sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')
            print("Agent response:", agent_response)
            play_agent_response(agent_response)

            # User speaks
            final_words = []
            print("Transcribing from your microphone...")

            while agent_is_speaking:
                time.sleep(0.001)

            for result in transcribe_microphone(client):
                new_final_words, non_final_words = split_words(result)
                final_words += new_final_words
                render_final_words(final_words)
                render_non_final_words(non_final_words)


                if len(final_words) > 2:
                    sales_agent.human_step(" ".join(final_words).strip())
                    print(f"User input sent to agent: {' '.join(final_words).strip()}")
                    break

            if '<END_OF_CALL>' in agent_response:
                print('Sales Agent determined it is time to end the conversation.')
                break



if __name__ == "__main__":
    # Run the profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Call the function you want to profile
    main()
    
    # Stop the profiler
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('data/code_profiling/stats_soniox.txt', 'w+') as f:
        f.write(s.getvalue())