import os
from soniox.speech_service import SpeechClient, Result
from soniox.transcribe_live import transcribe_microphone
from sales_gpt import SalesGPT
from langchain.chat_models import ChatOpenAI
from elevenlabs import stream
from typing import List, Tuple
import time
from scipy.io import wavfile
from io import BytesIO
from pydub import AudioSegment
import numpy as np
import requests
from pydub import AudioSegment
import cProfile
import io 
import pstats

os.environ['OPENAI_API_KEY'] = "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"
os.environ['SONIOX_API_KEY'] = "f7d0f5e9c111971168b9f9729048bc01ca16843a7fc50db9ca589d09b1c84318"
os.environ['ELEVEN_API_KEY'] = "0ddc8db042045085b262085b0acc096a"

agent_is_speaking = False


def add_silence_to_wav(wav_bytes, silence_duration = 0.05):
    fs_original, wav_data_original = wavfile.read(BytesIO(wav_bytes))
    silence_length = int(fs_original * silence_duration)
    silence_padding = np.zeros(silence_length)
    wav_data_padded = np.concatenate((silence_padding, wav_data_original))

    output_buf = BytesIO()
    wavfile.write(output_buf, fs_original, wav_data_padded.astype(np.int16))
    output_buf.seek(0)

    return output_buf.read()


def convert_mp3_to_wav(mp3_bytes):
    mp3_audio = AudioSegment.from_mp3(BytesIO(mp3_bytes))
    output_buf = BytesIO()
    mp3_audio.export(output_buf, format="wav")
    output_buf.seek(0)
    
    return output_buf.read()


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

def split_words(result:Result)->Tuple[List[str], List[str]]:
    """Split the words in a result into final and non-final words"""
    final_words, non_final_words = [], []
    for word in result.words:
        if word.is_final:
            final_words.append(word.text)
        else:
            non_final_words.append(word.text)
    return final_words, non_final_words


def render_final_words(words:List[str])->None:
    """Render the final words in a result"""
    if not words:
        return
    line = " ".join(words)
    print(f"User Input:{line}")

def render_non_final_words(words:List[str])->None:
    """Render the non-final words in a result"""
    if not words:
        return
    line = " ".join(words)
    print(f"Transcribing:{line}", end="\r")

import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

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

    with open('stats.txt', 'w+') as f:
        f.write(s.getvalue())