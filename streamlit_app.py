import streamlit as st
import os
import time
from io import BytesIO
from helpers import *
from langchain.cache import InMemoryCache
from langchain.chat_models import ChatOpenAI
from functionalities.sales_gpt import SalesGPT
from soniox.speech_service import SpeechClient
from soniox.transcribe_live import transcribe_microphone
from soniox.transcribe_file import transcribe_file_short
from elevenlabs import stream
import pyaudio
import requests
import soundfile as sf
import pvcheetah
import numpy as np
import cProfile
import io
import pstats
import langchain
from streamlit_chat import message 
import copy
import pygame
PICOVOICE_API_KEY = "SoMf0xO/J9PWWRHb3HSTHxDwiGY0RDbPebuEoJlZE/MuIecCZuGaqQ=="
os.environ['OPENAI_API_KEY'] = "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"

# Caching the responses in LLM memory
langchain.llm_cache = InMemoryCache()

# Define global variables
agent_is_speaking = False

# Picovoice initializations
selected_microphone_index = 1
CHANNELS = 1  # Mono
RATE = 16000  # Sample rate expected by DeepSpeech model
CHUNK_SIZE = 480  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format

def play_mp3_chunk(filelike_object, buffer_size=4096):
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=buffer_size)
    pygame.mixer.music.load(filelike_object)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.01)

    pygame.mixer.quit()


# Play agent response using ElevenLabs TTS API
def play_agent_response(text: str, voice_id: str = "pNInz6obpgDQGcFmaJgB", model_id: str = "eleven_monolingual_v1",
                        optimize_streaming_latency: int = 1):

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
        base_size = 5 * 128 * 1024 // 32
        size = base_size
        chunk_point = 0
        for chunk in response.iter_content(chunk_size=size):
            if chunk:
                audio_data.write(chunk)
                if audio_data.tell() > chunk_point + size:
                    data = copy.copy(audio_data)
                    data.seek(chunk_point)
                    play_mp3_chunk(BytesIO(data.read(size)), 4096)
                    chunk_point += size
                    size += base_size
        else:
            data = copy.copy(audio_data)
            data.seek(chunk_point)
            play_mp3_chunk(BytesIO(data.read()), 4096)
            chunk_point += size
            size += base_size
    else:
        print(f"Error streaming audio: {response.status_code} {response.text}")

    agent_is_speaking = False

# Function to transcribe audio stream using Cheetah ASR
def transcribe_audio_stream_using_cheetah(audio_chunk):
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

    # Ensure the audio_data input is int16 and one-dimensional (mono)
    assert audio_data.dtype == np.int16, f"Invalid data type: expected int16, got {audio_data.dtype}"
    assert len(audio_data.shape) == 1, f"Invalid data shape: expected one-dimensional, but got {audio_data.shape}"

    # Process audio data with Cheetah ASR
    partial_transcript, is_endpoint = cheetah.process(audio_data)

    return partial_transcript, is_endpoint

if "history" not in st.session_state:
    st.session_state.history = []
if "agent_responses" not in st.session_state:
    st.session_state.agent_responses = []

if "sales_agent" not in st.session_state:
    # Initialize the agent
    llm = ChatOpenAI(temperature=0.9)
    sales_agent = SalesGPT.from_llm(llm)
    st.session_state.sales_agent = sales_agent

def main():

    # Create an initial Cheetah ASR object to get the frame length
    cheetah = pvcheetah.create(access_key=PICOVOICE_API_KEY)
    CHUNK_SIZE = cheetah.frame_length  # Use the frame length specified by Cheetah
    cheetah = None  # Allow the Python garbage collector to clean up

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    try:
        if st.button("Record Voice"):
            # Create a new Cheetah ASR object before each loop
            cheetah = pvcheetah.create(access_key=PICOVOICE_API_KEY)

            st.session_state.sales_agent.step()
            agent_response = st.session_state.sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')
            st.session_state.agent_responses.append(agent_response)
            play_agent_response(agent_response)
            st.write("Transcribing from your microphone...")
            transcript = ""

            while agent_is_speaking:
                time.sleep(0.001)

            is_endpoint = False
            start_time = time.time()
            duration = 2  # Duration in seconds

            while not is_endpoint and time.time() - start_time <= duration:
                buffer = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_data = np.frombuffer(buffer, dtype=np.int16)

                # Call the transcription function on the buffered audio
                partial_transcript, is_endpoint = transcribe_audio_stream_using_cheetah(audio_data)

                if partial_transcript:
                    transcript += partial_transcript

            # Send the transcribed text after the specified duration
            st.session_state.sales_agent.human_step(transcript.strip())
            user_input = transcript.strip()
            st.session_state.history.append(user_input)

            # Reset the transcript to an empty string at the end of the loop
            transcript = ""

            # Allow the Python garbage collector to clean up the Cheetah ASR object
            cheetah = None

            if '<END_OF_CALL>' in agent_response:
                st.write('Sales Agent determined it is time to end the conversation.')
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

cheetah = pvcheetah.create(access_key=PICOVOICE_API_KEY)
main()



if st.session_state['agent_responses']:
    for i in range(len(st.session_state['agent_responses'])):
        message(st.session_state["agent_responses"][i], key=str(i))
        message(st.session_state['history'][i], is_user=True, key=str(i) + '_user')
