import os
import copy
import time
import argparse
from io import BytesIO
from dotenv import load_dotenv
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
import pygame
from pydub import AudioSegment
from speech_player.audio_generator import AudioPlayer

# load environment variables
load_dotenv(dotenv_path="configs/.env", override=True, verbose=True)

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

# Define constants for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
PICOVOICE_API_KEY = os.getenv("PICOVOICE_API_KEY")

# Play agent response using ElevenLabs TTS API
def play_mp3_chunk(filelike_object, buffer_size=4096):
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=buffer_size)
    pygame.mixer.music.load(filelike_object)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.01)

    pygame.mixer.quit()


def play_agent_response_bark(text: str, player):
    player.start(text)

# def play_agent_response(text: str, player):
#     player.start(text)
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

# Function to handle agent's response
def agent_speaks(sales_agent):
    sales_agent.step()
    agent_response = sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')

    print("Agent response:", agent_response)

    play_agent_response(agent_response)
    return agent_response

# Function to transcribe user's input from microphone
def transcribe_user_input(client):
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
            return " ".join(final_words).strip()

    return ""

# Function to transcribe audio stream using Cheetah ASR
def transcribe_audio_stream_using_cheetah(audio_chunk):
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

    # Ensure the audio_data input is int16 and one-dimensional (mono)
    assert audio_data.dtype == np.int16, f"Invalid data type: expected int16, got {audio_data.dtype}"
    assert len(audio_data.shape) == 1, f"Invalid data shape: expected one-dimensional, but got {audio_data.shape}"

    # Process audio data with Cheetah ASR
    partial_transcript, is_endpoint = cheetah.process(audio_data)

    return partial_transcript, is_endpoint


def main(model_name: str = 'soniox_microphone', duration: int = 3, sample_rate: int = 8000, audio_player_model:str='bark'):
    # Initialize the agent
    llm = ChatOpenAI(temperature=0.9)
    sales_agent = SalesGPT.from_llm(llm)
    audio_player = AudioPlayer()
    DURATION = duration
    SAMPLE_RATE = sample_rate

    if model_name == 'soniox_microphone':
        # Initialize SpeechClient and transcriber
        with SpeechClient() as client:
            count = 0
            max_num_turns = 4

            while count != max_num_turns:
                # Agent speaks
                count += 1
                sales_agent.step()
                agent_response = sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')
                if audio_player_model =='bark':
                    play_agent_response_bark(agent_response, audio_player)
                else:
                    play_agent_response(agent_response)

                # User speaks
                user_input = transcribe_user_input(client)
                if user_input:
                    sales_agent.human_step(user_input)
                    print(f"User input sent to agent: {user_input}")

                if '<END_OF_CALL>' in agent_response:
                    print('Sales Agent determined it is time to end the conversation.')
                    break

    elif model_name == 'soniox':
        with SpeechClient() as client:
            count = 0
            max_num_turns = 4

            while count != max_num_turns:
                # Agent speaks
                count += 1
                sales_agent.step()
                agent_response = sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')
                if audio_player_model =='bark':
                    play_agent_response_bark(agent_response, audio_player)
                else:
                    play_agent_response(agent_response)
                
                # User speaks
                print(f"Recording audio for {DURATION} seconds...")
                # Record audio for 4 seconds
                recorded_audio = record_audio()

                # Save the recorded audio to a file
                output_file = "recorded_audio.wav"
                sf.write(output_file, recorded_audio, SAMPLE_RATE)
                result = transcribe_file_short("recorded_audio.wav", client)
                user_input = " ".join(result.words)
                if user_input:
                    sales_agent.human_step(user_input)
                    print(f"User input sent to agent: {user_input}")

                if '<END_OF_CALL>' in agent_response:
                    print('Sales Agent determined it is time to end the conversation.')
                    break

    elif model_name == 'picovoice':
        # Create an initial Cheetah ASR object to get the frame length
        cheetah = pvcheetah.create(access_key=PICOVOICE_API_KEY)
        CHUNK_SIZE = cheetah.frame_length  # Use the frame length specified by Cheetah
        cheetah = None  # Allow the Python garbage collector to clean up

        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

        try:
            count = 0
            max_num_turns = 4

            while count != max_num_turns:
                count += 1

                # Create a new Cheetah ASR object before each loop
                cheetah = pvcheetah.create(access_key=PICOVOICE_API_KEY)

                sales_agent.step()
                agent_response = sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')
                print("Agent response:", agent_response)
                if audio_player_model =='bark':
                    play_agent_response_bark(agent_response, audio_player)
                else:
                    play_agent_response(agent_response)

                print("Transcribing from your microphone...")
                transcript = ""

                while agent_is_speaking:
                    time.sleep(0.001)

                is_endpoint = False
                start_time = time.time()
                duration = 3  # Duration in seconds

                while not is_endpoint and time.time() - start_time <= duration:
                    buffer = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    audio_data = np.frombuffer(buffer, dtype=np.int16)

                    # Call the transcription function on the buffered audio
                    partial_transcript, is_endpoint = transcribe_audio_stream_using_cheetah(audio_data)

                    if partial_transcript:
                        print(f"Transcribing: {partial_transcript}", end='\r')
                        transcript += partial_transcript

                # Send the transcribed text after the specified duration
                sales_agent.human_step(transcript.strip())
                print(f"User input sent to agent: {transcript.strip()}")

                # Reset the transcript to an empty string at the end of the loop
                transcript = ""

                # Allow the Python garbage collector to clean up the Cheetah ASR object
                cheetah = None

                if '<END_OF_CALL>' in agent_response:
                    print('Sales Agent determined it is time to end the conversation.')
                    break
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", default="soniox", help="name of the model you want to use")
    parser.add_argument("-p", "--audio_player", default="bark", help="name of the model of TTS player (elevenslabs or bark)")
    parser.add_argument("-d", "--duration", default=3, help="time in seconds for recorded audio using soniox")
    parser.add_argument("-s", "--sample_rate", default=8000, help="sample rate of input audio for soniox")
    args = parser.parse_args()

    # Run the profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Initialize Picovoice Cheetah ASR
    if args.model_name == "picovoice":
        cheetah = pvcheetah.create(access_key=PICOVOICE_API_KEY)

    # Call the function you want to profile
    main(model_name=args.model_name, duration=args.duration, sample_rate=args.sample_rate)

    # Stop the profiler
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open(f'data/code_profiling/stats_{args.model_name}.txt', 'w+') as f:
        f.write(s.getvalue())