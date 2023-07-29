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
import time 
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


def play_mp3_chunk(filelike_object, buffer_size=4096):
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=buffer_size)
    pygame.mixer.music.load(filelike_object)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.01)

    pygame.mixer.quit()


def play_agent_response_bark(text: str, player):
    player.start(text)


def play_agent_response(text: str, voice_id: str = "pNInz6obpgDQGcFmaJgB", model_id: str = "eleven_monolingual_v1",
                        optimize_streaming_latency: int = 1):
    start_time = time.time()
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

    if response.status_code == 200:
        audio_data = BytesIO()
        base_size = 5 * 128 * 1024 // 32
        size = base_size
        chunk_point = 0
        end_time = time.time()
        print('eleven labs Time Difference: ', end_time - start_time)
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
            play_mp3_chunk(BytesIO(data.read()), 2048)
            chunk_point += size
            size += base_size
    else:
        print(f"Error streaming audio: {response.status_code} {response.text}")

    agent_is_speaking = False


def agent_speaks(sales_agent):
    sales_agent.step()
    agent_response = sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')
    print("Agent response:", agent_response)
    play_agent_response(agent_response)
    return agent_response


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


def transcribe_audio_stream_using_cheetah(audio_chunk):
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

    assert audio_data.dtype == np.int16, f"Invalid data type: expected int16, got {audio_data.dtype}"
    assert len(audio_data.shape) == 1, f"Invalid data shape: expected one-dimensional, but got {audio_data.shape}"

    partial_transcript, is_endpoint = cheetah.process(audio_data)
    return partial_transcript, is_endpoint



import threading
import pygame

def play_mp3(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


def main(model_name: str = 'soniox_microphone', duration: int = 3, sample_rate: int = 8000, audio_player_model: str = 'bark'):
    # Initialize the agent
    llm = llm=ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo", streaming=True)
    sales_agent = SalesGPT.from_llm(llm)
    audio_player = AudioPlayer()
    DURATION = duration
    SAMPLE_RATE = sample_rate

    with SpeechClient() as client:
        count = 0
        max_num_turns = 4

        while count != max_num_turns:
            iteration_time = 0 
            # Agent speaks
            count += 1
            agent_start_time = time.time()
            sales_agent.step()
            agent_response = sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')
            agent_end_time = time.time()
            print('*'*50)
            print("Agent Response Time Difference", agent_end_time - agent_start_time)
            iteration_time += (agent_end_time - agent_start_time)
            print('*'*50)

            if model_name == 'soniox_microphone' or model_name == 'soniox':
                # User speaks
                if model_name == 'soniox':
                    print(f"Recording audio for {DURATION} seconds...")
                    # Record audio for the specified duration
                    recorded_audio = record_audio(duration)

                    # Save the recorded audio to a file
                    output_file = "recorded_audio.wav"
                    sf.write(output_file, recorded_audio, SAMPLE_RATE)
                    result = transcribe_file_short("recorded_audio.wav", client)
                    user_input = " ".join(result.words)
                else:
                    user_input = transcribe_user_input(client)

                if user_input:
                    sales_agent.human_step(user_input)
                    print(f"User input sent to agent: {user_input}")

            elif model_name == 'picovoice':
                # Create an initial Cheetah ASR object to get the frame length
                cheetah = pvcheetah.create(access_key=PICOVOICE_API_KEY)
                CHUNK_SIZE = cheetah.frame_length  # Use the frame length specified by Cheetah
                cheetah = None  # Allow the Python garbage collector to clean up

                audio = pyaudio.PyAudio()
                stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

                try:
                    if audio_player_model == 'bark':
                        play_agent_response_bark(agent_response, audio_player)
                        print('*'*50)
                    else:
                        play_agent_response(agent_response)

                    print("Transcribing from your microphone...")
                    transcript = ""

                    while agent_is_speaking:
                        time.sleep(0.001)

                    is_endpoint = False
                    start_time = time.time()
                    pico_start_time = time.time()
                    duration = 3  # Duration in seconds
                    # Create and start the thread
                    hello_thread = threading.Thread(target=play_mp3, args=('data/hmmmm.mp3',))
                    flag = 0 
                    while not is_endpoint and time.time() - start_time <= duration:
                        buffer = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        audio_data = np.frombuffer(buffer, dtype=np.int16)

                        # Call the transcription function on the buffered audio
                        partial_transcript, is_endpoint = transcribe_audio_stream_using_cheetah(audio_data)

                        if partial_transcript:
                            print(f"Transcribing: {partial_transcript}", end='\r')
                            transcript += partial_transcript
                        if (time.time()-start_time)>2.5 and flag == 0:
                            hello_thread.start()
                            hello_thread.join()
                            flag = 1
                    pico_end_time = time.time()
                    print('Picovoice Time Difference: ', pico_end_time - pico_start_time)
                    iteration_time += (pico_end_time - pico_start_time)
                    print('*'*50)
                    # Send the transcribed text after the specified duration
                    sales_agent.human_step(transcript.strip())

                    # Reset the transcript to an empty string at the end of the loop
                    transcript = ""

                    # Allow the Python garbage collector to clean up the Cheetah ASR object
                    cheetah = None
                finally:
                    stream.stop_stream()
                    stream.close()
                    audio.terminate()
            print('x'*50)
            print('Iteration Time: ', iteration_time)
            print('x'*50)

            if '<END_OF_CALL>' in agent_response:
                print('Sales Agent determined it is time to end the conversation.')
                break


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
    main(model_name=args.model_name, duration=args.duration, sample_rate=args.sample_rate, audio_player_model=args.audio_player)

    # Stop the profiler
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open(f'data/code_profiling/stats_{args.model_name}.txt', 'w+') as f:
        f.write(s.getvalue())