import os
import numpy as np
import pvcheetah
import pyaudio
import requests
from io import BytesIO
from pydub import AudioSegment
from collections import deque
from sales_gpt import SalesGPT
from langchain.chat_models import ChatOpenAI
from elevenlabs import stream as elevenlabs_stream
from typing import List, Tuple
import time
import cProfile
import io
import pstats
from scipy.io import wavfile

os.environ['OPENAI_API_KEY'] = "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"
os.environ['SONIOX_API_KEY'] = "f7d0f5e9c111971168b9f9729048bc01ca16843a7fc50db9ca589d09b1c84318"
os.environ['ELEVEN_API_KEY'] = "0ddc8db042045085b262085b0acc096a"

# Initialize Picovoice Cheetah ASR
ACCESS_KEY = "SoMf0xO/J9PWWRHb3HSTHxDwiGY0RDbPebuEoJlZE/MuIecCZuGaqQ=="
cheetah = pvcheetah.create(access_key=ACCESS_KEY)


def get_audio_devices():
    audio = pyaudio.PyAudio()
    device_count = audio.get_device_count()
    devices = []

    for i in range(device_count):
        device_info = audio.get_device_info_by_index(i)
        devices.append({"index": i, "name": device_info.get("name")})

    audio.terminate()
    return devices

audio_devices = get_audio_devices()
print("Available audio devices:")
for device in audio_devices:
    print(f"- {device['index']}: {device['name']}")

# Replace `selected_microphone_index` with the index of your desired microphone
selected_microphone_index = 1

CHANNELS = 1  # Mono
RATE = 16000  # Sample rate expected by DeepSpeech model
CHUNK_SIZE = 480  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format

audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
    input_device_index=selected_microphone_index,
)

agent_is_speaking = True

def add_silence_to_wav(wav_bytes, silence_duration=0.05):
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

    if response.status_code == 200:
        audio_data = BytesIO()
        for chunk in response.iter_content(chunk_size=512):
            if chunk:
                audio_data.write(chunk)

        audio_data.seek(0)
        audio_data = add_silence_to_wav(convert_mp3_to_wav(audio_data.read()))
        # Use the renamed `stream` function
        elevenlabs_stream(BytesIO(audio_data))
    else:
        print(f"Error streaming audio: {response.status_code} {response.text}")

    agent_is_speaking = False

import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

def transcribe_audio_stream_using_cheetah(audio_chunk):
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

    # Ensure the audio_data input is int16 and one-dimensional (mono)
    assert audio_data.dtype == np.int16, f"Invalid data type: expected int16, got {audio_data.dtype}"
    assert len(audio_data.shape) == 1, f"Invalid data shape: expected one-dimensional, but got {audio_data.shape}"

    # Process audio data with Cheetah ASR
    partial_transcript, is_endpoint = cheetah.process(audio_data)

    return partial_transcript, is_endpoint

def main():
    llm = ChatOpenAI(temperature=0.9)
    sales_agent = SalesGPT.from_llm(llm)

    # Create an initial Cheetah ASR object to get the frame length
    cheetah = pvcheetah.create(access_key=ACCESS_KEY)
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
            cheetah = pvcheetah.create(access_key=ACCESS_KEY)

            sales_agent.step()
            agent_response = sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')
            print("Agent response:", agent_response)
            play_agent_response(agent_response)

            print("Transcribing from your microphone...")
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
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('stats.txt', 'w+') as f:
        f.write(s.getvalue())