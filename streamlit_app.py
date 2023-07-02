import streamlit as st
import sounddevice as sd
import numpy as np
import os 
import soundfile as sf
from soniox.transcribe_file import transcribe_file_short
from soniox.speech_service import SpeechClient

os.environ['SONIOX_API_KEY'] = "f7d0f5e9c111971168b9f9729048bc01ca16843a7fc50db9ca589d09b1c84318"
SAMPLE_RATE = 8000
DURATION = 3


# Set up the Streamlit app
st.title("Sales Agent Chatbot")

# Function to record voice
def record_voice():
    duration = 4  # Set the duration of the recording (in seconds)
    fs = 44100  # Set the sample rate (you can adjust it if needed)

    # Record the voice
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until the recording is finished

    return recording

# Function to play back the recorded voice
def play_voice(recording):
    # Play the recording
    sd.play(recording)
    sd.wait()  # Wait until the playback is finished

def main():
    with SpeechClient() as client:
        result = transcribe_file_short("recorded_audio.wav", client)
        for word in result.words:
            st.write(f"{word.text} {word.start_ms} {word.duration_ms}")

# Create a button for recording voice
if st.button("Record Voice"):
    st.write("Recording started. Speak into your microphone...")
    recorded_audio = record_voice()
    st.write("Recording finished. Here is your recorded voice:")

    # Save the recorded audio to a file
    output_file = "recorded_audio.wav"
    sf.write(output_file, recorded_audio, SAMPLE_RATE)   
    main()
    # play agent voice 
    play_voice(recorded_audio)
