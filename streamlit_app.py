import streamlit as st
import sounddevice as sd
import numpy as np

# Set up the Streamlit app
st.title("Voice Recording App")

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

# Create a button for recording voice
if st.button("Record Voice"):
    st.write("Recording started. Speak into your microphone...")
    recording = record_voice()
    st.write("Recording finished. Here is your recorded voice:")
    play_voice(recording)
