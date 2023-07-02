import streamlit as st
import sounddevice as sd
import numpy as np
import os
PICOVOICE_API_KEY = os.getenv("PICOVOICE_API_KEY")
DURATION = 3


# Set up the Streamlit app
st.title("Sales Agent Chatbot")

# Function to record voice
def record_voice():
    duration = 4 
    fs = 44100 

    # Record the voice
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait() 

    return recording

# Function to play back the recorded voice
def play_voice(recording):
    # Play the recording
    sd.play(recording)
    sd.wait() 

# Create a button for recording voice
if st.button("Record Voice"):
    st.write("Recording started. Speak into your microphone...")
    recording = record_voice()
    st.write("Recording finished. Here is your recorded voice:")
    play_voice(recording)
