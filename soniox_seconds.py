import sounddevice as sd
import soundfile as sf
from soniox.transcribe_file import transcribe_file_short
from soniox.speech_service import SpeechClient
import os 
import time 

os.environ['SONIOX_API_KEY'] = "f7d0f5e9c111971168b9f9729048bc01ca16843a7fc50db9ca589d09b1c84318"
SAMPLE_RATE = 8000
DURATION = 3

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    channels = 1  # Number of audio channels (mono)
    # Record audio
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()  # Wait for the recording to complete
    return audio



def main():
    with SpeechClient() as client:
        result = transcribe_file_short("recorded_audio.wav", client)
        for word in result.words:
            print(f"{word.text} {word.start_ms} {word.duration_ms}")


if __name__ == "__main__":
    start_time = time.time()
    print(f"Recording audio for {DURATION} seconds...")
    # Record audio for 4 seconds
    recorded_audio = record_audio()

    # Save the recorded audio to a file
    output_file = "recorded_audio.wav"
    sf.write(output_file, recorded_audio, SAMPLE_RATE)
    print("Recording saved to", output_file)

    main()
    end_time = time.time()
    diff = end_time - start_time
    print(f"Time taken: {diff} seconds")