from soniox.speech_service import Result
from typing import List, Tuple
from scipy.io import wavfile
from io import BytesIO
from pydub import AudioSegment
import numpy as np
from pydub import AudioSegment
import sounddevice as sd

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


# soniox seconds 
def record_audio(duration=3, sample_rate=8000):
    channels = 1  # Number of audio channels (mono)
    # Record audio
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()  # Wait for the recording to complete
    return audio