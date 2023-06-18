import os
import threading
from soniox.speech_service import SpeechClient, Result
from soniox.transcribe_live import transcribe_microphone
from sales_gpt import SalesGPT
from langchain.chat_models import ChatOpenAI
from elevenlabs import generate, play, set_api_key, stream
from typing import List, Tuple
import time
from scipy.io import wavfile
from io import BytesIO
from pydub import AudioSegment
import numpy as np
import requests
from pydub import AudioSegment
from pydub.playback import play as pydub_play
import threading

os.environ['OPENAI_API_KEY'] = "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"
os.environ['SONIOX_API_KEY'] = "f7d0f5e9c111971168b9f9729048bc01ca16843a7fc50db9ca589d09b1c84318"
os.environ['ELEVEN_API_KEY'] = "0ddc8db042045085b262085b0acc096a"

agent_is_speaking = False


def play_silent_audio():
    silent_audio = AudioSegment.silent(duration=10000)  # 10 seconds of silence
    while True:
        pydub_play(silent_audio)
        
        
def add_silence_to_wav(wav_bytes, silence_duration = 0.5):
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
        for chunk in response.iter_content(chunk_size=1024):
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

def split_words(result: Result) -> Tuple[List[str], List[str]]:
    final_words = []
    non_final_words = []

    for word in result.words:
        if word.is_final:
            final_words.append(word.text)
        else:
            non_final_words.append(word.text)
    return final_words, non_final_words


def render_final_words(words: List[str]) -> None:
    if len(words) > 0:
        line = " ".join(words)
        print(f"User input: {line}")


def render_non_final_words(words: List[str]) -> None:
    if len(words) > 0:
        line = " ".join(words)
        print(f"Transcribing: {line}", end="\r")


# def transcribe_and_send(client, sales_agent):
#     global agent_is_speaking
#     final_words = []

#     pa = pyaudio.PyAudio()
#     sample_rate = 16000
#     block_size = int(sample_rate * 0.3)

#     stream = pa.open(format=pyaudio.paInt16,
#                      channels=1,
#                      rate=sample_rate,
#                      input=True,
#                      frames_per_buffer=block_size)

#     try:
#         while True:
#             if agent_is_speaking:
#                 continue

#             captured_audio = stream.read(block_size)
#             input_audio = bytes(captured_audio)
#             result = client.transcribe_audio(input_audio, sample_rate)

#             new_final_words, non_final_words = split_words(result)
#             final_words += new_final_words
#             render_final_words(final_words)
#             render_non_final_words(non_final_words)

#             if len(final_words) > 2:
#                 user_input = " ".join(final_words).strip()
#                 sales_agent.human_step(user_input)
#                 print(f"User input sent to agent: {user_input}")
#                 break
#     finally:
#         stream.stop_stream()
#         stream.close()
#         pa.terminate()


        
def main():
    background_sound_thread = threading.Thread(target=play_silent_audio)
    background_sound_thread.setDaemon(True)
    background_sound_thread.start()

    llm = ChatOpenAI(temperature=0.9)
    sales_agent = SalesGPT.from_llm(llm)

    count = 0
    max_num_turns = 10
    non_final_word_limit = 4
    time_limit = 6

    with SpeechClient() as client:
        print("Transcribing from your microphone...")
        while count != max_num_turns:
            count += 1
            try:
                sales_agent.step()
                agent_response = sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')
                print("Agent response:", agent_response)

                agent_response_thread = threading.Thread(target=play_agent_response, args=(agent_response, ))
                agent_response_thread.start()

                agent_response_thread.join()

                start_time = time.time()
                running_non_final_words = []
                for result in transcribe_microphone(client):
                    if agent_is_speaking:
                        continue

                    new_final_words, non_final_words = split_words(result)
                    running_non_final_words += non_final_words
                    render_final_words(new_final_words)
                    render_non_final_words(non_final_words)

                    elapsed_time = time.time() - start_time
                    if len(running_non_final_words) >= non_final_word_limit or elapsed_time >= time_limit:
                        user_input = " ".join(running_non_final_words).strip()
                        sales_agent.human_step(user_input)
                        print(f"User input sent to agent: {user_input}")
                        break
            except RuntimeError as e:
                if 'cannot dereference null pointer' in str(e.args):
                    print("Restarting transcribe session due to error")
                else:
                    raise e
   
if __name__ == "__main__":
    main()