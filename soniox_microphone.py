from soniox.speech_service import SpeechClient
from soniox.transcribe_live import transcribe_microphone
import os 
import time 
from soniox.speech_service import SpeechClient, Result
from typing import List, Tuple

os.environ['SONIOX_API_KEY'] = "f7d0f5e9c111971168b9f9729048bc01ca16843a7fc50db9ca589d09b1c84318"

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



def main():
    final_words = []
    with SpeechClient() as client:
        print("Transcribing from your microphone...")
        for result in transcribe_microphone(client):
            new_final_words, non_final_words = split_words(result)
            final_words += new_final_words
            render_final_words(final_words)
            render_non_final_words(non_final_words)
            if len(final_words) > 2:
                print(f"User input sent to agent: {' '.join(final_words).strip()}")
                break


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    diff = end_time - start_time
    print(f"Time taken: {diff} seconds")