import os
import threading
import signal
import sys
from typing import List, Tuple
from soniox.speech_service import SpeechClient, Result
from sales_gpt import SalesGPT
from langchain.chat_models import ChatOpenAI
from elevenlabs import generate, play, set_api_key, stream

os.environ['OPENAI_API_KEY'] = "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"
os.environ['SONIOX_API_KEY'] = "f7d0f5e9c111971168b9f9729048bc01ca16843a7fc50db9ca589d09b1c84318"
os.environ['ELEVEN_API_KEY'] = "0ddc8db042045085b262085b0acc096a"

def play_agent_response(text: str, voice: str = "Adam", model: str = "eleven_monolingual_v1"):
    print(f"Playing agent response: {text}")
    audio_stream = generate(text=text, voice=voice, model=model, stream=True)
    stream(audio_stream)


def split_words(result: Result) -> Tuple[List[str], List[str]]:
    final_words = []
    non_final_words = []
    for word in result.words:
        if word.is_final:
            final_words.append(word.text)
        else:
            non_final_words.append(word.text)
    return final_words, non_final_words


def render_final_words(words: List[str]) -> List[str]:
    if len(words) > 0:
        line = " ".join(words)
        print(f"User input: {line}")
    return words


def render_non_final_words(words: List[str]) -> None:
    if len(words) > 0:
        line = " ".join(words)
        print(f"Transcribing: {line}", end="\r")


def main():
    # Initialize conversation
    llm = ChatOpenAI(temperature=0.9)
    sales_agent = SalesGPT.from_llm(llm)

    stop_event = threading.Event()

    def sigint_handler(sig, stack):
        print("Interrupted, finishing transcription...")
        stop_event.set()

    signal.signal(signal.SIGINT, sigint_handler)

    with SpeechClient() as client:
        count = 0
        max_num_turns = 4
        tts_enabled = True

        while count != max_num_turns:
            # Agent speaks
            count += 1
            sales_agent.step()
            agent_response = sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')
            print("Agent response:", agent_response)

            if tts_enabled:
                play_agent_response(agent_response)

            if '<END_OF_CALL>' in agent_response:
                print('Sales Agent determined it is time to end the conversation.')
                break

            # User speaks
            print("Transcribing from your microphone...")

            final_words = []

            for result in transcribe_microphone(client, stop_event=stop_event):
                # Split words into final words and non-final words.
                new_final_words, non_final_words = split_words(result)

                # Render final words in last line.
                final_words += new_final_words
                render_final_words(final_words)

                # Render non-final words.
                render_non_final_words(non_final_words)

                if len(new_final_words) > 0:
                    break

            word_count = len([word for word in final_words if word not in [',', ' ']])

            if word_count < 2:
                print(f"No valid input received: {final_words}, word count: {word_count}")
                continue

            sales_agent.human_step(" ".join(final_words).strip())
            print(f"User input sent to agent: {' '.join(final_words).strip()}")

            print('=' * 10)


if __name__ == "__main__":
    main()