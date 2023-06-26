import os
import threading
from soniox.speech_service import SpeechClient, Result
from soniox.transcribe_live import transcribe_microphone
from sales_gpt import SalesGPT
from langchain.chat_models import ChatOpenAI
from elevenlabs import generate, play, set_api_key, stream
from typing import List, Tuple
import time

os.environ['OPENAI_API_KEY'] = "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"
os.environ['SONIOX_API_KEY'] = "f7d0f5e9c111971168b9f9729048bc01ca16843a7fc50db9ca589d09b1c84318"
os.environ['ELEVEN_API_KEY'] = "0ddc8db042045085b262085b0acc096a"

agent_is_speaking = False


def play_agent_response(text: str, voice: str = "Adam", model: str = "eleven_monolingual_v1"):
    global agent_is_speaking
    print(f"Playing agent response: {text}")
    agent_is_speaking = True
    audio_stream = generate(text=text, voice=voice, model=model, stream=True)
    stream(audio_stream)
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


def main():
    # Initialize conversation
    llm = ChatOpenAI(temperature=0.9)
    sales_agent = SalesGPT.from_llm(llm)

    # Initialize SpeechClient and transcriber
    with SpeechClient() as client:

        count = 0
        max_num_turns = 4

        while count != max_num_turns:
            # Agent speaks
            count += 1
            sales_agent.step()
            agent_response = sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')
            print("Agent response:", agent_response)

            agent_response_thread = threading.Thread(target=play_agent_response, args=(agent_response, ))
            agent_response_thread.start()

            # User speaks
            final_words = []
            print("Transcribing from your microphone...")

            while agent_is_speaking:
                time.sleep(0.01)

            for result in transcribe_microphone(client):
                new_final_words, non_final_words = split_words(result)
                final_words += new_final_words
                render_final_words(final_words)
                render_non_final_words(non_final_words)

                if len(final_words) > 2:
                    sales_agent.human_step(" ".join(final_words).strip())
                    print(f"User input sent to agent: {' '.join(final_words).strip()}")
                    break

            if '<END_OF_CALL>' in agent_response:
                print('Sales Agent determined it is time to end the conversation.')
                break

        agent_response_thread.join()


if __name__ == "__main__":
    main()