import os
import asyncio
import aiohttp
import json
from typing import List, Tuple
from soniox.speech_service import SpeechClient, Result
from soniox.transcribe_live import transcribe_microphone
from sales_gpt import SalesGPT
from langchain.chat_models import ChatOpenAI
from elevenlabs import generate, play, set_api_key, stream

os.environ['OPENAI_API_KEY'] = "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"
os.environ['SONIOX_API_KEY'] = "f7d0f5e9c111971168b9f9729048bc01ca16843a7fc50db9ca589d09b1c84318"
os.environ['ELEVEN_API_KEY'] = "0ddc8db042045085b262085b0acc096a"


async def transcribe(client):
    async for result in transcribe_microphone(client):
        return result


async def process_audio(voice: str = "Adam", model: str = "eleven_monolingual_v1"):
    async with aiohttp.ClientSession() as session:
        response = {}
        try:
            response = await generate(session)
        except Exception as e:
            print(f"Error: {e}")
        audio_stream = response.get("audio_stream", None)
        if audio_stream:
            stream(audio_stream)
        else:
            print("No audio stream received")


async def call_gpt():
    llm = await ChatOpenAI(temperature=0.9)
    sales_agent = SalesGPT.from_llm(llm)
    return sales_agent


async def main():
    # Initialize conversation
    sales_agent = await call_gpt()

    # Initialize SpeechClient and transcriber
    async with SpeechClient() as client:
        count = 0
        max_num_turns = 4

        while count != max_num_turns:
            # Agent speaks
            count += 1
            sales_agent.step()
            agent_response = sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')
            print("Agent response:", agent_response)

            # Generate and play agent response
            await process_audio()

            if '<END_OF_CALL>' in agent_response:
                print('Sales Agent determined it is time to end the conversation.')
                break

            # User speaks
            print("Transcribing from your microphone...")

            final_words = []

            try:
                result = await transcribe(client)
                # Split words into final words and non-final words.
                new_final_words, non_final_words = split_words(result)

                # Render final words in last line.
                final_words += new_final_words
                render_final_words(final_words)

                # Render non-final words.
                render_non_final_words(non_final_words)

                if len(new_final_words) > 0:
                    break
            except RuntimeError as e:
                print(f"Caught runtime error: {e}, continuing with transcription...")

            word_count = len([word for word in final_words if word not in [',', ' ']])

            if word_count < 2:
                print(f"No valid input received: {final_words}, word count: {word_count}")
                continue

            sales_agent.human_step(" ".join(final_words).strip())
            print(f"User input sent to agent: {' '.join(final_words).strip()}")

            print('=' * 10)


if __name__ == "__main__":
    asyncio.run(main())