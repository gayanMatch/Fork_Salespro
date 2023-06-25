from elevenlabs import generate

class TextToSpeechService:
    def __init__(self):
        pass

    def synthesize(self, text):
        audio_data = generate(text)
        return audio_data
