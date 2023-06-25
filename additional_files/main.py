from flask import Flask, request, jsonify
from speech_recognition_service import SpeechRecognitionService
from ai_salesman import AISalesman
from text_to_speech_service import TextToSpeechService

app = Flask(__name__)

speech_recognition_service = SpeechRecognitionService()
ai_salesman = AISalesman()
text_to_speech_service = TextToSpeechService()

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_data = request.files['audio']
    text = speech_recognition_service.transcribe(audio_data)
    return jsonify({'text': text})

@app.route('/response', methods=['POST'])
def response():
    user_input = request.form['text']
    ai_response = ai_salesman.get_response(user_input)
    return jsonify({'response': ai_response})

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.form['text']
    audio_data = text_to_speech_service.synthesize(text)
    return jsonify({'audio': audio_data})

if __name__ == '__main__':
    app.run()
