import speech_recognition as sr

def transcribe_from_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Transcribing...")
        text = recognizer.recognize_google(audio, key="AIzaSyA7zF-z249Bozd6tw0VBB5qusUDwXF11mg")
        print(text)
    except sr.UnknownValueError:
        print("Sorry, I didn't understand what you said.")
    except sr.RequestError as e:
        print("Error: {}.".format(e))

if __name__ == "__main__":
    transcribe_from_microphone()

