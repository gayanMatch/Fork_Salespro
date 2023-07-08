import requests
import time
import sounddevice as sd
import soundfile as sf
import threading


# url = f"https://209.20.159.176:5000/synthesize"
ip = '209.20.159.176'
class AudioPlayer:
    def __init__(self):
        self.wav_array = []
        self.sample_rate = 24000
        self.channels = 1

    def generate_audio(self, text="Hi Bob, it's Scott Anderson from American Hartford gold, how are you today?"):
        url = f"http://{ip}:7070/synthesize"
        data = {
            "text": text,
        }
        response = requests.post(url, data=data)
        if response.status_code == 200:
            print("Successfully sended post")
        else:
            print("Error")
        index = 0
        temp_path = '~temp.wav'
        while True:
            url = f"http://{ip}:7070/audio/audio_{index}.wav"
            response = requests.get(url)
            if response.status_code == 404:
                break
            temp_file = open(temp_path, 'wb')
            temp_file.write(response.content)
            temp_file.close()
            print(index)
            wavdata, _ = sf.read(temp_path, dtype='float32')
            self.wav_array.append(wavdata)
            index += 1
        print("Done")

    def play_audio(self):
        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels
        )
        stream.start()
        chunk_size = 1024
        while True:
            while self.wav_array:
                wav_data = self.wav_array.pop(0)
                for i in range(0, len(wav_data), chunk_size):
                    stream.write(wav_data[i:i + chunk_size])
            if self.generator.is_alive():
                time.sleep(0.01)
            else:
                break

    def start(self, text):
        self.generator = threading.Thread(target=self.generate_audio, args=(text,))
        self.generator.start()
        self.play_audio()


if __name__ == '__main__':
    player = AudioPlayer()
    player.start("Hi Bob, it's Scott Anderson from American Hartford Smith, how are you today?")
    player.start("""How often do we search for guidance and wisdom in conquering our own personal battles?
If we turn to the teachings of an ancient military strategist, we may find valuable advice applicable beyond the confines of war.
Once you begin to analyze the core tenets of this tactical masterpiece, you'll realize the vast pool of life lessons residing within it.""")
