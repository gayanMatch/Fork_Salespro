import os
from elevenlabs import generate, save

api_key = '0ddc8db042045085b262085b0acc096a'
voice = 'Adam'
model = 'eleven_monolingual_v1'
output_folder = 'generated_clips'

phrases = [
    'Mhmm', 'I see what you\'re saying', 'Absolutely', 'Sure', 'Alright', 'Okay',
    'Understood', 'I get it', 'I hear you', 'I\'m following you', 'I\'m on board',
    'Makes sense', 'I\'m tracking', 'I\'m with you', 'That\'s right', 'No doubt',
    'Exactly', 'Precisely', 'You\'ve got it', 'Without a doubt', 'Uh-huh', 'Um',
    'Like', 'Well', 'You know', 'So', 'I mean', 'Anyway', 'Basically', 'Actually',
    'Kind of', 'Sort of', 'Really', 'Honestly', 'Right', 'Okay', 'OK', 'Sure',
    'Well, you see', 'In a way'
]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i, phrase in enumerate(phrases):
    print(f'Generating audio for phrase {i + 1}/{len(phrases)}: {phrase}')
    audio = generate(text=phrase, voice=voice, model=model, api_key=api_key)
    save(audio, f'{output_folder}/{i + 1:02d}_{voice}_{phrase}.wav')