import os
import openai

openai.api_key = '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'

# print(openai.api_key)

response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo-16k",
              messages=[{"role": "system", "content": 'you are an ai write'},
                        {"role": "user", "content": 'write a 20,000 word e3ssay on the moon'}
              ])

response["choices"][0]["message"]["content"]


print(response)
# response = openai.ChatCompletion.create(
#   engine="gpt-4",
#   messages="write 20,000 words on the moon",
#   # Your task or question 
#   max_tokens=16000
# )

# # This will output the answer
# generated_text = response.choices[0].text.strip()
# print(f"Generated text: {generated_text}")


