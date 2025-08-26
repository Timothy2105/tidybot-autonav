import os, openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(
    api_key=os.environ["CIRRASCALE_API_KEY"],
    base_url="https://ai2endpoints.cirrascale.ai/api",
)

resp = client.chat.completions.create(
    model="Molmo-7B-D-0924",
    messages=[
        {"role": "user", "content": "Describe a sunset over mountains in vivid detail."},
    ],
)
print(resp.choices[0].message.content)
