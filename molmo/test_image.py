import os, base64, requests
from io import BytesIO
from PIL import Image
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(
    api_key=os.environ["CIRRASCALE_API_KEY"],
    base_url="https://ai2endpoints.cirrascale.ai/api",
)

def to_data_url(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

image_data_url = to_data_url(
    "https://picsum.photos/id/237/536/354"
)

resp = client.chat.completions.create(
    model="Molmo-7B-D-0924",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "point_qa: identify the ears in this image"},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]
    }]
)

print(resp.choices[0].message.content)
