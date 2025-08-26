import os, base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

IMAGE_PATH = "saved-states/refined-room-small/kf-imgs/keyframe_000001_frame000011.png"
PROMPT = "point_qa: identify the wheels in this image and point to them"

MODEL_ID = "Molmo-7B-D-0924"
BASE_URL = "https://ai2endpoints.cirrascale.ai/api"

def to_data_url_from_file(path: str, max_side=1280, jpeg_quality=90) -> str:
    img = Image.open(path)
    # fix transparency and non-RGB
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, (0, 0), img)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((int(w * s), int(h * s)))

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/jpeg;base64," + b64

def main():
    api_key = os.environ.get("CIRRASCALE_API_KEY")
    if not api_key:
        raise RuntimeError("Set CIRRASCALE_API_KEY in your environment.")

    image_data_url = to_data_url_from_file(IMAGE_PATH)

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }],
    )
    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()
