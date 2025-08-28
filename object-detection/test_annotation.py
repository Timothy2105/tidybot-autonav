import os, base64, argparse, glob, re
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "gpt-4o"  # Vision-capable GPT model

def create_prompt(object_name: str, width: int, height: int) -> str:
    target = object_name.strip().lower().rstrip('s')
    return f"""point_qa STRICT:

The image is {width} pixels wide and {height} pixels tall.

You are a visual detector. Return EXACTLY ONE <point> tag OR the string <none>.
If multiple candidates exist, pick ONE using this order: completeness > clarity/focus > centrality > size.
Also output a confidence score between 0.0 and 1.0 indicating how confident you are in this detection.

Rules:
- Coordinates MUST be in absolute pixels within the image, not percentages.
- Output format for point: <point x="<x>" y="<y>" alt="{target}"/>
- Output format for confidence: <confidence value="<score>"/>
- x is pixels from the left; y is pixels from the top.
- alt must be a terse noun phrase for the target only.
- If you are not >=80% confident the {target} exists, output <none>.

Output schema (no extra text, no explanations):
<point x="<x>" y="<y>" alt="{target}"/>
<confidence value="<score>"/>
OR
<none>"""

def to_data_url_from_file(path: str, max_side=1280, jpeg_quality=90) -> str:
    img = Image.open(path)
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

def has_coordinates(response_text):
    return "<point" in response_text

def extract_coordinates(response_text, target_object=None):
    coordinates = []
    confidence = None

    point_pattern = r'<point\s+x="([^"]+)"\s+y="([^"]+)"\s+alt="([^"]*)"[^>]*>'
    point_matches = re.findall(point_pattern, response_text)

    for x, y, alt_text in point_matches:
        try:
            if target_object and is_relevant_alt(alt_text, target_object):
                coordinates.append((int(float(x)), int(float(y))))
            elif not target_object:
                coordinates.append((int(float(x)), int(float(y))))
        except ValueError:
            continue

    conf_match = re.search(r'<confidence\s+value="([^"]+)"\s*/?>', response_text)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except ValueError:
            confidence = None

    return coordinates, confidence

def is_relevant_alt(alt_text, target_object):
    alt_lower = alt_text.lower()
    target_lower = target_object.lower()
    if target_lower in alt_lower:
        return True

    target_variations = [target_lower, target_lower.rstrip('s'), target_lower + 's']
    synonyms = {
        'chair': ['seat', 'chair', 'stool'],
        'wheel': ['wheel', 'tire', 'rim'],
        'door': ['door', 'entrance', 'doorway'],
        'window': ['window', 'glass'],
        'table': ['table', 'desk'],
        'computer': ['computer', 'pc', 'laptop', 'monitor', 'screen'],
        'phone': ['phone', 'telephone', 'mobile'],
    }

    for key, values in synonyms.items():
        if target_lower in values:
            if any(s in alt_lower for s in values):
                return True

    return any(v in alt_lower for v in target_variations)

def annotate_image_with_coordinates_local(image_path, coordinates, output_path):
    """Local PIL-based annotation"""
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for (x, y) in coordinates:
            radius = max(5, min(img.size) // 100)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         fill='red', outline='darkred', width=2)
            text = f"({x}, {y})"
            try:
                font = ImageFont.load_default()
            except:
                font = None
            draw.text((x + radius + 5, y - radius - 5), text, fill='red', font=font)

        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error creating annotated image locally: {e}")
        return False

def annotate_image_with_coordinates_gpt(image_path, coordinates, output_path, client):
    """GPT-generated annotation"""
    if not coordinates:
        return False

    x, y = coordinates[0]
    image_data_url = to_data_url_from_file(image_path)

    try:
        prompt = f"Draw a red circle of radius 12px at pixel ({x},{y}) on this image."
        resp = client.images.generate(
            model="gpt-4o",
            prompt=prompt,
            image=[{"url": image_data_url}]
        )

        # Extract image (base64 or URL)
        img_b64 = resp.data[0].b64_json
        img_bytes = base64.b64decode(img_b64)

        with open(output_path, "wb") as f:
            f.write(img_bytes)

        return True
    except Exception as e:
        print(f"Error creating GPT-annotated image: {e}")
        return False

def find_kf_imgs_dir(saved_state_dir):
    kf_imgs_path = Path(saved_state_dir) / "kf-imgs"
    return kf_imgs_path if kf_imgs_path.exists() and kf_imgs_path.is_dir() else None

def get_keyframe_images(kf_imgs_dir):
    return sorted(glob.glob(str(Path(kf_imgs_dir) / "keyframe_*.png")))

def process_single_image(image_path, prompt, client):
    try:
        image_data_url = to_data_url_from_file(image_path)
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }],
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Search for objects in keyframe images until coordinates are found')
    parser.add_argument('--dir', '-d', required=True, help='Path to saved-state directory')
    parser.add_argument('--obj', '-o', required=True, help='Object to search for (e.g., "chair", "wheels", "door")')
    parser.add_argument('--prompt', '-p', help='Custom prompt (overrides --obj if provided)')
    parser.add_argument('--save-annotated', '-s', action='store_true', help='Save annotated image with detected coordinates')
    parser.add_argument('--use-gpt-annotation', action='store_true', help='Use GPT to generate annotated image instead of local drawing')
    parser.add_argument('--output', help='Output path for annotated image (default: annotated_result.png)')

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")

    saved_state_path = Path(args.dir)
    if not saved_state_path.exists():
        print(f"Error: Saved-state directory {saved_state_path} does not exist")
        return 1

    kf_imgs_dir = find_kf_imgs_dir(saved_state_path)
    if not kf_imgs_dir:
        print(f"Error: No kf-imgs directory found in {saved_state_path}")
        return 1

    keyframe_images = get_keyframe_images(kf_imgs_dir)
    if not keyframe_images:
        print(f"Error: No keyframe images found in {kf_imgs_dir}")
        return 1

    print(f"Found {len(keyframe_images)} keyframe images")
    print(f"Searching for: {args.obj if not args.prompt else 'custom prompt'}")
    print("Processing images until coordinates are found...\n")

    client = OpenAI(api_key=api_key)

    for i, image_path in enumerate(keyframe_images):
        img = Image.open(image_path)
        width, height = img.size
        prompt = args.prompt if args.prompt else create_prompt(args.obj, width, height)
        image_name = Path(image_path).name

        print(f"[{i+1}/{len(keyframe_images)}] Processing: {image_name} ({width}x{height})")
        response = process_single_image(image_path, prompt, client)
        if response is None:
            continue

        print(f"Response: {response}")

        if has_coordinates(response):
            coordinates, confidence = extract_coordinates(response, args.obj)
            if coordinates:
                print(f"\nFOUND COORDINATES in {image_name}!")
                print(f"Coordinates: {coordinates}")
                if confidence is not None:
                    print(f"Confidence: {confidence:.2f}")

                if args.save_annotated:
                    output_path = args.output or "annotated_result.png"
                    if args.use_gpt_annotation:
                        success = annotate_image_with_coordinates_gpt(image_path, coordinates, output_path, client)
                    else:
                        success = annotate_image_with_coordinates_local(image_path, coordinates, output_path)

                    if success:
                        print(f"Annotated image saved to: {output_path}")
                    else:
                        print("Failed to create annotated image")

                return 0
            else:
                print("Found coordinates but none were relevant to the target object, continuing...")
        else:
            print("No coordinates found, continuing...\n")

    print("No coordinates found in any keyframe images.")
    return 1

if __name__ == "__main__":
    main()
