import os, base64, argparse, glob, re
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "Molmo-7B-D-0924"
BASE_URL = "https://ai2endpoints.cirrascale.ai/api"

def create_prompt(object_name: str) -> str:
    target = object_name.strip().lower().rstrip('s')
    return f"""point_qa STRICT:

You are a visual detector. Return EXACTLY ONE <point> tag OR the string <none>.
If multiple candidates exist, pick ONE using this order: completeness > clarity/focus > centrality > size.
If you are not >=80% confident the {target} exists, output <none>.

Rules:
- Coordinates MUST mark the *visual center* of the detected {target}, not edges or corners.
- Coordinates are floating point percentages of the image width/height: 0.0–100.0
- x is % from left; y is % from top
- alt must be a terse noun phrase for the target only
- Do not include any text outside the allowed schema.

Output schema (no extra text, no explanations):
<point x="<x>" y="<y>" alt="{target}"/>  OR  <none>

Task: Identify the single best {target} in this image and output its center coordinates."""


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

# check if we found desired object
def has_coordinates(response_text):
    # look for x y patterns
    x_pattern = r'x\d*="([^"]+)"'
    y_pattern = r'y\d*="([^"]+)"'
    return bool(re.search(x_pattern, response_text) and re.search(y_pattern, response_text))

# get coords
def extract_coordinates(response_text, target_object=None):
    coordinates = []
    
    point_pattern = r'<point\s+x="([^"]+)"\s+y="([^"]+)"\s+alt="([^"]*)"[^>]*>'
    point_matches = re.findall(point_pattern, response_text)
    
    for x, y, alt_text in point_matches:
        try:
            if target_object and is_relevant_alt(alt_text, target_object):
                coordinates.append((float(x), float(y)))
            elif not target_object: 
                coordinates.append((float(x), float(y)))
        except ValueError:
            continue
    
    if not coordinates:
        x_matches = re.findall(r'x\d*="([^"]+)"', response_text)
        y_matches = re.findall(r'y\d*="([^"]+)"', response_text)
        
        for i in range(min(len(x_matches), len(y_matches))):
            try:
                x = float(x_matches[i])
                y = float(y_matches[i])
                coordinates.append((x, y))
            except ValueError:
                continue
    
    return coordinates

def is_relevant_alt(alt_text, target_object):
    alt_lower = alt_text.lower()
    target_lower = target_object.lower()
    
    if target_lower in alt_lower:
        return True
    
    # handle plurals and variations
    target_variations = [
        target_lower,
        target_lower.rstrip('s'), 
        target_lower + 's',
    ]
    
    # common synonyms/variations
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
            for synonym in values:
                if synonym in alt_lower:
                    return True
    
    # check variations
    for variation in target_variations:
        if variation in alt_lower:
            return True
    
    return False

def annotate_image_with_coordinates(image_path, coordinates, output_path):
    try:
        # open the original image
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # get image dimensions
        width, height = img.size
        
        # draw dots for each coordinate
        for i, (x, y) in enumerate(coordinates):
            # convert percentage coordinates to pixel coordinates
            pixel_x = int((x / 100.0) * width)
            pixel_y = int((y / 100.0) * height)
            
            # draw a red circle
            radius = max(5, min(width, height) // 100)  # scale dot size with image
            draw.ellipse([
                pixel_x - radius, pixel_y - radius,
                pixel_x + radius, pixel_y + radius
            ], fill='red', outline='darkred', width=2)
            
            # add coordinate text
            text = f"({x:.1f}, {y:.1f})"
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # position text slightly offset from the dot
            text_x = pixel_x + radius + 5
            text_y = pixel_y - radius - 5
            draw.text((text_x, text_y), text, fill='red', font=font)
        
        # save the annotated image
        img.save(output_path)
        return True
        
    except Exception as e:
        print(f"Error creating annotated image: {e}")
        return False

def find_kf_imgs_dir(saved_state_dir):
    kf_imgs_path = Path(saved_state_dir) / "kf-imgs"
    if kf_imgs_path.exists() and kf_imgs_path.is_dir():
        return kf_imgs_path
    return None

def get_keyframe_images(kf_imgs_dir):
    pattern = str(Path(kf_imgs_dir) / "keyframe_*.png")
    images = glob.glob(pattern)
    return sorted(images)

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
    parser.add_argument('--output', help='Output path for annotated image (default: annotated_result.png)')
    
    args = parser.parse_args()
    
    # create prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = create_prompt(args.obj)
    
    # check api key
    api_key = os.environ.get("CIRRASCALE_API_KEY")
    if not api_key:
        raise RuntimeError("Set CIRRASCALE_API_KEY in your environment.")
    
    # find kf-imgs dir
    saved_state_path = Path(args.dir)
    if not saved_state_path.exists():
        print(f"Error: Saved-state directory {saved_state_path} does not exist")
        return 1
    
    kf_imgs_dir = find_kf_imgs_dir(saved_state_path)
    if not kf_imgs_dir:
        print(f"Error: No kf-imgs directory found in {saved_state_path}")
        return 1
    
    # get all keyframe images
    keyframe_images = get_keyframe_images(kf_imgs_dir)
    if not keyframe_images:
        print(f"Error: No keyframe images found in {kf_imgs_dir}")
        return 1
    
    print(f"Found {len(keyframe_images)} keyframe images")
    print(f"Searching for: {args.obj if not args.prompt else 'custom prompt'}")
    print(f"Using prompt: '{prompt}'")
    print("Processing images until coordinates are found...\n")
    
    # create client 
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    
    # process images
    for i, image_path in enumerate(keyframe_images):
        image_name = Path(image_path).name
        print(f"[{i+1}/{len(keyframe_images)}] Processing: {image_name}")
        
        response = process_single_image(image_path, prompt, client)
        if response is None:
            continue
        
        print(f"Response: {response}")
        
        if has_coordinates(response):
            coordinates = extract_coordinates(response, args.obj)
            if coordinates: 
                print(f"\nFOUND COORDINATES in {image_name}!")
                print(f"Coordinates: {coordinates}")
                
                # save annotated image
                if args.save_annotated:
                    output_path = args.output or "annotated_result.png"
                    if annotate_image_with_coordinates(image_path, coordinates, output_path):
                        print(f"Annotated image saved to: {output_path}")
                        print(f"You can view it with: xdg-open {output_path}")
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
