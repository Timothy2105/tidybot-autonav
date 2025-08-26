import os, base64, argparse, glob, re, sys, time
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DETECT_MODEL = os.environ.get("OPENAI_DETECT_MODEL", "gpt-4o-mini")
DEFAULT_VERIFY_MODEL = os.environ.get("OPENAI_VERIFY_MODEL", "gpt-5")
CONFIDENCE_KEY = "confidence" 

def create_detect_prompt(object_name: str) -> str:
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
- Also output a confidence score between 0.0 and 1.0 as <confidence value="<score>"/>
- Do not include any text outside the allowed schema.

Output schema (no extra text, no explanations):
<point x="<x>" y="<y>" alt="{target}"/>
<confidence value="<score>"/>
OR
<none>

Task: Identify the single best {target} in this image and output its center coordinates."""

def create_verify_prompt(object_name: str, candidate_xy, width=None, height=None) -> str:
    target = object_name.strip().lower().rstrip('s')
    size_line = ""
    if width is not None and height is not None:
        size_line = f"\nThe image is {width} px wide and {height} px tall. Coordinates are in PERCENT of width/height."

    x_c, y_c = candidate_xy
    return f"""point_qa STRICT (verification pass):

You are verifying coordinates for a {target}.{size_line}

A candidate point was proposed at:
<point x="{x_c:.4g}" y="{y_c:.4g}" alt="{target}"/>

Your job:
1) Inspect the image and decide whether this point is the correct visual CENTER of the best {target}.
2) If correct, return the SAME point.
3) If not correct, return a BETTER SINGLE point for the best {target}.
4) Only return ONE point.
5) Also return a confidence score between 0.0 and 1.0 as <confidence value="<score>"/> reflecting your belief that the returned point is correct and that the object exists.
6) If you are not >=80% confident the {target} exists, output <none>.

Rules:
- Coordinates MUST be percentages (0.0–100.0).
- x is % from left; y is % from top
- alt must be a terse noun phrase for the target only
- Do not include any text outside the allowed schema.

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

def find_kf_imgs_dir(saved_state_dir):
    kf_imgs_path = Path(saved_state_dir) / "kf-imgs"
    return kf_imgs_path if kf_imgs_path.exists() and kf_imgs_path.is_dir() else None

def get_keyframe_images(kf_imgs_dir):
    return sorted(glob.glob(str(Path(kf_imgs_dir) / "keyframe_*.png")))


POINT_RE = r'<point\s+x="([^"]+)"\s+y="([^"]+)"\s+alt="([^"]*)"[^>]*>'
CONF_RE = r'<confidence\s+value="([^"]+)"\s*/?>'

def parse_point_and_conf(text, target_object=None):
    coords = []
    conf = None

    pts = re.findall(POINT_RE, text)
    for x, y, alt in pts:
        try:
            x_f = float(x)
            y_f = float(y)
        except ValueError:
            continue
        # quick alt relevance
        if target_object:
            t = target_object.lower()
            a = alt.lower()
            if (t not in a) and (t.rstrip('s') not in a) and (t + 's' not in a):
                # allow, but lower priority; we still keep the point
                pass
        coords.append((x_f, y_f))

    m = re.search(CONF_RE, text)
    if m:
        try:
            conf = float(m.group(1))
        except ValueError:
            conf = None

    return coords, conf, bool(pts)

def has_coordinates(text):
    return "<point" in text

def annotate_image_with_percent(image_path, coordinates, output_path):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        for (x, y) in coordinates:
            px = int((x / 100.0) * w)
            py = int((y / 100.0) * h)
            radius = max(5, min(w, h) // 100)
            draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                         fill='red', outline='darkred', width=2)
            label = f"({x:.1f}%, {y:.1f}%)"
            try:
                font = ImageFont.load_default()
            except:
                font = None
            draw.text((px + radius + 5, py - radius - 5), label, fill='red', font=font)

        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error creating annotated image: {e}")
        return False

def call_vision_chat(model, prompt, image_path, client: OpenAI):
    try:
        data_url = to_data_url_from_file(image_path)
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[{model}] error: {e}")
        return None

def detect_then_verify(image_path, obj, client, detect_model, verify_model, verbose=True):
    # detect
    detect_prompt = create_detect_prompt(obj)
    det_text = call_vision_chat(detect_model, detect_prompt, image_path, client)
    if verbose and det_text is not None:
        print(f"[DETECT:{detect_model}] {det_text}")

    if not det_text:
        return None, None, None

    if "<none>" in det_text.lower():
        return None, None, None

    det_coords, det_conf, saw_point = parse_point_and_conf(det_text, obj)
    if not det_coords:
        return None, None, None

    # verify
    try:
        w, h = Image.open(image_path).size
    except:
        w = h = None

    verify_prompt = create_verify_prompt(obj, det_coords[0], width=w, height=h)
    ver_text = call_vision_chat(verify_model, verify_prompt, image_path, client)
    if verbose and ver_text is not None:
        print(f"[VERIFY:{verify_model}] {ver_text}")

    if not ver_text:
        # no verification response
        return det_coords, det_conf, "detected"

    if "<none>" in ver_text.lower():
        # verifier rejected existence
        return None, None, None

    ver_coords, ver_conf, ver_saw_point = parse_point_and_conf(ver_text, obj)
    if ver_coords:
        return ver_coords, ver_conf, "verified"

    # if verifier failed to produce a point, keep detection
    return det_coords, det_conf, "detected"

def check_and_clear_object_file():
    object_file = "calib-results/runtime/object.txt"
    try:
        if os.path.exists(object_file):
            with open(object_file, 'r') as f:
                content = f.read().strip()
                if content:  
                    # clear file
                    with open(object_file, 'w') as f:
                        f.write("")
                    return content
    except Exception as e:
        print(f"Error checking object file: {e}")
    return None

def process_object_detection(target_object, keyframe_images, client, detect_model, verify_model, save_annotated, output_path):
    for i, image_path in enumerate(keyframe_images):
        image_name = Path(image_path).name
        print(f"[{i+1}/{len(keyframe_images)}] {image_name}")

        coords, conf, stage = detect_then_verify(
            image_path=image_path,
            obj=target_object,
            client=client,
            detect_model=detect_model,
            verify_model=verify_model,
            verbose=True
        )

        if coords:
            print(f"\nFOUND COORDINATES ({stage}) in {image_name}!")
            print(f"Coordinates: {coords}")
            if conf is not None:
                print(f"{CONFIDENCE_KEY.capitalize()}: {conf:.2f}")

            if save_annotated:
                actual_output = output_path or f"annotated_result.png"
                if annotate_image_with_percent(image_path, coords, actual_output):
                    print(f"Annotated image saved to: {actual_output}")
                else:
                    print("Failed to create annotated image")
            return 0
        else:
            print("No valid coordinates for this frame, continuing...\n")

    print(f"No coordinates found for {target_object} in any keyframe images.")
    return 1

def monitor_object_file(saved_state_dir, detect_model, verify_model, save_annotated, output_path):
    print("Starting object monitoring mode...")
    print("Monitoring calib-results/runtime/object.txt every 2 seconds")
    print("Save an object name in the hand_eye_calib_viser.py GUI to start detection")
    print("Press Ctrl+C to stop monitoring\n")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")
    
    client = OpenAI(api_key=api_key)
    
    # Validate directory once
    saved_state_path = Path(saved_state_dir)
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
    
    print(f"Ready to process {len(keyframe_images)} keyframe images")
    print(f"Detect model: {detect_model}")
    print(f"Verify model: {verify_model}")
    
    try:
        while True:
            # check for new object
            target_object = check_and_clear_object_file()
            
            if target_object:
                print(f"\n{'='*50}")
                print(f"NEW OBJECT DETECTED: {target_object}")
                print(f"{'='*50}")
                print(f"Starting detection process...")
                
                # process object
                result = process_object_detection(
                    target_object=target_object,
                    keyframe_images=keyframe_images,
                    client=client,
                    detect_model=detect_model,
                    verify_model=verify_model,
                    save_annotated=save_annotated,
                    output_path=output_path
                )
                
                if result == 0:
                    print(f"Successfully found {target_object}!")
                else:
                    print(f"Could not find {target_object} in any keyframe")
                
                print(f"{'='*50}")
                print("Returning to monitoring mode...")
                print("Save another object name to continue detection\n")
            
            # wait 2 seconds
            time.sleep(1.5)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Two-stage (detect -> verify) visual point detector')
    parser.add_argument('--dir', '-d', required=True, help='Path to saved-state directory')
    parser.add_argument('--obj', '-o', help='Object to search for (e.g., "chair", "wheel", "door")')
    parser.add_argument('--monitor', '-m', action='store_true', help='Monitor object.txt file continuously for new objects')
    parser.add_argument('--save-annotated', '-s', action='store_true', help='Save annotated image with detected/verified coordinates')
    parser.add_argument('--output', help='Output path for annotated image (default: annotated_result.png)')
    parser.add_argument('--detect-model', default=DEFAULT_DETECT_MODEL, help=f'Detector model (default: {DEFAULT_DETECT_MODEL})')
    parser.add_argument('--verify-model', default=DEFAULT_VERIFY_MODEL, help=f'Verifier model (default: {DEFAULT_VERIFY_MODEL})')
    args = parser.parse_args()
    
    # monitoring mode
    if args.monitor:
        return monitor_object_file(
            saved_state_dir=args.dir,
            detect_model=args.detect_model,
            verify_model=args.verify_model,
            save_annotated=args.save_annotated,
            output_path=args.output
        )
    
    # single-object mode
    if not args.obj:
        print("Error: --obj is required when not using --monitor mode")
        print("Use --monitor to continuously watch for objects, or --obj <object_name> for single detection")
        return 1

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
    print(f"Detect model: {args.detect_model}")
    print(f"Verify model: {args.verify_model}")
    print("Processing images until coordinates are found & verified...\n")

    client = OpenAI(api_key=api_key)

    # use refactored detection
    return process_object_detection(
        target_object=args.obj,
        keyframe_images=keyframe_images,
        client=client,
        detect_model=args.detect_model,
        verify_model=args.verify_model,
        save_annotated=args.save_annotated,
        output_path=args.output
    )

if __name__ == "__main__":
    sys.exit(main())
