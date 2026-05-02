import os
import requests
import json
import zipfile
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Configuration
NUM_IMAGES = 5000  # Total number of images to download (adjustable)
DATA_DIR = "data/coco"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

def setup():
    """
    Downloads a subset of COCO dataset and resizes images to 256x256.
    Splits data into train and validation sets.
    """
    os.makedirs(f"{DATA_DIR}/train", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/val", exist_ok=True)
    
    # 1. Download annotations (to get image URLs)
    if not os.path.exists("data/instances_val2017.json"):
        print("Downloading metadata (annotations)...")
        response = requests.get(ANNOTATIONS_URL)
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            # Extract specific annotation file
            z.extract("annotations/instances_val2017.json", "data")
        
        # Cleanup folder structure
        if os.path.exists("data/annotations/instances_val2017.json"):
            os.rename("data/annotations/instances_val2017.json", "data/instances_val2017.json")
            try:
                os.rmdir("data/annotations")
            except:
                pass

    # 2. Parse image links and metadata
    with open("data/instances_val2017.json") as f:
        data = json.load(f)
    
    images = data['images'][:NUM_IMAGES]
    print(f"Downloading and resizing {len(images)} images to {DATA_DIR}...")

    success_count = 0
    for i, img in enumerate(tqdm(images)):
        # Split: 80% train, 20% validation
        folder = "train" if i < (NUM_IMAGES * 0.8) else "val"
        path = os.path.join(DATA_DIR, folder, img['file_name'])
        
        if not os.path.exists(path):
            try:
                response = requests.get(img['coco_url'], timeout=10)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    # Force 256x256 resolution for training consistency
                    image = image.resize((256, 256), Image.LANCZOS)
                    image.save(path, "JPEG", quality=95)
                    success_count += 1
            except Exception:
                # Skip failed downloads silently to keep tqdm clear
                continue
        else:
            success_count += 1

    print(f"\nDone! {success_count} images ready in {DATA_DIR}")

if __name__ == "__main__":
    setup()
