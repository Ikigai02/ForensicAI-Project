import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

# --- Configuration ---
RAW_DATA_DIR = "dataset/raw"
PROCESSED_DATA_DIR = "dataset/processed"
IMG_SIZE = (128, 128)
ELA_QUALITY = 75       # LOCKED: Must match app.py, batch_analyze.py, everywhere

for category in ["authentic", "tampered"]:
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, category), exist_ok=True)

def run_ela(image_path, quality):
    original = Image.open(image_path).convert('RGB')
    
    # --- SMART CROP: Matches app.py logic exactly ---
    w, h = original.size
    crop_size = min(w, int(h * 0.7)) 
    left, top, right, bottom = 0, h - crop_size, w, h
    working = original.crop((left, top, right, bottom))
    
    # --- ELA Logic ---
    tmp_ela_path = "tmp_ela.jpg"
    working.save(tmp_ela_path, 'JPEG', quality=quality)
    resaved = Image.open(tmp_ela_path)
    
    ela_image_raw = ImageChops.difference(working, resaved)
    
    extrema = ela_image_raw.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    ela_image = ImageEnhance.Brightness(ela_image_raw).enhance(255.0 / max_diff)
    
    if os.path.exists(tmp_ela_path):
        os.remove(tmp_ela_path)
        
    return ela_image.resize(IMG_SIZE, resample=Image.LANCZOS)

def process_dataset():
    categories = ["authentic", "tampered"]
    total_processed = 0
    errors = 0

    print("🛠️ Starting Data Cleaning & ELA Preprocessing...")
    print(f"   ELA Quality: {ELA_QUALITY}")

    for category in categories:
        folder_path = os.path.join(RAW_DATA_DIR, category)
        save_path = os.path.join(PROCESSED_DATA_DIR, category)
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Processing {len(files)} images in '{category}' folder...")

        for filename in files:
            img_path = os.path.join(folder_path, filename)
            try:
                processed_img = run_ela(img_path, ELA_QUALITY)
                processed_img.save(os.path.join(save_path, filename))
                total_processed += 1
            except Exception as e:
                print(f"❌ Error with {filename}: {e}")
                errors += 1

    print(f"✅ Success! {total_processed} processed.")

if __name__ == "__main__":
    process_dataset()