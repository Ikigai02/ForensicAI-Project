# fix_weak_banks.py
import os
import random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# Banks with high miss rates from your batch analysis
TARGET_BANKS = ['BSN', 'Maybank', 'PublicBank', 'RHB', 'HLB']
AUTHENTIC_DIR = "dataset/raw/authentic"
TAMPERED_DIR = "dataset/raw/tampered"

# Ensure tampered directory exists
os.makedirs(TAMPERED_DIR, exist_ok=True)

def add_strong_tampering(image_path, output_path):
    """
    Add MUCH stronger tampering artifacts to help the model learn.
    """
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Always use 2-4 methods (more aggressive)
    methods = random.sample([
        'white_box', 
        'blur_region', 
        'color_shift', 
        'text_overlay'
    ], k=random.randint(2, 4))

    if 'white_box' in methods:
        # Larger, more obvious white box
        x = random.randint(int(w*0.15), int(w*0.45))
        y = random.randint(int(h*0.25), int(h*0.55))
        box_w = random.randint(100, 200)
        box_h = random.randint(40, 70)
        draw.rectangle([(x, y), (x+box_w, y+box_h)], fill=(255, 255, 255))
        # Add text with slightly different font color
        draw.text((x+15, y+15), f"RM {random.randint(100, 9999)}", fill=(20,20,20))

    if 'blur_region' in methods:
        x = random.randint(30, w-180)
        y = random.randint(30, h-120)
        box = (x, y, x+random.randint(120, 250), y+random.randint(50, 100))
        region = img.crop(box)
        region = region.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 4.0)))
        img.paste(region, box)

    if 'color_shift' in methods:
        img_np = np.array(img)
        x = random.randint(30, w-180)
        y = random.randint(30, h-120)
        roi = img_np[y:y+80, x:x+150].copy()
        roi = roi.astype(np.float32)
        roi[:,:,0] = np.clip(roi[:,:,0] * random.uniform(1.1, 1.5), 0, 255)
        roi[:,:,1] = np.clip(roi[:,:,1] * random.uniform(0.6, 0.9), 0, 255)
        roi = roi.astype(np.uint8)
        img_np[y:y+80, x:x+150] = roi
        img = Image.fromarray(img_np)

    if 'text_overlay' in methods:
        x = random.randint(int(w*0.05), int(w*0.30))
        y = random.randint(int(h*0.05), int(h*0.20))
        draw.text((x, y), "EDITED", fill=(50,50,50), font=None)

    # Save with slightly lower quality to create compression artifacts
    quality = random.randint(60, 85)
    img.save(output_path, 'JPEG', quality=quality)

def fix_weak_banks():
    """
    Generate 30 new strong tampered samples for each weak bank.
    """
    print("🔧 FIXING WEAK BANKS")
    print("="*50)
    
    for bank in TARGET_BANKS:
        # Find authentic samples for this bank
        auth_files = [f for f in os.listdir(AUTHENTIC_DIR) 
                     if bank.lower() in f.lower() and f.endswith('.jpg')]
        
        if not auth_files:
            print(f"⚠️ No authentic files found for {bank} – skipping.")
            continue
            
        print(f"\n📝 Generating 30 strong tampered samples for {bank}...")
        
        generated = 0
        for i in range(30):
            src = random.choice(auth_files)
            src_path = os.path.join(AUTHENTIC_DIR, src)
            out_name = f"{bank}_strong_tamper_{i+1:04d}.jpg"
            out_path = os.path.join(TAMPERED_DIR, out_name)
            
            try:
                add_strong_tampering(src_path, out_path)
                generated += 1
                print(f"  ✅ {out_name}")
            except Exception as e:
                print(f"  ❌ Error on {src}: {e}")
        
        print(f"  ✅ Generated {generated}/30 for {bank}")
    
    print("\n" + "="*50)
    print(f"✅ Done! Generated strong tampered samples for weak banks.")
    print(f"📁 Added to: {TAMPERED_DIR}")
    print("\n💡 Next steps:")
    print("   1. python 2_preprocess_ela.py")
    print("   2. python 3_split_data.py")
    print("   3. python 4_train_model.py")
    print("   4. python batch_analyze.py (to see improvement)")

if __name__ == "__main__":
    fix_weak_banks()