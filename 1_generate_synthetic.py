import random 
import os
import io
import numpy as np
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --- Section 1: Folder Setup ---
TEMPLATES_DIR = "blank_templates"
OUTPUT_DIR = "dataset/raw/tampered"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Section 2: Randomization Engine ---
COMPANY_NAMES = [
    "MEGA SUPPLY", "KRIL SYSTEMS", "MOHD AZIZ ENTERPRISE", "TEGUH BERSATU", 
    "BINA JAYA SDN BHD", "MAJU LOGISTICS", "PUNCAK GLOBAL", "SINAR RESOURCES",
    "AZMAN HARDWARE", "KITA TRADING", "APEX DYNAMICS", "NUSA SENTRAL"
]

def get_random_date():
    """Generates a random datetime object within a 2-year range (2024-2026)."""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 12, 31)
    random_days = random.randrange((end_date - start_date).days)
    random_seconds = random.randrange(24 * 60 * 60)
    return start_date + timedelta(days=random_days, seconds=random_seconds)

def get_random_amount():
    """Generates a random float amount between 50.00 and 5000.00."""
    return f"{random.uniform(50.0, 5000.0):.2f}"

def get_scaled_font(img_width, size_ratio=0.035, is_bold=False):
    """Dynamically scales font based on image width to prevent pixelation."""
    size = int(img_width * size_ratio)
    font_path = "arialbd.ttf" if is_bold else "arial.ttf"
    try:
        return ImageFont.truetype(font_path, size)
    except:
        return ImageFont.load_default()

# =============================================================
# THE KEY FIX: Realistic degradation pipeline
# =============================================================
# This simulates the real lifecycle of a tampered receipt:
#   1. Fraudster edits the image (Pillow draw - already done above)
#   2. Fraudster saves it (first JPEG compression)
#   3. Fraudster screenshots or re-saves (second compression)
#   4. Image gets sent via WhatsApp/Telegram (resize + recompress)
#   5. Random noise from phone camera/screen capture
# =============================================================

def save_tampered_image(img, output_path):
    """
    Simulates realistic image degradation chain that a tampered receipt
    goes through before reaching the forensic system.
    """
    w, h = img.size
    
    # --- Stage 1: First save by the fraudster (editing app export) ---
    buf1 = io.BytesIO()
    first_quality = random.randint(80, 95)
    img.save(buf1, "JPEG", quality=first_quality)
    buf1.seek(0)
    img = Image.open(buf1).convert("RGB")
    
    # --- Stage 2: WhatsApp/Telegram simulation (resize down then back up) ---
    # These apps resize images to ~1280px max width and recompress at ~70-80%
    if random.random() < 0.7:  # 70% chance (most receipts are forwarded)
        # Resize down (simulates WhatsApp compression)
        scale = random.uniform(0.5, 0.8)
        small_w, small_h = int(w * scale), int(h * scale)
        img = img.resize((small_w, small_h), Image.LANCZOS)
        # Resize back up (when viewed fullscreen on phone)
        img = img.resize((w, h), Image.LANCZOS)
        # WhatsApp recompression
        buf2 = io.BytesIO()
        wa_quality = random.randint(65, 80)
        img.save(buf2, "JPEG", quality=wa_quality)
        buf2.seek(0)
        img = Image.open(buf2).convert("RGB")
    
    # --- Stage 3: Add slight Gaussian noise (sensor/screen capture noise) ---
    if random.random() < 0.5:  # 50% chance
        arr = np.array(img, dtype=np.float32)
        noise_strength = random.uniform(1.0, 4.0)
        noise = np.random.normal(0, noise_strength, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    
    # --- Stage 4: Slight blur (screen-to-camera capture or app smoothing) ---
    if random.random() < 0.3:  # 30% chance
        blur_radius = random.uniform(0.3, 0.8)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # --- Stage 5: Random slight brightness/contrast shift ---
    if random.random() < 0.4:  # 40% chance
        from PIL import ImageEnhance
        # Brightness shift (different screen brightness when screenshotting)
        brightness = random.uniform(0.92, 1.08)
        img = ImageEnhance.Brightness(img).enhance(brightness)
        # Contrast shift
        contrast = random.uniform(0.95, 1.05)
        img = ImageEnhance.Contrast(img).enhance(contrast)
    
    # --- Stage 6: Final save (the file that reaches the forensic system) ---
    final_quality = random.randint(70, 92)
    img.save(output_path, "JPEG", quality=final_quality)


# --- Section 3: Bank Template Stampers ---
# (All 10 bank generators remain IDENTICAL - only save_tampered_image changed)

def generate_rhb(output_path, dt_obj, f_amt, f_name):
    t_path = os.path.join(TEMPLATES_DIR, "template_1.jpg")
    if not os.path.exists(t_path): return
    
    img = Image.open(t_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    f_date = dt_obj.strftime("%I:%M%p %A, %d %b %Y")
    text_color = (12, 33, 54) 

    draw.text((int(w * 0.05), int(h * 0.16)), f_date, fill=text_color, font=get_scaled_font(w, 0.030), anchor="ls")
    draw.text((int(w * 0.21), int(h * 0.385)), f_amt, fill=text_color, font=get_scaled_font(w, 0.075, True), anchor="ls")
    draw.text((int(w * 0.05), int(h * 0.48)), f_name, fill=text_color, font=get_scaled_font(w, 0.035, True), anchor="ls")
    save_tampered_image(img, output_path)

def generate_bank_islam(output_path, dt_obj, f_amt, f_name):
    t_path = os.path.join(TEMPLATES_DIR, "template_2.jpg")
    if not os.path.exists(t_path): return
    
    img = Image.open(t_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    f_date_top = dt_obj.strftime("%d %b %Y %I:%M:%S %p")
    f_date_ref = (dt_obj + timedelta(seconds=1)).strftime("%d %b %Y %I:%M:%S %p")
    text_color = (30, 30, 30) 

    draw.text((int(w * 0.11), int(h * 0.178)), f_date_top, fill=text_color, font=get_scaled_font(w, 0.025), anchor="ls")
    draw.text((int(w * 0.11), int(h * 0.315)), f_date_ref, fill=text_color, font=get_scaled_font(w, 0.025), anchor="ls")
    draw.text((int(w * 0.11), int(h * 0.355)), f_name, fill=text_color, font=get_scaled_font(w, 0.026, False), anchor="ls")
    draw.text((int(w * 0.20), int(h * 0.685)), f_amt, fill=text_color, font=get_scaled_font(w, 0.055, True), anchor="ls")
    save_tampered_image(img, output_path)

def generate_cimb(output_path, dt_obj, f_amt, f_name):
    t_path = os.path.join(TEMPLATES_DIR, "template_3.jpg")
    if not os.path.exists(t_path): return
    
    img = Image.open(t_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    f_date = dt_obj.strftime("%d %b %Y %I:%M:%S %p")
    text_color = (35, 35, 35) 

    draw.text((int(w * 0.50), int(h * 0.288)), f_date, fill=text_color, font=get_scaled_font(w, 0.016, False), anchor="ms")
    draw.text((int(w * 0.495), int(h * 0.258)), f_amt, fill=text_color, font=get_scaled_font(w, 0.034, False), anchor="ls")
    draw.text((int(w * 0.76), int(h * 0.405)), f_name, fill=text_color, font=get_scaled_font(w, 0.019, True), anchor="rs")
    save_tampered_image(img, output_path)

def generate_bsn(output_path, dt_obj, f_amt, f_name):
    t_path = os.path.join(TEMPLATES_DIR, "template_4.jpg")
    if not os.path.exists(t_path): return
    
    img = Image.open(t_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    f_date = dt_obj.strftime("%d %B %Y, %I:%M %p")

    draw.text((int(w * 0.45), int(h * 0.230)), f_amt, fill=(255,255,255), font=get_scaled_font(w, 0.055, True), anchor="ls")
    draw.text((int(w * 0.26), int(h * 0.530)), f_name, fill=(30,30,30), font=get_scaled_font(w, 0.028, True), anchor="ls")
    draw.text((int(w * 0.26), int(h * 0.625)), f_date, fill=(30,30,30), font=get_scaled_font(w, 0.028, True), anchor="ls")
    save_tampered_image(img, output_path)

def generate_hlb(output_path, dt_obj, f_amt, f_name):
    t_path = os.path.join(TEMPLATES_DIR, "template_5.jpg")
    if not os.path.exists(t_path): return
    
    img = Image.open(t_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    f_datetime = dt_obj.strftime("%d %b %Y %I:%M%p")
    f_transfer_date = dt_obj.strftime("%d %b %Y")
    
    base_font = get_scaled_font(w, 0.020, False)
    col_x = 0.540

    draw.text((int(w * col_x), int(h * 0.153)), f_datetime, fill=(20,20,20), font=base_font, anchor="ls")
    draw.text((int(w * col_x), int(h * 0.194)), f_amt, fill=(20,20,20), font=base_font, anchor="ls")
    draw.text((int(w * col_x), int(h * 0.296)), f_name, fill=(20,20,20), font=base_font, anchor="ls")
    draw.text((int(w * col_x), int(h * 0.483)), f_transfer_date, fill=(20,20,20), font=base_font, anchor="ls")
    save_tampered_image(img, output_path)

def generate_maybank(output_path, dt_obj, f_amt, f_name):
    t_path = os.path.join(TEMPLATES_DIR, "template_6.jpg")
    if not os.path.exists(t_path): return
    
    img = Image.open(t_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    f_date = dt_obj.strftime("%d %b %Y, %I:%M %p")
    f_amt_str = f"RM {f_amt}" 

    left_margin = 0.065
    draw.text((int(w * left_margin), int(h * 0.290)), f_name, fill=(10,10,10), font=get_scaled_font(w, 0.030, True), anchor="ls")
    draw.text((int(w * left_margin), int(h * 0.660)), f_amt_str, fill=(10,10,10), font=get_scaled_font(w, 0.052, True), anchor="ls")
    draw.text((int(w * 0.95), int(h * 0.210)), f_date, fill=(90,90,90), font=get_scaled_font(w, 0.035, False), anchor="rs")
    save_tampered_image(img, output_path)

def generate_bank_rakyat(output_path, dt_obj, f_amt, f_name):
    t_path = os.path.join(TEMPLATES_DIR, "template_7.jpg")
    if not os.path.exists(t_path): return
    
    img = Image.open(t_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    f_date_top = dt_obj.strftime("%d %b %Y %I:%M %p")
    f_date_bottom = dt_obj.strftime("%d/%m/%Y")

    draw.text((int(w * 0.065), int(h * 0.300)), f_date_top, fill=(15,15,15), font=get_scaled_font(w, 0.030, False), anchor="ls")
    draw.text((int(w * 0.370), int(h * 0.393)), f_name, fill=(15,15,15), font=get_scaled_font(w, 0.028, False), anchor="ls")
    draw.text((int(w * 0.135), int(h * 0.527)), f_amt, fill=(15,15,15), font=get_scaled_font(w, 0.032, True), anchor="ls")
    draw.text((int(w * 0.135), int(h * 0.649)), f_amt, fill=(15,15,15), font=get_scaled_font(w, 0.032, True), anchor="ls")
    draw.text((int(w * 0.065), int(h * 0.710)), f_date_bottom, fill=(15,15,15), font=get_scaled_font(w, 0.032, True), anchor="ls")
    save_tampered_image(img, output_path)

def generate_tng(output_path, dt_obj, f_amt, f_name):
    t_path = os.path.join(TEMPLATES_DIR, "template_8.jpg")
    if not os.path.exists(t_path): return
    
    img = Image.open(t_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    f_date = dt_obj.strftime("%d/%m/%Y %H:%M:%S")
    right_margin = 0.95

    draw.text((int(w * 0.41), int(h * 0.285)), f_amt, fill=(10,10,10), font=get_scaled_font(w, 0.068, False), anchor="ls")
    draw.text((int(w * right_margin), int(h * 0.400)), f_name, fill=(10,10,10), font=get_scaled_font(w, 0.030, False), anchor="rs")
    draw.text((int(w * right_margin), int(h * 0.660)), f_date, fill=(10,10,10), font=get_scaled_font(w, 0.030, False), anchor="rs")
    save_tampered_image(img, output_path)

def generate_public_bank(output_path, dt_obj, f_amt, f_name):
    t_path = os.path.join(TEMPLATES_DIR, "template_9.jpg")
    if not os.path.exists(t_path): return
    
    img = Image.open(t_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    f_date = dt_obj.strftime("%d/%m/%Y %I:%M:%S %p")
    right_margin = 0.915

    draw.text((int(w * 0.495), int(h * 0.166)), f_amt, fill=(40,40,40), font=get_scaled_font(w, 0.045, True), anchor="ls")
    draw.text((int(w * right_margin), int(h * 0.380)), f_date, fill=(40,40,40), font=get_scaled_font(w, 0.028, False), anchor="rs")
    draw.text((int(w * right_margin), int(h * 0.655)), f_name, fill=(40,40,40), font=get_scaled_font(w, 0.028, False), anchor="rs")
    save_tampered_image(img, output_path)

def generate_ryt_bank(output_path, dt_obj, f_amt, f_name):
    t_path = os.path.join(TEMPLATES_DIR, "template_10.jpg")
    if not os.path.exists(t_path): return
    
    img = Image.open(t_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    f_date = dt_obj.strftime("%d %b %Y, %I:%M %p").lstrip("0") 

    draw.text((int(w * 0.455), int(h * 0.170)), f_amt, fill=(255,255,255), font=get_scaled_font(w, 0.065, True), anchor="ls")
    draw.text((int(w * 0.500), int(h * 0.220)), f_date, fill=(255,255,255), font=get_scaled_font(w, 0.025, False), anchor="ms")
    draw.text((int(w * 0.260), int(h * 0.410)), f_name, fill=(20,20,20), font=get_scaled_font(w, 0.032, True), anchor="ls")
    save_tampered_image(img, output_path)

# --- Section 4: The Factory Engine ---

def generate_dataset(num_samples_per_bank=50):
    """
    Loops through all 10 bank templates and creates randomized tampered images.
    If num_samples_per_bank=50, this outputs 500 total images.
    """
    bank_generators = {
        "RHB": generate_rhb,
        "BankIslam": generate_bank_islam,
        "CIMB": generate_cimb,
        "BSN": generate_bsn,
        "HLB": generate_hlb,
        "Maybank": generate_maybank,
        "BankRakyat": generate_bank_rakyat,
        "TNG": generate_tng,
        "PublicBank": generate_public_bank,
        "RYT": generate_ryt_bank
    }

    print(f"🚀 Starting mass generation! Target: {num_samples_per_bank} images per bank.")
    total_generated = 0

    for bank_name, generator_func in bank_generators.items():
        print(f"Generating for {bank_name}...")
        
        for i in range(1, num_samples_per_bank + 1):
            f_name = random.choice(COMPANY_NAMES)
            f_amt = get_random_amount()
            dt_obj = get_random_date()
            
            filename = f"{bank_name}_tampered_{str(i).zfill(4)}.jpg"
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            try:
                generator_func(output_path, dt_obj, f_amt, f_name)
                total_generated += 1
            except Exception as e:
                print(f"Error generating {filename}: {e}")

    print(f"✅ Generation complete! {total_generated} tampered receipts saved to {OUTPUT_DIR}/")

# --- Execution ---
if __name__ == "__main__":
    generate_dataset(num_samples_per_bank=63)