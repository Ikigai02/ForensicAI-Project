# diagnose_misclassified.py (standalone – no import from app.py)
import os
import io
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance

# --------------------- COPY OF prepare_ela_image (without Flask) ---------------------
def prepare_ela_image(path, quality=90):
    """
    Generate ELA image using an in‑memory buffer.
    Returns: (ela_image_resized, cv_tamper_flag, tamper_regions)
    """
    original = Image.open(path).convert('RGB')

    # Save to BytesIO buffer
    buffer = io.BytesIO()
    original.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    # Difference
    ela_raw = ImageChops.difference(original, resaved)

    # Scale contrast
    extrema = ela_raw.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    ela_scaled = ImageEnhance.Brightness(ela_raw).enhance(255.0 / max_diff)

    # OpenCV detection with adaptive threshold
    ela_cv = cv2.cvtColor(np.array(ela_scaled), cv2.COLOR_RGB2GRAY)
    mean_val = np.mean(ela_cv)
    std_val = np.std(ela_cv)
    adaptive_thresh = max(120, mean_val + std_val * 1.5)
    _, thresh = cv2.threshold(ela_cv, int(adaptive_thresh), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv_tamper_flag = False
    tamper_regions = []
    for c in contours:
        area = cv2.contourArea(c)
        if 50 < area < 3000:
            x, y, w, h = cv2.boundingRect(c)
            if w < ela_cv.shape[1] * 0.8 and h < ela_cv.shape[0] * 0.8:
                cv_tamper_flag = True
                tamper_regions.append(area)

    # Resize for CNN (128x128)
    ela_resized = ela_scaled.resize((128, 128), resample=Image.LANCZOS)

    return ela_resized, cv_tamper_flag, tamper_regions

# --------------------- DIAGNOSTIC FUNCTION ---------------------
def test_single_image(image_path, model):
    print(f"\n📄 Testing: {os.path.basename(image_path)}")
    try:
        ela_img, cv_flag, tamper_regions = prepare_ela_image(image_path)
        ela_arr = np.array(ela_img) / 255.0
        ela_arr = ela_arr.reshape(-1, 128, 128, 3)
        pred = model.predict(ela_arr, verbose=0)[0][0]
        ela_var = np.var(ela_arr) * 1000
        ela_peak = np.max(ela_arr) * 100
        print(f"  CNN prediction: {pred:.3f}")
        print(f"  ELA variance: {ela_var:.2f}, peak: {ela_peak:.1f}")
        print(f"  OpenCV flagged: {cv_flag}, regions: {len(tamper_regions)}")
        # Determine result with thresholds used in app.py
        if pred > 0.30:
            verdict = "TAMPERED (low threshold)"
        elif pred > 0.20:
            verdict = "TAMPERED (very low threshold)"
        else:
            verdict = "AUTHENTIC"
        print(f"  → {verdict}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# --------------------- MAIN ---------------------
if __name__ == "__main__":
    # Load model once
    model = load_model('forensic_ai_model.h5')
    print("✅ Model loaded.\n")

    # Change this path to your test folder
    test_folder = "dataset/final_split/test/tampered"

    if not os.path.exists(test_folder):
        print(f"❌ Folder not found: {test_folder}")
        print("Please update the 'test_folder' variable with the correct path.")
        exit()

    # Loop through Bank Islam images
    for f in os.listdir(test_folder):
        if "bankislam" in f.lower():
            test_single_image(os.path.join(test_folder, f), model)

    # Optional: test all tampered images (uncomment next lines)
    # print("\n" + "="*50)
    # print("Testing ALL tampered images:")
    # for f in os.listdir(test_folder):
    #     test_single_image(os.path.join(test_folder, f), model)