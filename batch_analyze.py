import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageChops, ImageEnhance

MODEL_PATH = "forensic_ai_model.h5"
ELA_QUALITY = 75
OPTIMAL_THRESHOLD = 0.70

# =====================================================
# ELA function (matches app.py exactly)
# =====================================================
def prepare_ela_image(path):
    original = Image.open(path).convert("RGB")
    
    w, h = original.size
    crop_size = min(w, int(h * 0.7))
    left, top, right, bottom = 0, h - crop_size, w, h
    working = original.crop((left, top, right, bottom))

    temp = "temp_resave.jpg"
    working.save(temp, "JPEG", quality=ELA_QUALITY)
    resaved = Image.open(temp)

    ela_raw = ImageChops.difference(working, resaved)
    extrema = ela_raw.getextrema()
    max_diff = max([e[1] for e in extrema]) or 1
    ela = ImageEnhance.Brightness(ela_raw).enhance(255.0 / max_diff)

    if os.path.exists(temp):
        os.remove(temp)

    return ela.resize((128, 128), Image.LANCZOS)

# =====================================================
# Load model
# =====================================================
model = load_model(MODEL_PATH)
print(f"✅ Model loaded. Threshold: {OPTIMAL_THRESHOLD}")

# =====================================================
# Batch analyze function
# =====================================================
def analyze_batch(folder_path, label_is_tampered):
    results = []
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return results
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"\nAnalyzing {len(files)} files from: {folder_path}")

    for f in files:
        path = os.path.join(folder_path, f)
        try:
            ela_img = prepare_ela_image(path)
            
            # CRITICAL: MobileNetV2 preprocessing
            ela_arr = np.array(ela_img, dtype=np.float32)
            ela_arr = ela_arr.reshape(1, 128, 128, 3)
            ela_arr = preprocess_input(ela_arr)
            
            score = float(model.predict(ela_arr, verbose=0)[0][0])
            predicted_tampered = score > OPTIMAL_THRESHOLD
            actual_tampered = label_is_tampered
            correct = predicted_tampered == actual_tampered
            
            results.append({
                "file": f,
                "score": score,
                "label": 1 if label_is_tampered else 0,
                "predicted": 1 if predicted_tampered else 0,
                "correct": correct
            })
        except Exception as e:
            print(f"❌ Skipping {f}: {e}")
    return results

# =====================================================
# Run on test set
# =====================================================
print("\n" + "=" * 60)
print("BATCH ANALYSIS — TEST SET")
print("=" * 60)

# Use RAW images, not the ELA-processed ones in final_split
# This simulates exactly what app.py does with a fresh upload
auth_results = analyze_batch("dataset/raw/authentic", label_is_tampered=False)
tam_results = analyze_batch("dataset/raw/tampered", label_is_tampered=True)

all_results = auth_results + tam_results

# =====================================================
# Summary stats
# =====================================================
total = len(all_results)
correct = sum(1 for r in all_results if r["correct"])
accuracy = correct / total * 100 if total > 0 else 0

auth_correct = sum(1 for r in auth_results if r["correct"])
auth_total = len(auth_results)
auth_acc = auth_correct / auth_total * 100 if auth_total > 0 else 0

tam_correct = sum(1 for r in tam_results if r["correct"])
tam_total = len(tam_results)
tam_acc = tam_correct / tam_total * 100 if tam_total > 0 else 0

print(f"\n{'=' * 60}")
print(f"RESULTS SUMMARY (Threshold: {OPTIMAL_THRESHOLD})")
print(f"{'=' * 60}")
print(f"Overall Accuracy : {accuracy:.2f}% ({correct}/{total})")
print(f"Authentic Accuracy: {auth_acc:.2f}% ({auth_correct}/{auth_total})")
print(f"Tampered Accuracy : {tam_acc:.2f}% ({tam_correct}/{tam_total})")

# =====================================================
# Show misclassified files
# =====================================================
fp = [r for r in auth_results if not r["correct"]]
fn = [r for r in tam_results if not r["correct"]]

print(f"\nFalse Positives (authentic flagged as tampered): {len(fp)}")
for r in sorted(fp, key=lambda x: -x["score"])[:10]:
    print(f"  Score: {r['score']:.4f} | {r['file']}")

print(f"\nFalse Negatives (tampered passed as authentic): {len(fn)}")
for r in sorted(fn, key=lambda x: x["score"])[:10]:
    print(f"  Score: {r['score']:.4f} | {r['file']}")

if len(fp) > 10:
    print(f"  ... and {len(fp) - 10} more")
if len(fn) > 10:
    print(f"  ... and {len(fn) - 10} more")