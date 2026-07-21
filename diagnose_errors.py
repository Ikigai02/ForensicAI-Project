import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import defaultdict

MODEL_PATH = "forensic_ai_model.h5"
TEST_DIR = "dataset/final_split/test"
IMG_SIZE = (128, 128)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load test data WITH filenames
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=32,
    shuffle=False,
    label_mode="binary"
)

# Get the filenames and class names
filenames = test_ds.file_paths
class_names = test_ds.class_names
print(f"Classes: {class_names}")
print(f"Total test images: {len(filenames)}")

# Collect predictions
all_scores = []
all_labels = []

for images, labels in test_ds:
    images_processed = preprocess_input(images)
    predictions = model.predict(images_processed, verbose=0)
    all_scores.extend(predictions.flatten())
    all_labels.extend(labels.numpy().flatten())

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

THRESHOLD = 0.50

# ============================================================
# FIND ALL MISCLASSIFIED IMAGES
# ============================================================

print("\n" + "=" * 70)
print("FALSE POSITIVES (Authentic images wrongly flagged as TAMPERED)")
print("=" * 70)

fp_count = 0
fp_banks = defaultdict(int)

for i in range(len(all_scores)):
    if all_labels[i] == 0 and all_scores[i] > THRESHOLD:
        fname = filenames[i].split("/")[-1]
        print(f"  Score: {all_scores[i]:.4f} | {fname}")
        fp_count += 1
        # Try to extract bank name from filename
        bank = fname.split("_")[0] if "_" in fname else "unknown"
        fp_banks[bank] += 1

print(f"\nTotal False Positives: {fp_count}")
print("\nFP by source/bank:")
for bank, count in sorted(fp_banks.items(), key=lambda x: -x[1]):
    print(f"  {bank}: {count}")

print("\n" + "=" * 70)
print("FALSE NEGATIVES (Tampered images that PASSED as authentic)")
print("=" * 70)

fn_count = 0
fn_banks = defaultdict(int)

for i in range(len(all_scores)):
    if all_labels[i] == 1 and all_scores[i] <= THRESHOLD:
        fname = filenames[i].split("/")[-1]
        print(f"  Score: {all_scores[i]:.4f} | {fname}")
        fn_count += 1
        bank = fname.split("_")[0] if "_" in fname else "unknown"
        fn_banks[bank] += 1

print(f"\nTotal False Negatives: {fn_count}")
print("\nFN by source/bank:")
for bank, count in sorted(fn_banks.items(), key=lambda x: -x[1]):
    print(f"  {bank}: {count}")

# ============================================================
# SUMMARY: Which banks are problematic?
# ============================================================

print("\n" + "=" * 70)
print("PROBLEM BANKS SUMMARY")
print("=" * 70)

all_banks = set(list(fp_banks.keys()) + list(fn_banks.keys()))
for bank in sorted(all_banks):
    fp = fp_banks.get(bank, 0)
    fn = fn_banks.get(bank, 0)
    total_errors = fp + fn
    print(f"  {bank:20s} | FP: {fp:3d} | FN: {fn:3d} | Total Errors: {total_errors}")