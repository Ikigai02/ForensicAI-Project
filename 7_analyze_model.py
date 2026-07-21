import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

MODEL_PATH = "forensic_ai_model.h5"
TEST_DIR = "dataset/final_split/test"
IMG_SIZE = (128, 128)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=32,
    shuffle=False,
    label_mode="binary"
)

# CRITICAL: preprocess_input for MobileNetV2 (scales to [-1, 1])
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

all_scores = []
all_labels = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    all_scores.extend(predictions.flatten())
    all_labels.extend(labels.numpy().flatten())

all_scores = np.array(all_scores).flatten()
all_labels = np.array(all_labels).flatten()

print("\n" + "=" * 60)
print("DEBUG INFO")
print("=" * 60)
print("all_scores shape:", all_scores.shape)
print("all_labels shape:", all_labels.shape)
print("\nFirst 20 Scores:")
print(np.round(all_scores[:20], 4))
print("\nFirst 20 Labels:")
print(all_labels[:20])

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

thresholds = np.arange(0.05, 1.0, 0.05)
best_acc = 0
best_threshold = 0

for threshold in thresholds:
    preds = (all_scores > threshold).astype(int)
    accuracy = np.mean(preds == all_labels)
    print(f"Threshold {threshold:.2f} -> Accuracy: {accuracy * 100:.2f}%")
    if accuracy > best_acc:
        best_acc = accuracy
        best_threshold = threshold

print(f"\nBEST THRESHOLD: {best_threshold:.2f}")
print(f"BEST ACCURACY: {best_acc*100:.2f}%")

preds = (all_scores > best_threshold).astype(int)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
cm = confusion_matrix(all_labels, preds)
print(cm)

print("\nCLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(all_labels, preds, target_names=["Authentic", "Tampered"]))

auth_scores = all_scores[all_labels == 0]
tamper_scores = all_scores[all_labels == 1]

print("\n" + "=" * 60)
print("SCORE DISTRIBUTION")
print("=" * 60)
print(f"Authentic Count : {len(auth_scores)}")
print(f"Authentic Avg   : {np.mean(auth_scores):.4f}")
print(f"Authentic Min   : {np.min(auth_scores):.4f}")
print(f"Authentic Max   : {np.max(auth_scores):.4f}")
print()
print(f"Tampered Count  : {len(tamper_scores)}")
print(f"Tampered Avg    : {np.mean(tamper_scores):.4f}")
print(f"Tampered Min    : {np.min(tamper_scores):.4f}")
print(f"Tampered Max    : {np.max(tamper_scores):.4f}")

print(f"\nSEPARATION GAP: {abs(np.mean(tamper_scores) - np.mean(auth_scores)):.4f}")
print("(Higher = better. Target: > 0.30)")