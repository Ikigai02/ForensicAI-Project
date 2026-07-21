# save as: generate_fig4_7.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MODEL_PATH = "forensic_ai_model.h5"
TEST_DIR = "dataset/final_split/test"
IMG_SIZE = (128, 128)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=32, shuffle=False, label_mode="binary"
)

all_scores = []
all_labels = []

for images, labels in test_ds:
    images_processed = preprocess_input(images)
    predictions = model.predict(images_processed, verbose=0)
    all_scores.extend(predictions.flatten())
    all_labels.extend(labels.numpy().flatten())

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

auth_scores = all_scores[all_labels == 0]
tamp_scores = all_scores[all_labels == 1]

fig, ax = plt.subplots(figsize=(12, 6))

# Shade the three zones
ax.axvspan(0, 0.40, alpha=0.1, color='green', label='AUTHENTIC Zone (< 0.40)')
ax.axvspan(0.40, 0.70, alpha=0.1, color='orange', label='SUSPICIOUS Zone (0.40 - 0.70)')
ax.axvspan(0.70, 1.0, alpha=0.1, color='red', label='TAMPERED Zone (> 0.70)')

# Plot histograms
ax.hist(auth_scores, bins=30, alpha=0.7, color='#2ECC71', edgecolor='black', label=f'Authentic (n={len(auth_scores)}, avg={np.mean(auth_scores):.4f})')
ax.hist(tamp_scores, bins=30, alpha=0.7, color='#E74C3C', edgecolor='black', label=f'Tampered (n={len(tamp_scores)}, avg={np.mean(tamp_scores):.4f})')

# Threshold lines
ax.axvline(x=0.40, color='orange', linestyle='--', linewidth=2)
ax.axvline(x=0.70, color='red', linestyle='--', linewidth=2)

ax.text(0.40, ax.get_ylim()[1]*0.95, '0.40', ha='center', va='top', fontsize=11, fontweight='bold', color='orange')
ax.text(0.70, ax.get_ylim()[1]*0.95, '0.70', ha='center', va='top', fontsize=11, fontweight='bold', color='red')

ax.set_title('Figure 4.7: Score Distribution with Three-Tier Verdict Boundaries', fontsize=14)
ax.set_xlabel('CNN Confidence Score', fontsize=12)
ax.set_ylabel('Number of Images', fontsize=12)
ax.legend(fontsize=10, loc='upper center')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure_4_7.png', dpi=300)
plt.show()
print("Saved as figure_4_7.png")