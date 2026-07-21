import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
TRAIN_DIR = 'dataset/final_split/train'
VAL_DIR = 'dataset/final_split/val'

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    brightness_range=[0.95, 1.05]
)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=True)
val = val_gen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

class_counts = np.bincount(train.classes)
total = train.samples
class_weight = {i: total / (len(class_counts) * class_counts[i]) for i in range(len(class_counts))}

# Build model
base = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
out = Dense(1, activation='sigmoid')(x)
model = Model(base.input, out)

# ===== PHASE 1 =====
print("\n" + "=" * 50)
print("PHASE 1: Training top layers (base frozen)")
print("=" * 50)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

h1 = model.fit(
    train, validation_data=val, epochs=15,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)],
    class_weight=class_weight
)

# ===== PHASE 2 =====
print("\n" + "=" * 50)
print("PHASE 2: Fine-tuning last 30 layers")
print("=" * 50)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

h2 = model.fit(
    train, validation_data=val, epochs=30,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
    ],
    class_weight=class_weight
)

# ===== PLOT =====
phase1_acc = h1.history['accuracy']
phase1_val_acc = h1.history['val_accuracy']
phase2_acc = h2.history['accuracy']
phase2_val_acc = h2.history['val_accuracy']

all_acc = phase1_acc + phase2_acc
all_val_acc = phase1_val_acc + phase2_val_acc

total_epochs = len(all_acc)
phase1_end = len(phase1_acc)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, total_epochs+1), all_acc, label='Training Accuracy', linewidth=2)
ax.plot(range(1, total_epochs+1), all_val_acc, label='Validation Accuracy', linewidth=2)
ax.axvline(x=phase1_end, color='red', linestyle='--', alpha=0.7, label=f'Phase 1 → Phase 2 (Epoch {phase1_end})')

ax.set_title('Figure 4.3: MobileNetV2 Two-Phase Training Performance', fontsize=14)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax.text(phase1_end/2, 0.55, 'Phase 1\n(Frozen Base)', ha='center', fontsize=10, color='gray')
ax.text(phase1_end + (total_epochs - phase1_end)/2, 0.55, 'Phase 2\n(Fine-Tuning)', ha='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('figure_4_3.png', dpi=300)
plt.show()
print("Saved as figure_4_3.png")