import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

# =====================================================
# CONFIG
# =====================================================
IMG_SIZE = (128, 128)   # Keep 128x128 to match ELA preprocessing
BATCH_SIZE = 32
TRAIN_DIR = 'dataset/final_split/train'
VAL_DIR = 'dataset/final_split/val'

# =====================================================
# DATA GENERATORS
# CRITICAL: preprocess_input scales pixels to [-1, 1]
# This MUST be used everywhere: training, validation, 
# testing (7_analyze_model.py), and inference (app.py)
# =====================================================

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    brightness_range=[0.95, 1.05]
)

# Validation: preprocess_input ONLY, no augmentation
val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\nClass mapping: {train.class_indices}")
print(f"Training samples: {train.samples}")
print(f"Validation samples: {val.samples}")

# =====================================================
# CLASS WEIGHTS
# =====================================================
class_counts = np.bincount(train.classes)
total = train.samples
class_weight = {
    i: total / (len(class_counts) * class_counts[i]) 
    for i in range(len(class_counts))
}
print(f"Class weights: {class_weight}")

# =====================================================
# MODEL: MobileNetV2 with frozen base
# Using 128x128 input (matches our ELA size)
# =====================================================

base = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False  # Freeze all base layers initially

x = GlobalAveragePooling2D()(base.output)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(base.input, out)

# =====================================================
# PHASE 1: Train only the top layers (base frozen)
# =====================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks_phase1 = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint('forensic_ai_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\n" + "=" * 50)
print("PHASE 1: Training top layers (base frozen)")
print("=" * 50)

model.fit(
    train,
    validation_data=val,
    epochs=15,
    callbacks=callbacks_phase1,
    class_weight=class_weight
)

# =====================================================
# PHASE 2: Fine-tune the last 30 layers of MobileNetV2
# Lower learning rate to avoid destroying pretrained weights
# =====================================================

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint('forensic_ai_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\n" + "=" * 50)
print("PHASE 2: Fine-tuning last 30 layers")
print("=" * 50)

history = model.fit(
    train,
    validation_data=val,
    epochs=30,
    callbacks=callbacks_phase2,
    class_weight=class_weight
)

# =====================================================
# RESULTS
# =====================================================
print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)
best_val_acc = max(history.history['val_accuracy'])
print(f"Best Phase 2 Val Accuracy: {best_val_acc:.4f}")
print(f"Model saved to: forensic_ai_model.h5")