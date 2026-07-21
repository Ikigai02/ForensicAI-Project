import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# =====================================================
# MODEL: Smaller architecture to prevent overfitting
# With ~900 training images, the old model (128 filters + 
# Dense 256) had too many parameters and memorized 
# the training set instead of learning real patterns.
# =====================================================

def build_model():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        
        # Light noise to improve generalization
        layers.GaussianNoise(0.02),
        
        # Block 1 — fewer filters
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        
        # Use GlobalAveragePooling instead of Flatten
        # This drastically reduces parameters and overfitting
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model

# =====================================================
# FIX 1: Separate datagens for train and validation
# Training gets augmentation, validation gets ONLY rescaling
# =====================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,           # Reduced from 10 — receipts are mostly upright
    width_shift_range=0.05,     # Reduced from 0.1
    height_shift_range=0.05,
    brightness_range=[0.95, 1.05],  # Subtle brightness variation
    zoom_range=0.05
)

# FIX 2: Validation datagen has NO augmentation — only rescaling
val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    "dataset/final_split/train",
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    "dataset/final_split/val",
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Print class mapping so you know which is 0 and which is 1
print("\nClass mapping:", train_generator.class_indices)
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

# =====================================================
# FIX 3: Class weights to handle imbalance (500 auth vs 630 tampered)
# =====================================================
import numpy as np

total = train_generator.samples
n_classes = len(train_generator.class_indices)
class_counts = np.bincount(train_generator.classes)
class_weight = {}

for i in range(n_classes):
    class_weight[i] = total / (n_classes * class_counts[i])

print(f"Class weights: {class_weight}")

# =====================================================
# FIX 4: EarlyStopping — stops training when val_loss 
# stops improving instead of running all 60 epochs
# =====================================================

model = build_model()
model.summary()

callbacks = [
    # Save best model based on val_accuracy
    ModelCheckpoint(
        "forensic_ai_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    # Stop if val_loss doesn't improve for 8 epochs
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce LR if val_loss plateaus for 3 epochs
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=60,
    callbacks=callbacks,
    class_weight=class_weight
)

# Print final results
print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)
best_val_acc = max(history.history['val_accuracy'])
best_val_loss = min(history.history['val_loss'])
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Model saved to: forensic_ai_model.h5")