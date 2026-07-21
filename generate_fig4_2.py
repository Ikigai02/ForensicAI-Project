# save as: generate_fig4_2.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
TRAIN_DIR = 'dataset/final_split/train'
VAL_DIR = 'dataset/final_split/val'

train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=True)
val = val_gen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

class_counts = np.bincount(train.classes)
total = train.samples
class_weight = {i: total / (len(class_counts) * class_counts[i]) for i in range(len(class_counts))}

# Three LR configurations: (Phase1_LR, Phase2_LR)
configs = [
    ("Aggressive (1e-2 / 1e-4)", 1e-2, 1e-4),
    ("Conservative (1e-4 / 1e-6)", 1e-4, 1e-6),
    ("Optimal (1e-3 / 1e-5)", 1e-3, 1e-5),
]

all_val_losses = {}

for label, lr1, lr2 in configs:
    print(f"\n{'='*50}")
    print(f"Training: {label}")
    print(f"{'='*50}")
    
    base = MobileNetV2(input_shape=(128,128,3), include_top=False, weights='imagenet')
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(base.input, out)
    
    # Phase 1
    model.compile(optimizer=tf.keras.optimizers.Adam(lr1), loss='binary_crossentropy', metrics=['accuracy'])
    h1 = model.fit(train, validation_data=val, epochs=10, class_weight=class_weight, verbose=1)
    
    # Phase 2
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(lr2), loss='binary_crossentropy', metrics=['accuracy'])
    h2 = model.fit(train, validation_data=val, epochs=10, class_weight=class_weight, verbose=1)
    
    combined_loss = h1.history['val_loss'] + h2.history['val_loss']
    all_val_losses[label] = combined_loss

plt.figure(figsize=(10, 6))
for label, losses in all_val_losses.items():
    plt.plot(range(1, len(losses)+1), losses, label=label, linewidth=2)

plt.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='Phase 1 → Phase 2')
plt.title('Figure 4.2: Validation Loss Across Learning Rate Configurations', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure_4_2.png', dpi=300)
plt.show()
print("Saved as figure_4_2.png")