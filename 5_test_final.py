import tensorflow as tf
import os

# Load the trained model
model = tf.keras.models.load_model('forensic_ai_model.h5')

# Load the test set
test_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/final_split/test',
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary'
)

# =====================================================
# FIX: Normalization must match training
# Custom CNN (4_train_model.py) uses rescale=1./255
# MobileNetV2 (4_train_model_mobilenetv2.py) uses preprocess_input
# Using 1./255 here for the custom CNN
# =====================================================
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Get the final score
print("\n🔍 Evaluating on Test Set...")
loss, accuracy = model.evaluate(test_ds)

print(f"\n✅ FINAL TEST ACCURACY: {accuracy * 100:.2f}%")
print(f"   Loss: {loss:.4f}")