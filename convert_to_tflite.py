import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("forensic_ai_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("forensic_ai_model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Wrote forensic_ai_model.tflite ({len(tflite_model) / 1e6:.2f} MB)")
