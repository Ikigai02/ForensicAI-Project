import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageChops

# This part is CRITICAL for the interface
def prepare_ela_image(path, quality):
    temp_filename = 'temp_resave.jpg'
    image = Image.open(path).convert('RGB')
    
    # Save at a lower quality to highlight compression differences
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    
    # Calculate the pixel-by-pixel difference
    ela_image = ImageChops.difference(image, temp_image)
    
    # Scale the contrast so the "noise" is visible to the CNN
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image.resize((128, 128))

# Load the saved "Brain"
model = load_model('forensic_ai_model.h5')

# The logic: Process -> Predict -> Result
# We will move this logic into the Flask 'app.py' next.