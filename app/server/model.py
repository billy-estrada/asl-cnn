import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import json
import os
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

MODEL_PATH = 'asl_model.h5'
IMG_SIZE = 64

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/api/hello')
def hello():
    return {'message': 'Hello from Flask!'}

@app.route("/predict", methods=["POST"])
def predict():
    # Load label mapping
    with open("class_labels.json", "r") as f:
        class_indices = json.load(f)

    # Receive image from client
    file = request.files['image']
    image = Image.open(file.stream)

    # Timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw uploaded image
    raw_image_path = os.path.join("./", f"raw_{timestamp}.jpg")
    image.save(raw_image_path)

    # Preprocess image for model
    processed_image = preprocess_image(image)

    # Optional: Save processed image for debugging
    processed_image_to_save = Image.fromarray((processed_image[0, :, :, 0] * 255).astype(np.uint8))
    processed_path = os.path.join("./", f"processed_{timestamp}.jpg")
    processed_image_to_save.save(processed_path)

    # Predict
    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)

    # Map index to label
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_letter = class_labels[int(predicted_index)]

    print("Predicted letter:", predicted_letter)
    return jsonify({'letter': predicted_letter})


def preprocess_image(image):
    """
    Matches OpenCV-style preprocessing:
    - center crop 300x300
    - grayscale
    - resize to 64x64
    - normalize
    - reshape to (1, 64, 64, 1)
    """
    # Convert to RGB NumPy array from PIL image
    image_rgb = np.array(image.convert("RGB"))

    h, w, _ = image_rgb.shape
    cx, cy = w // 2, h // 2
    half = 150
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    roi = image_rgb[y1:y2, x1:x2]

    # Pad to 300x300 if ROI is smaller


    # Convert to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Resize to model input
    roi_resized = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))

    # Normalize and reshape
    normalized = roi_resized.astype("float32") / 255.0
    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return reshaped


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
