import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import json
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

IMG_SIZE = 64
MODEL_PATH = 'asl_model.h5'

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

    # Preprocess it to match training input
    processed_image = preprocess_image(image, IMG_SIZE)

    # Predict
    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)

    # Reverse class_indices mapping (int → letter)
    class_labels = {v: k for k, v in class_indices.items()}

    predicted_letter = class_labels[int(predicted_index)]
    print("Predicted letter:", predicted_letter)

    return jsonify({'letter': predicted_letter})

def preprocess_image(image, img_size):
    # Convert to grayscale
    image_rgb = np.array(image)


    # Step 1: Crop center 300x300 box
    h, w = image_rbg.shape
    cx, cy = w // 2, h // 2
    half = 150
    x1, x2 = cx - half, cx + half
    y1, y2 = cy - half, cy + half

    # Ensure within bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    roi = image_rgb[y1:y2, x1:x2]  # Same as OpenCV script

    # Step 2: Convert to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)  # RGB → GRAY

    # Step 3: Resize to 64x64
    roi_resized = cv2.resize(roi_gray, (img_size, img_size))

    # Step 4: Normalize and reshape
    normalized = roi_resized / 255.0
    reshaped = normalized.reshape(1, img_size, img_size, 1)

    return roi_resized

    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
