import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import json

app = Flask(__name__, static_folder='static')

IMG_SIZE = 64
MODEL_PATH = 'asl_model.h5'

model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/api/hello')
def hello():
    return {'message': 'Hello from Flask!'}

@app.route("/predict", methods=["POST"])
def predict():
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    file = request.files['image']
    image = Image.open(file.stream)
    processed_image = preprocess_image(image, IMG_SIZE)

    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    class_labels = {v: k for k, v in class_indices.items()}
    class_labels_inv = {v: k for k, v in class_labels.items()}
    predicted_letter = class_labels_inv[predicted_index]

    return jsonify({'letter': predicted_letter})

def preprocess_image(image, img_size):
    # Convert to grayscale, resize, normalize, and reshape
    image = image.convert('L')  # 'L' mode is for grayscale
    image = image.resize((img_size, img_size))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape(1, img_size, img_size, 1)  # Add batch and channel dimensions
    return image_array

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)