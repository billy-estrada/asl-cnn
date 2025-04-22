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

model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/api/hello')
def hello():
    return {'message': 'Hello from Flask!'}

@app.route("/predict", methods=["POST"])
def predict():
    with open("class_labels.json", "r") as f:
        class_indices = json.load(f)
    file = request.files['image']
    image = Image.open(file.stream)
    processed_image = preprocess_image(image, IMG_SIZE)

    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    print(class_indices)
    class_labels = {v: k for k, v in class_indices.items()}
    print("Predicted index:", predicted_index)
    print("class_labels_inv keys:", list(class_labels.keys()))
    print("class_labels_inv items:", list(class_labels.values()))
    
    predicted_letter = class_labels[int(predicted_index)]
    print("predicted letter", predicted_letter)

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