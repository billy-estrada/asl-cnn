import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import json
from flask_cors import CORS

app = Flask(__name__, static_folder='static')

CORS(app)

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
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    class_labels = {v: k for k, v in class_indices.items()}
    print("Predicted index:", predicted_index)
    print("class_labels_inv keys:", list(class_labels.keys()))
    print("class_labels_inv items:", list(class_labels.values()))
    
    predicted_letter = class_labels[int(predicted_index)]
    print("predicted letter", predicted_letter)

    return jsonify({'letter': predicted_letter})

def preprocess_image(image):
    # Convert to grayscale, resize, normalize, and reshape
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0  # Normalize
    img = img.reshape(1, 64, 64, 1)
    return img

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)