import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import json
import os
from datetime import datetime
from flask_cors import CORS
import statistics

app = Flask(__name__, static_folder='static')
CORS(app)

MODEL_PATH = 'asl_model_v07.h5'
IMG_SIZE = 64

model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/api/hello')
def hello():
    return {'message': 'Hello from Flask!'}

@app.route("/predict", methods=["POST"])
def predict():
    # Load label mapping
    with open("class_labels.json", "r") as f:
        class_indices = json.load(f)

    # unpack body
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # raw_image_path = os.path.join("./", f"raw_{timestamp}.jpg")
    # cv2.imwrite(raw_image_path, image_bgr)

    # preprocess image for model
    processed_image = preprocess_image(image_bgr)
    processed_image = np.expand_dims(processed_image, axis=0)  # shape: (1, 64, 64, 1)

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

@app.route("/predict-batch", methods=["POST"])
def predict_batch():

    with open("class_labels.json", "r") as f:
        class_indices = json.load(f)

    # unpack body
    files = request.files.getlist("image")

    results = []

    for index, file in enumerate(files):
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save image
        # raw_image_path = os.path.join("./", f"raw_{timestamp}_{index}.jpg")
        # cv2.imwrite(raw_image_path, image_bgr)

        # preprocess image for model
        processed_image = preprocess_image(image_bgr)
        processed_image = np.expand_dims(processed_image, axis=0)  # shape: (1, 64, 64, 1)

        # processed_image_to_save = Image.fromarray((processed_image[0, :, :, 0] * 255).astype(np.uint8))    
        # processed_path = os.path.join("./", f"processed_{timestamp}.jpg")
        # processed_image_to_save.save(processed_path)

        # Predict
        prediction = model.predict(processed_image)
        predicted_index = np.argmax(prediction)

        # Map index to label
        class_labels = {v: k for k, v in class_indices.items()}
        predicted_letter = class_labels[int(predicted_index)]
        print("Predicted letter:", predicted_letter)
        results.append(predicted_letter)
            
    mode_label = statistics.mode(results)
    return jsonify({'letter': mode_label})


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

    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    half = 150
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    roi = image[y1:y2, x1:x2]
    # Pad to 300x300 if ROI is smaller



    # Convert to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, (64, 64))   

    img = roi_resized.astype(np.uint8)  
    
    denoised = cv2.medianBlur(img, ksize=3)
    
    sobelx = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.convertScaleAbs(magnitude)
    
    _, strong_edges = cv2.threshold(denoised, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
    result = cv2.bitwise_and(denoised, denoised, mask=strong_edges)

    magnitude = result.astype("float32") / 255.0  # Scale back to 0-1

    return np.expand_dims(magnitude, axis=-1) 

# def new_func(img):
#     img = img.astype(np.uint8)  
    
#     denoised = cv2.medianBlur(img, ksize=3)
    
#     sobelx = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
#     sobely = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
#     magnitude = np.sqrt(sobelx**2 + sobely**2)
#     magnitude = cv2.convertScaleAbs(magnitude)
    
#     _, strong_edges = cv2.threshold(denoised, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
#     result = cv2.bitwise_and(denoised, denoised, mask=strong_edges)
    


#     magnitude = result.astype("float32") / 255.0  # Scale back to 0-1

#     return np.expand_dims(magnitude, axis=-1) 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
