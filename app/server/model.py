import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import cv2

app = Flask(__name__, static_folder='static')


@app.route('/api/hello')
def hello():
    return {'message': 'Hello from Flask!'}

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['image']
    image = Image.open(file.stream).resize((64, 64))
    # preprocess

    # predicted_letter 
    
    return jsonify({'letter': predicted_letter})

def preprocess_image(image):
    grayscale_iamge = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)