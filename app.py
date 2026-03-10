from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
from flask_cors import CORS
CORS(app)

app = Flask(__name__)

MODEL_URL = "https://huggingface.co/Trish004/cat-dog-classifier/resolve/main/cat_dog_classifier.keras"
MODEL_PATH = "model.keras"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    open(MODEL_PATH, "wb").write(r.content)

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(img):
    img = img.resize((128,128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file)

    img_array = preprocess(img)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Dog 🐶"
        confidence = float(prediction)
    else:
        label = "Cat 🐱"
        confidence = float(1 - prediction)

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)



