from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests

app = Flask(__name__)

MODEL_URL = "https://huggingface.co/Trish004/cat-dog-classifier/resolve/main/cat_dog_classifier.keras"

model_path = "model.keras"

# download model if not present
try:
    open(model_path)
except:
    r = requests.get(MODEL_URL)
    open(model_path, "wb").write(r.content)

model = tf.keras.models.load_model(model_path)
