import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "model/skin_classifier.keras"
LABELS_PATH = "model/labels.json"
BLANK_IMAGE_STD_CUTOFF = 5.0

st.set_page_config(page_title="Tinea vs Candidiasis Classifier", layout="centered")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        meta = json.load(f)

    return model, meta


def preprocess(image, img_size):
    image = image.convert("RGB").resize(img_size)
    array = np.array(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    return preprocess_input(array)
