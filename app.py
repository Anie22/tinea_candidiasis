import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "model/tinea_candidiasis_skin_classifier.keras"
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


def is_valid_photo(image):
    # rejects blank, corrupted, or solid-color uploads before they hit the model
    grayscale = image.convert("L").resize((64, 64))
    pixels = np.array(grayscale, dtype=np.float32)
    return pixels.std() > BLANK_IMAGE_STD_CUTOFF


def render_result(label, confidence, threshold):
    st.subheader("Result")

    if confidence < threshold:
        st.error("This isn't Tinea or Candidiasis.")
    else:
        st.success(label.title())

    st.write(f"Confidence: **{confidence * 100:.1f}%**  (threshold: {threshold * 100:.0f}%)")
    st.progress(confidence)


model, meta = load_model()

st.title("Tinea vs Candidiasis Classifier")
st.caption("Identifying Tinea and Candidiasis skin lesions")

if model is None:
    st.error(
        "No trained model found at `model/tinea_candidiasis_skin_classifier.keras`. "
        "Run `python train.py` after adding images to `data/tinea/` and "
        "`data/candidiasis/`, then reload this app."
    )
    st.stop()

class_names = meta["class_names"]
img_size = tuple(meta["img_size"])
threshold = meta["confidence_threshold"]

st.write(
    "Upload a close-up photo of a skin lesion, then click **Analyze** to check "
    "if it's Tinea, Candidiasis, or neither of the two."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("Couldn't read that file as an image. Please upload a JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    analyze_clicked = st.button("Analyze", type="primary")

    if analyze_clicked:
        if not is_valid_photo(image):
            st.error("This isn't Tinea or Candidiasis.")
            st.stop()

        with st.spinner("Analyzing..."):
            probs = model.predict(preprocess(image, img_size), verbose=0)[0]

        top_index = int(np.argmax(probs))
        confidence = float(probs[top_index])
        label = class_names[top_index]

        render_result(label, confidence, threshold)

        st.caption(
            "This is a student mini-project demo, not a medical diagnostic tool. "
            "Consult a healthcare professional for an actual diagnosis."
)    # rejects blank, corrupted, or solid-color uploads before they hit the model
    grayscale = image.convert("L").resize((64, 64))
    pixels = np.array(grayscale, dtype=np.float32)
    return pixels.std() > BLANK_IMAGE_STD_CUTOFF


def render_result(label, confidence, threshold):
    st.subheader("Result")

    if confidence < threshold:
        st.error("This isn't Tinea or Candidiasis.")
    else:
        st.success(label.title())

    st.write(f"Confidence: **{confidence * 100:.1f}%**  (threshold: {threshold * 100:.0f}%)")
    st.progress(confidence)


model, meta = load_model()

st.title("Tinea vs Candidiasis Classifier")
st.caption("Identifying Tinea and Candidiasis skin lesions")

if model is None:
    st.error(
        "No trained model found at `model/tinea_candidiasis_skin_classifier.keras`. "
        "Run `python train.py` after adding images to `data/tinea/` and "
        "`data/candidiasis/`, then reload this app."
    )
    st.stop()

class_names = meta["class_names"]
img_size = tuple(meta["img_size"])
threshold = meta["confidence_threshold"]

st.write(
    "Upload a close-up photo of a skin lesion, then click **Analyze** to check "
    "if it's Tinea, Candidiasis, or neither of the two."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("Couldn't read that file as an image. Please upload a JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    analyze_clicked = st.button("Analyze", type="primary")

    if analyze_clicked:
        if not is_valid_photo(image):
            st.error("This isn't Tinea or Candidiasis.")
            st.stop()

        with st.spinner("Analyzing..."):
            probs = model.predict(preprocess(image, img_size), verbose=0)[0]

        top_index = int(np.argmax(probs))
        confidence = float(probs[top_index])
        label = class_names[top_index]

        render_result(label, confidence, threshold)

        st.caption(
            "This is a student mini-project demo, not a medical diagnostic tool. "
            "Consult a healthcare professional for an actual diagnosis."
        )    # rejects blank, corrupted, or solid-color uploads before they hit the model
    grayscale = image.convert("L").resize((64, 64))
    pixels = np.array(grayscale, dtype=np.float32)
    return pixels.std() > BLANK_IMAGE_STD_CUTOFF


def render_result(label, confidence, threshold):
    st.subheader("Result")
    if confidence < threshold:
        st.error("This isn't Tinea or Candidiasis.")
    else:
        st.success(label.title())

    with st.expander("Details"):
        st.write(f"Confidence: {confidence * 100:.1f}% (threshold: {threshold * 100:.0f}%)")


model, meta = load_model()

st.title("Tinea vs Candidiasis Classifier")
st.caption("Identifying Tinea and Candidiasis skin lesions")

if model is None:
    st.error(
        "No trained model found at `model/skin_classifier.keras`. "
        "Run `python train.py` after adding images to `data/tinea/` and "
        "`data/candidiasis/`, then reload this app."
    )
    st.stop()

class_names = meta["class_names"]
img_size = tuple(meta["img_size"])
threshold = meta["confidence_threshold"]

st.write(
    "Upload a close-up photo of a skin lesion. The app will tell you if it's "
    "Tinea, Candidiasis, or neither of the two."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("Couldn't read that file as an image. Please upload a JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded image", use_column_width=True)

    if not is_valid_photo(image):
        st.error("This isn't Tinea or Candidiasis.")
        st.stop()

    with st.spinner("Analyzing..."):
        probs = model.predict(preprocess(image, img_size), verbose=0)[0]

    top_index = int(np.argmax(probs))
    confidence = float(probs[top_index])

    label = class_names[top_index]
    st.write("Confidence:", confidence)

    if confidence < threshold:
        st.warning(
            "Uncertain."
        )

    elif label == "tinea":
        st.success("Tinea detected")

    elif label == "candidiasis":
        st.success("Candidiasis detected")

    st.caption(
        "This is a student mini-project demo, not a medical diagnostic tool. "
        "Consult a healthcare professional for an actual diagnosis."
    )
