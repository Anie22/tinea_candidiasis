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
