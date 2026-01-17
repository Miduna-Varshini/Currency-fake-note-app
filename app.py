import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# =========================
# CONFIG
# =========================
MODEL_URL = "https://drive.google.com/uc?id=1bYipthbNsOZexDJvQ6tUJemUY__PilZH"
MODEL_PATH = "currency_cnn_final.h5"
IMG_SIZE = (128, 128)

# =========================
# DOWNLOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Currency Fake Note Detector", layout="centered")

st.title("💵 Indian Currency Fake Note Detection")
st.write("Upload a currency note image to check whether it is **Real** or **Fake**.")

uploaded_file = st.file_uploader("Upload Currency Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success("✅ **REAL Currency Note**")
    else:
        st.error("❌ **FAKE Currency Note**")

    st.write(f"Prediction Confidence: `{prediction:.2f}`")
