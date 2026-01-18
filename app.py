import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import google.generativeai as genai

# =========================
# APP CONFIG
# =========================
st.set_page_config(
    page_title="Currency Fake Note Detector",
    layout="centered"
)

# =========================
# GEMINI API CONFIG (RENDER)
# =========================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning("⚠️ Gemini API key not found. Add it in Render Environment Variables.")

# =========================
# MODEL CONFIG
# =========================
MODEL_URL = "https://drive.google.com/uc?id=1bYipthbNsOZexDJvQ6tUJemUY__PilZH"
MODEL_PATH = "currency_cnn_final.h5"
IMG_SIZE = (128, 128)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_currency_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_currency_model()

# =========================
# UI
# =========================
st.title("💵 Indian Currency Fake Note Detection")
st.write(
    "Upload an image of an Indian currency note to check whether it is **REAL** or **FAKE**."
)

uploaded_file = st.file_uploader(
    "📤 Upload Currency Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction) * 100

    st.markdown("---")

    if prediction >= 0.5:
        result = "REAL"
        st.success("✅ **REAL Currency Note**")
    else:
        result = "FAKE"
        st.error("❌ **FAKE Currency Note**")

    st.info(f"🔍 Confidence: **{confidence:.2f}%**")

    # =========================
    # GEMINI EXPLANATION
    # =========================
    if GEMINI_API_KEY:
        with st.spinner("🤖 Gemini is analyzing..."):
            model_gemini = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""
            A machine learning model predicted that the uploaded Indian currency note is {result}.
            Explain in simple terms what features usually indicate a {result} note.
            """

            response = model_gemini.generate_content(prompt)
            st.markdown("### 🧠 Gemini Explanation")
            st.write(response.text)
