import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
import io

st.set_page_config(page_title="Deteksi Daun Padi", layout="centered")
st.title("🌾 Deteksi Penyakit Daun Padi")

MODEL_URL = "https://github.com/adamrizz/padi-cnn/releases/download/v1.0/daun_padi_cnn_model.keras"
MODEL_PATH = "daun_padi_cnn_model.keras"

# Cek & unduh model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Mengunduh model dari GitHub..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
st.success("✅ Model berhasil dimuat.")

CLASS_NAMES = [
    "Bacterial Leaf Blight", "Leaf Blast", "Leaf Scald",
    "Brown Spot", "Narrow  Brown Spot", "Healthy"
]

# Upload gambar
uploaded_file = st.file_uploader("🖼️ Upload gambar daun padi", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Daun", use_column_width=True)

    img_resized = image.resize((150, 150))
    img_array = np.expand_dims(np.array(img_resized), axis=0) / 255.0

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_index = int(np.argmax(score))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(np.max(score))

    st.markdown(f"### ✅ Hasil Prediksi: `{predicted_label}`")
    st.markdown(f"**Confidence:** {confidence:.2%}")
    
    st.markdown("#### 📊 Semua Probabilitas:")
    for i, cls in enumerate(CLASS_NAMES):
        st.progress(score[i], text=f"{cls}: {score[i]:.2%}")
