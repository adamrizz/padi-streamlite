import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
import tensorflow as tf  # masih dipakai untuk decode dan resize image

st.set_page_config(page_title="Deteksi Daun Padi", layout="centered")
st.title("🌾 Deteksi Penyakit Daun Padi (TFLite)")

MODEL_URL = "https://github.com/adamrizz/padi-streamlite/releases/download/v1.0/daun_padi_cnn_model.tflite"
MODEL_PATH = "daun_padi_cnn_model.tflite"

# Unduh model TFLite jika belum ada
@st.cache_resource
def load_tflite_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Mengunduh model TFLite dari GitHub..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
st.success("✅ Model TFLite berhasil dimuat.")

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
    img_array = np.expand_dims(np.array(img_resized), axis=0).astype(np.float32) / 255.0

    # Menyiapkan input dan output TFLite
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    predicted_index = int(np.argmax(output_data))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(np.max(tf.nn.softmax(output_data)))

    st.markdown(f"### ✅ Hasil Prediksi: `{predicted_label}`")
    st.markdown(f"**Confidence:** {confidence:.2%}")
    
    st.markdown("#### 📊 Semua Probabilitas:")
    for i, cls in enumerate(CLASS_NAMES):
        st.progress(output_data[i], text=f"{cls}: {output_data[i]:.2%}")
