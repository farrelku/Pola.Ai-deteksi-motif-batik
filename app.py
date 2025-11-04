import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pickle
import os

# --- Load Model ---
@st.cache_resource
def load_model_file():
    model_path = "batik_resnet50_model_final.h5"
    if not os.path.exists(model_path):
        st.error("❌ File model tidak ditemukan! Pastikan 'batik_resnet50_model_final.h5' ada di folder yang sama dengan app.py")
        st.stop()
    return tf.keras.models.load_model(model_path)

# --- Load Labels ---
@st.cache_resource
def load_labels():
    with open("labels.pkl", "rb") as f:
        labels = pickle.load(f)
    return {v: k for k, v in labels.items()}

# --- Load Deskripsi ---
@st.cache_resource
def load_descriptions():
    try:
        with open("deskripsi_batik.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("⚠️ File deskripsi_batik.json tidak ditemukan.")
        return {}

# --- Inisialisasi ---
model = load_model_file()
labels = load_labels()
deskripsi_batik = load_descriptions()

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Pola.Ai - Deteksi Motif Batik",
    page_icon="🧵",
    layout="wide"
)

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        """
        <h2 style='color:#FFFFFF;'>Pola.Ai</h2>
        <p style='color:#BBBBBB; text-align:justify;'>
        Pola.Ai adalah sistem berbasis kecerdasan buatan (AI) yang dirancang untuk mendeteksi dan mengidentifikasi
        berbagai motif batik dari gambar yang diunggah oleh pengguna.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.button("🏠 Home", use_container_width=True)

# --- Konten Utama ---
st.markdown(
    """
    <h1 style='text-align:center;'>🧵 Deteksi Motif Batik</h1>
    <p style='text-align:center; color:#cccccc;'>
    Unggah gambar batik, lalu sistem akan menampilkan <b>nama motif</b> dan <b>deskripsinya</b>.
    </p>
    """,
    unsafe_allow_html=True
)

# --- Upload Gambar ---
uploaded_file = st.file_uploader("📂 Pilih Gambar Batik", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Batik", use_column_width=True)

    # --- Prediksi ---
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    pred_class_idx = np.argmax(preds, axis=1)[0]
    pred_label = labels[pred_class_idx]

    # --- Ambil Deskripsi ---
    deskripsi = deskripsi_batik.get(pred_label.lower(), "Deskripsi belum tersedia untuk motif ini.")

    # --- Tampilan Card Hasil ---
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e1e2f, #27293d);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            max-width: 700px;
            margin: 30px auto;
        ">
            <h2 style='text-align:center; color:#ffffff; font-size:2em; margin-bottom:15px;'>
                🎨 {pred_label.replace('-', ' ').title()}
            </h2>
            <p style='font-size:16px; line-height:1.6; color:#dddddd; text-align:justify;'>
                {deskripsi}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("⬆️ Silakan unggah gambar batik untuk mendeteksi motif.")
