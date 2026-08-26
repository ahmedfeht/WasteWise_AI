"""WasteWise AI Streamlit application."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import tensorflow as tf

from wastewise.inference import CLASS_NAMES, find_model_path, get_recommendation, load_rgb_image, predict_image

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = find_model_path(PROJECT_ROOT)
CLASS_PATH = PROJECT_ROOT / "artifacts" / "class_names.json"

st.set_page_config(page_title="WasteWise AI", page_icon="♻️", layout="wide")

st.markdown(
    """
    <style>
    .hero {padding: 1.3rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#14532d,#22c55e); color:white; margin-bottom:1rem;}
    .hero h1 {margin:0; font-size:2.4rem;}
    .hero p {margin:.35rem 0 0 0; font-size:1.05rem;}
    .result {padding:1rem 1.2rem; border-left:6px solid #16a34a; border-radius:10px; background:#ecfdf3; color:#17351f;}
    </style>
    <div class="hero">
      <h1>♻️ WasteWise AI</h1>
      <p>Upload one waste image and receive its predicted category, confidence and recycling guidance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_classifier() -> tuple[tf.keras.Model, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model is missing. Run data_preprocessing.py and train_model.py first."
        )
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    labels = CLASS_NAMES
    if CLASS_PATH.exists():
        labels = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
    return model, labels


with st.sidebar:
    st.header("About the project")
    st.write("A six-class waste classifier created with transfer learning.")
    st.markdown("**Model:** MobileNetV2  ")
    st.markdown("**Input:** 224 × 224 RGB image  ")
    st.markdown("**Classes:** Cardboard, Glass, Metal, Paper, Plastic, Trash")
    confidence_notice = st.slider("Minimum confidence notice", 0.40, 0.90, 0.60, 0.05)
    st.caption("Predictions support sorting decisions. Always follow local recycling rules.")

uploaded_file = st.file_uploader(
    "Upload a clear waste image", type=["jpg", "jpeg", "png", "webp"], help="Use one main object and good lighting."
)

if uploaded_file is None:
    st.info("Upload an image to begin. The application does not permanently store uploaded images.")
else:
    try:
        image = load_rgb_image(uploaded_file)
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.subheader("Uploaded image")
            st.image(image, use_container_width=True)
        with right:
            st.subheader("AI classification")
            if st.button("Classify Waste", type="primary", use_container_width=True):
                with st.spinner("MobileNetV2 is analysing the image..."):
                    model, labels = load_classifier()
                    ranked = predict_image(model, image, labels)
                best = ranked[0]
                category = str(best["category"])
                confidence = float(best["confidence"])
                st.markdown(
                    f'<div class="result"><h2>{category.title()}</h2><p>Confidence: <strong>{confidence:.1%}</strong></p></div>',
                    unsafe_allow_html=True,
                )
                st.progress(confidence)
                if confidence < confidence_notice:
                    st.warning("The confidence is low. Try a clearer image with one object and a simple background.")
                st.success(f"Recommendation: {get_recommendation(category)}")

                top_three = pd.DataFrame(ranked[:3])
                top_three["category"] = top_three["category"].str.title()
                top_three["confidence_percent"] = (top_three["confidence"] * 100).round(2)
                st.markdown("#### Top three predictions")
                st.bar_chart(top_three.set_index("category")["confidence_percent"], horizontal=True)
                st.dataframe(
                    top_three[["category", "confidence_percent"]].rename(
                        columns={"category": "Category", "confidence_percent": "Confidence (%)"}
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
    except Exception as error:
        st.error(f"The image could not be classified: {error}")

st.divider()
st.caption("BIT4443 Deep Learning Group Project • Streamlit + TensorFlow/Keras + MobileNetV2")

