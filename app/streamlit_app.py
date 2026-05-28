import numpy as np
import streamlit as st
from PIL import Image

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import predict_image

st.set_page_config(page_title="Robust Squat Posture Analyzer", layout="wide")
st.title("Robust Squat Posture Analyzer (Photo MVP)")
st.caption("Prototype only: side-view squat photos, not medical advice.")

model_choice = st.selectbox(
    "Model",
    options=["robust", "baseline"],
    help="Choose robust model (augmentation-focused) or baseline model.",
)

model_path = "models/robust_model.pt" if model_choice == "robust" else "models/baseline_model.pt"

uploaded = st.file_uploader("Upload a squat photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        pil_img = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(pil_img)
    except Exception:
        img_rgb = None

    if img_rgb is None:
        st.error("Could not read image.")
    elif not Path(model_path).exists():
        st.error(f"Model file not found: {model_path}. Train models first.")
    else:
        result = predict_image(model_path, img_rgb)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(img_rgb, use_container_width=True)

        with col2:
            st.subheader("Pose Overlay")
            st.image(result.overlay_bgr, use_container_width=True)

        st.subheader("Prediction")
        st.write(f"Class: **{result.label}**")
        st.write(f"Confidence: **{result.confidence:.2f}**")

        st.subheader("Suggestions")
        for item in result.feedback:
            st.write(f"- {item}")

        if result.features:
            st.subheader("Feature Snapshot")
            st.json({k: round(float(v), 4) if v == v else None for k, v in result.features.items()})
