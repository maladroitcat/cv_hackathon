from dataclasses import dataclass
from typing import Dict

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models


@dataclass
class PredictionResult:
    label: str
    confidence: float
    feedback: list
    features: Dict[str, float]
    overlay_bgr: np.ndarray


def _embed_image(image_bgr: np.ndarray):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    weights = models.ResNet18_Weights.DEFAULT
    tfm = weights.transforms()
    base = models.resnet18(weights=weights)
    enc = torch.nn.Sequential(*list(base.children())[:-1])
    enc.eval()

    with torch.inference_mode():
        x = tfm(pil).unsqueeze(0)
        emb = enc(x).squeeze().numpy()
    return emb


def _feedback(label: str, confidence: float):
    if label == "good_squat":
        return ["Form looks acceptable in this photo.", "Maintain balance, neutral spine, and controlled depth."]
    return [
        "Posture likely needs correction.",
        "Try a deeper squat while keeping chest more upright.",
        "Track knees steadily over toes and keep the stance stable.",
        f"Model confidence: {confidence:.2f} (review manually before action).",
    ]


def predict_image(model_path: str, image_bgr: np.ndarray) -> PredictionResult:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    le = bundle["label_encoder"]
    fcols = bundle["features"]

    emb = _embed_image(image_bgr)
    row = {c: float(emb[int(c.split("_")[1])]) for c in fcols}
    x = pd.DataFrame([row])

    pred_idx = int(model.predict(x)[0])
    label = str(le.inverse_transform([pred_idx])[0])
    if hasattr(model, "predict_proba"):
        conf = float(np.max(model.predict_proba(x)[0]))
    else:
        conf = 0.5

    overlay = image_bgr.copy()
    cv2.putText(overlay, f"Prediction: {label}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Confidence: {conf:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)

    return PredictionResult(label=label, confidence=conf, feedback=_feedback(label, conf), features={}, overlay_bgr=overlay)
