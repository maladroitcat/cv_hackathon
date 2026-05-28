from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
import torch
from PIL import Image

from src.transfer import build_transforms, load_torch_model


@dataclass
class PredictionResult:
    label: str
    confidence: float
    feedback: list
    features: Dict[str, float]
    overlay_bgr: np.ndarray


_MODEL_CACHE = {}


def _feedback(label: str, confidence: float):
    if label == "good_squat":
        return [
            "Form looks acceptable in this photo.",
            "Maintain balance, neutral spine, and controlled depth.",
        ]
    return [
        "Posture likely needs correction.",
        "Try a deeper squat while keeping chest more upright.",
        "Track knees steadily over toes and keep the stance stable.",
        f"Model confidence: {confidence:.2f} (review manually before action).",
    ]


def _get_model(model_path: str):
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = load_torch_model(model_path)
    return _MODEL_CACHE[model_path]


def predict_image(model_path: str, image_bgr: np.ndarray) -> PredictionResult:
    model, class_names, device = _get_model(model_path)
    _, eval_tf, _ = build_transforms(robust=False)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = eval_tf(pil).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    label = class_names[pred_idx]

    overlay = image_bgr.copy()
    cv2.putText(overlay, f"Prediction: {label}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Confidence: {conf:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)

    return PredictionResult(
        label=label,
        confidence=conf,
        feedback=_feedback(label, conf),
        features={},
        overlay_bgr=overlay,
    )
