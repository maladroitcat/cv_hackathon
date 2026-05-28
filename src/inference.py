from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from PIL import Image, ImageDraw

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


def predict_image(model_path: str, image_rgb: np.ndarray) -> PredictionResult:
    model, class_names, device = _get_model(model_path)
    _, eval_tf, _ = build_transforms(robust=False)

    pil = Image.fromarray(image_rgb)
    x = eval_tf(pil).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    label = class_names[pred_idx]

    overlay_pil = pil.copy()
    draw = ImageDraw.Draw(overlay_pil)
    draw.rectangle((12, 12, 430, 76), fill=(0, 0, 0))
    draw.text((20, 20), f"Prediction: {label}", fill=(120, 255, 120))
    draw.text((20, 45), f"Confidence: {conf:.2f}", fill=(120, 255, 120))
    overlay = np.array(overlay_pil)

    return PredictionResult(
        label=label,
        confidence=conf,
        feedback=_feedback(label, conf),
        features={},
        overlay_bgr=overlay,
    )
