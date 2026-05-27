import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

from common import ensure_parent_dir, load_labels, normalize_path


IMG_SIZE = 224


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract transfer-learning embeddings from squat photos")
    parser.add_argument("--labels", required=True, help="Path to labels CSV")
    parser.add_argument("--base-dir", default=".", help="Base directory for relative image paths")
    parser.add_argument("--out", required=True, help="Output CSV path")
    return parser


def build_encoder(device: torch.device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    encoder = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    encoder.eval()
    tfm = weights.transforms()
    return encoder, tfm


def main() -> None:
    args = build_parser().parse_args()
    labels = load_labels(args.labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, tfm = build_encoder(device)

    rows = []
    with torch.inference_mode():
        for _, row in labels.iterrows():
            full_path = normalize_path(args.base_dir, row["image_path"])
            if not Path(full_path).exists():
                continue
            try:
                img = Image.open(full_path).convert("RGB")
                x = tfm(img).unsqueeze(0).to(device)
                emb = encoder(x).squeeze().detach().cpu().numpy().astype(np.float32)
            except Exception:
                continue

            out = {
                "image_path": row["image_path"],
                "label": row["label"],
                "split": row["split"],
            }
            for i, v in enumerate(emb):
                out[f"emb_{i}"] = float(v)
            rows.append(out)

    df = pd.DataFrame(rows)
    ensure_parent_dir(args.out)
    df.to_csv(args.out, index=False)
    print(f"Saved embeddings to {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
