import argparse
from pathlib import Path

import albumentations as A
import cv2
import pandas as pd

from common import ensure_parent_dir, load_labels, normalize_path


AUG = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.35),
        A.MotionBlur(blur_limit=7, p=0.25),
        A.Rotate(limit=12, p=0.5),
        A.Perspective(scale=(0.02, 0.06), p=0.35),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.35),
        A.ImageCompression(quality_range=(30, 70), p=0.4),
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.08, 0.18), hole_width_range=(0.08, 0.18), p=0.25),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate augmented training images and updated labels")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-labels", required=True)
    parser.add_argument("--copies", type=int, default=2)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    out_rows = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in labels.iterrows():
        rel_path = row["image_path"]
        abs_path = normalize_path(args.base_dir, rel_path)
        img = cv2.imread(abs_path)
        if img is None:
            continue

        out_rows.append(row.to_dict())

        if row["split"] != "train":
            continue

        stem = Path(rel_path).stem
        suffix = Path(rel_path).suffix or ".jpg"
        for i in range(args.copies):
            aug = AUG(image=img)["image"]
            out_name = f"{stem}__aug{i}{suffix}"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), aug)

            new_row = row.to_dict()
            new_row["image_path"] = str(out_path)
            out_rows.append(new_row)

    out_df = pd.DataFrame(out_rows)
    ensure_parent_dir(args.out_labels)
    out_df.to_csv(args.out_labels, index=False)
    print(f"Saved augmented labels to {args.out_labels} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
