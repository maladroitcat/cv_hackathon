import argparse

import pandas as pd

from common import ensure_parent_dir


def main():
    parser = argparse.ArgumentParser(description="Validate and export embedding features")
    parser.add_argument("--in", dest="in_path", required=True, help="Input embeddings CSV")
    parser.add_argument("--out", required=True, help="Output features CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if df.empty or not emb_cols:
        raise ValueError("Input is empty or has no embedding columns.")

    keep_cols = ["image_path", "label", "split"] + emb_cols
    feat = df[keep_cols].copy()

    ensure_parent_dir(args.out)
    feat.to_csv(args.out, index=False)
    print(f"Saved features to {args.out} ({len(feat)} rows, {len(emb_cols)} dims)")


if __name__ == "__main__":
    main()
