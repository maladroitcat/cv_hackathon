import argparse

import pandas as pd

from common import ensure_parent_dir, save_json
from modeling import add_feature_noise, save_bundle, train_and_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Train robustness-focused squat classifier")
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics-out", default="results/metrics_robust.json")
    parser.add_argument("--noise-strength", type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    robust_df = add_feature_noise(df, strength=args.noise_strength, random_state=42)
    model, le, metrics, fcols = train_and_eval(robust_df, random_state=43)

    ensure_parent_dir(args.out)
    save_bundle(args.out, model, le, fcols)
    save_json(metrics, args.metrics_out)
    print(f"Saved robust model to {args.out}")
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
