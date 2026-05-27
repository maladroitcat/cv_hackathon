import argparse

import pandas as pd

from common import ensure_parent_dir, save_json
from modeling import save_bundle, train_and_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline squat classifier")
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics-out", default="results/metrics_baseline.json")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    model, le, metrics, fcols = train_and_eval(df, random_state=42)

    ensure_parent_dir(args.out)
    save_bundle(args.out, model, le, fcols)
    save_json(metrics, args.metrics_out)
    print(f"Saved baseline model to {args.out}")
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
