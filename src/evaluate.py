import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score

from common import ensure_parent_dir, save_json
from modeling import build_stress_test, load_bundle


def eval_model(bundle, df: pd.DataFrame):
    model = bundle["model"]
    le = bundle["label_encoder"]
    fcols = bundle["features"]
    x = df[fcols]
    y = le.transform(df["label"])
    pred = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classes": list(le.classes_),
    }


def plot_cm(cm, classes, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=classes)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline and robust models")
    parser.add_argument("--features", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--robust", required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument("--stress-noise", type=float, default=0.07)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    clean_test = df[df["split"].isin(["test", "test_clean"])]
    if clean_test.empty:
        raise ValueError("No test rows found (expected split == test or test_clean)")

    stress_df = build_stress_test(df, strength=args.stress_noise)
    stress_test = stress_df[stress_df["split"] == "test_stress"]

    base_bundle = load_bundle(args.baseline)
    robust_bundle = load_bundle(args.robust)

    baseline_clean = eval_model(base_bundle, clean_test)
    robust_clean = eval_model(robust_bundle, clean_test)
    baseline_stress = eval_model(base_bundle, stress_test)
    robust_stress = eval_model(robust_bundle, stress_test)

    summary = {
        "clean": {"baseline": baseline_clean, "robust": robust_clean},
        "stress": {"baseline": baseline_stress, "robust": robust_stress},
    }

    save_json(summary, f"{args.out}/metrics_comparison.json")

    plot_cm(baseline_stress["confusion_matrix"], baseline_stress["classes"], "Baseline - Stress Test", f"{args.out}/confusion_baseline_stress.png")
    plot_cm(robust_stress["confusion_matrix"], robust_stress["classes"], "Robust - Stress Test", f"{args.out}/confusion_robust_stress.png")

    print(json.dumps(summary, indent=2))
    print(f"Saved comparison outputs under {args.out}/")


if __name__ == "__main__":
    main()
