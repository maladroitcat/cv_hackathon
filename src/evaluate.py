import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from common import save_json
from transfer import CLASS_NAMES, SquatImageDataset, build_transforms, evaluate_model, load_torch_model, split_labels_df


def plot_cm(cm, classes, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=classes)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def eval_checkpoint(model_path: str, labels_csv: str, batch_size: int):
    _, eval_tf, stress_tf = build_transforms(robust=False)
    _, _, test_df = split_labels_df(labels_csv)
    clean_loader = DataLoader(SquatImageDataset(test_df, eval_tf), batch_size=batch_size, shuffle=False)
    stress_loader = DataLoader(SquatImageDataset(test_df, stress_tf), batch_size=batch_size, shuffle=False)

    model, _, device = load_torch_model(model_path)
    clean = evaluate_model(model, clean_loader, device)
    stress = evaluate_model(model, stress_loader, device)
    return {"clean": clean, "stress": stress}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline and robust models")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--baseline", default="models/baseline_model.pt")
    parser.add_argument("--robust", default="models/robust_model.pt")
    parser.add_argument("--out", default="results")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    base = eval_checkpoint(args.baseline, args.labels, args.batch_size)
    rob = eval_checkpoint(args.robust, args.labels, args.batch_size)

    summary = {
        "clean": {"baseline": base["clean"], "robust": rob["clean"]},
        "stress": {"baseline": base["stress"], "robust": rob["stress"]},
    }

    save_json(summary, f"{args.out}/metrics_comparison.json")

    plot_cm(summary["stress"]["baseline"]["confusion_matrix"], CLASS_NAMES, "Baseline - Stress Test", f"{args.out}/confusion_baseline_stress.png")
    plot_cm(summary["stress"]["robust"]["confusion_matrix"], CLASS_NAMES, "Robust - Stress Test", f"{args.out}/confusion_robust_stress.png")

    print(summary)


if __name__ == "__main__":
    main()
