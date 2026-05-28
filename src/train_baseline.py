import argparse

from common import save_json
from transfer import save_torch_bundle, train_transfer_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline transfer-learning squat classifier")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--out", default="models/baseline_model.pt")
    parser.add_argument("--metrics-out", default="results/metrics_baseline.json")
    parser.add_argument("--epochs-head", type=int, default=4)
    parser.add_argument("--epochs-ft", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    bundle, metrics = train_transfer_model(
        labels_csv=args.labels,
        robust=False,
        epochs_head=args.epochs_head,
        epochs_ft=args.epochs_ft,
        batch_size=args.batch_size,
        seed=42,
    )

    save_torch_bundle(args.out, bundle)
    save_json(metrics, args.metrics_out)
    print(f"Saved baseline model to {args.out}")
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
