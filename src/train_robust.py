import argparse

from common import save_json
from transfer import save_torch_bundle, train_transfer_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train robust transfer-learning squat classifier")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--out", default="models/robust_model.pt")
    parser.add_argument("--metrics-out", default="results/metrics_robust.json")
    parser.add_argument("--epochs-head", type=int, default=4)
    parser.add_argument("--epochs-ft", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--arch", choices=["resnet18", "resnet50"], default="resnet18")
    args = parser.parse_args()

    bundle, metrics = train_transfer_model(
        labels_csv=args.labels,
        robust=True,
        epochs_head=args.epochs_head,
        epochs_ft=args.epochs_ft,
        batch_size=args.batch_size,
        seed=43,
        arch=args.arch,
    )

    save_torch_bundle(args.out, bundle)
    save_json(metrics, args.metrics_out)
    print(f"Saved robust model to {args.out}")
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
