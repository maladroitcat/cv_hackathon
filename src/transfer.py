import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = ["good_squat", "needs_correction"]


class SquatImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.transform(img)
        y = self.class_to_idx[row["label"]]
        return x, y


def build_transforms(robust: bool = False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if robust:
        train_tf = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.25),
                transforms.RandomRotation(12),
                transforms.RandomPerspective(distortion_scale=0.18, p=0.3),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    stress_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ColorJitter(brightness=0.35, contrast=0.35),
            transforms.GaussianBlur(kernel_size=5),
            transforms.RandomRotation(15),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_tf, eval_tf, stress_tf


def split_labels_df(labels_csv: str):
    df = pd.read_csv(labels_csv)
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"].isin(["test", "test_clean"])].copy()
    return train, val, test


def build_model(num_classes: int = 2, arch: str = "resnet18"):
    arch = arch.lower()
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_trainable(model, mode: str):
    for p in model.parameters():
        p.requires_grad = False
    if mode == "head":
        for p in model.fc.parameters():
            p.requires_grad = True
    elif mode == "finetune":
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    losses = []
    all_y = []
    all_p = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        pred = logits.argmax(dim=1)
        all_y.extend(y.cpu().numpy().tolist())
        all_p.extend(pred.cpu().numpy().tolist())

    acc = accuracy_score(all_y, all_p) if all_y else 0.0
    return float(np.mean(losses) if losses else 0.0), float(acc)


def evaluate_model(model, loader, device):
    model.eval()
    all_y = []
    all_p = []

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            all_y.extend(y.cpu().numpy().tolist())
            all_p.extend(pred.cpu().numpy().tolist())

    return {
        "accuracy": float(accuracy_score(all_y, all_p)),
        "macro_f1": float(f1_score(all_y, all_p, average="macro")),
        "classification_report": classification_report(all_y, all_p, target_names=CLASS_NAMES, output_dict=True),
        "confusion_matrix": confusion_matrix(all_y, all_p).tolist(),
        "classes": CLASS_NAMES,
    }


def train_transfer_model(
    labels_csv: str,
    robust: bool,
    epochs_head: int,
    epochs_ft: int,
    batch_size: int,
    seed: int = 42,
    arch: str = "resnet18",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df = split_labels_df(labels_csv)
    train_tf, eval_tf, stress_tf = build_transforms(robust=robust)

    train_loader = DataLoader(SquatImageDataset(train_df, train_tf), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SquatImageDataset(val_df, eval_tf), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SquatImageDataset(test_df, eval_tf), batch_size=batch_size, shuffle=False)
    stress_loader = DataLoader(SquatImageDataset(test_df, stress_tf), batch_size=batch_size, shuffle=False)

    model = build_model(num_classes=2, arch=arch).to(device)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0

    set_trainable(model, "head")
    opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    for _ in range(epochs_head):
        run_epoch(model, train_loader, criterion, opt, device, train=True)
        _, val_acc = run_epoch(model, val_loader, criterion, opt, device, train=False)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # Optional fine-tuning stage intentionally disabled for strict head-only transfer learning.
    # Keep backbone frozen and train only the replaced classification head.

    if best_state is not None:
        model.load_state_dict(best_state)

    clean_metrics = evaluate_model(model, test_loader, device)
    stress_metrics = evaluate_model(model, stress_loader, device)

    bundle = {
        "state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "arch": arch.lower(),
    }

    return bundle, {"clean": clean_metrics, "stress": stress_metrics}


def save_torch_bundle(path: str, bundle: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


def load_torch_model(path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = build_model(num_classes=len(ckpt["class_names"]), arch=ckpt.get("arch", "resnet18"))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["class_names"], device
