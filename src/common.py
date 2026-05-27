import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"image_path", "label", "split"}


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_labels(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in labels CSV: {missing}")
    return df


def normalize_path(base_dir: str, image_path: str) -> str:
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    return str(Path(base_dir) / image_path)


def parse_args_base(description: str):
    return argparse.ArgumentParser(description=description)


def save_json(payload: dict, path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def np_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan
