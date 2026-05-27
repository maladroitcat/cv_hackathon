import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def feature_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith("emb_")]
    if not cols:
        raise ValueError("No emb_* columns found in features dataframe")
    return cols


def split_df(df: pd.DataFrame):
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"].isin(["test", "test_clean"])].copy()
    return train, val, test


def build_pipeline(random_state: int = 42) -> Pipeline:
    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=14,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("clf", clf)])


def train_and_eval(df: pd.DataFrame, random_state: int = 42):
    fcols = feature_columns(df)
    train, val, test = split_df(df)
    le = LabelEncoder()
    y_train = le.fit_transform(train["label"])
    y_val = le.transform(val["label"]) if not val.empty else None
    y_test = le.transform(test["label"]) if not test.empty else None

    model = build_pipeline(random_state=random_state)
    model.fit(train[fcols], y_train)

    outputs = {}
    for name, x, y in [("val", val, y_val), ("test", test, y_test)]:
        if x.empty or y is None:
            continue
        pred = model.predict(x[fcols])
        outputs[name] = {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "classification_report": classification_report(y, pred, target_names=list(le.classes_), output_dict=True),
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
        }

    return model, le, outputs, fcols


def save_bundle(path: str, model: Pipeline, label_encoder: LabelEncoder, features: list[str]):
    joblib.dump({"model": model, "label_encoder": label_encoder, "features": features}, path)


def load_bundle(path: str):
    return joblib.load(path)


def add_feature_noise(df: pd.DataFrame, strength: float = 0.03, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = df.copy()
    fcols = feature_columns(out)
    mask = out["split"] == "train"
    noisy = out.loc[mask, fcols].copy()
    for col in fcols:
        std = noisy[col].std(skipna=True)
        if pd.isna(std) or std == 0:
            continue
        noisy[col] = noisy[col] + rng.normal(0.0, std * strength, size=len(noisy))
    out.loc[mask, fcols] = noisy
    return out


def build_stress_test(df: pd.DataFrame, strength: float = 0.05, random_state: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = df.copy()
    fcols = feature_columns(out)
    mask = out["split"].isin(["test", "test_clean"])
    stress = out.loc[mask, fcols].copy()
    for col in fcols:
        std = stress[col].std(skipna=True)
        if pd.isna(std) or std == 0:
            continue
        stress[col] = stress[col] + rng.normal(0.0, std * strength, size=len(stress))
    out.loc[mask, fcols] = stress
    out.loc[mask, "split"] = "test_stress"
    return out[mask | (out["split"].isin(["train", "val"]))].copy()
