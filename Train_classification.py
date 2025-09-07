#!/usr/bin/env python3
"""
Train_classification.py

Train a simple 3-class classifier that determines
whether the pig is interacting with an object and, if so, which side
('left' or 'right') based on per-frame CSV features.

Labeling rule used to build targets from the CSV:
- if HasMarker == True and "left"  in MarkerName.lower() -> "interacting_left"
- if HasMarker == True and "right" in MarkerName.lower() -> "interacting_right"
- else -> "not_interacting"

Expected CSV columns (some may be missing; we handle that safely):
    pig_cx, pig_cy, pig_w, pig_h,
    left_object_cx, left_object_cy, left_object_w, left_object_h,
    right_object_cx, right_object_cy, right_object_w, right_object_h,
    head_cx, head_cy, head_w, head_h,
    HasMarker, MarkerName

We also derive helpful distance features (e.g., head ↔ left/right object centers).
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ----------------------------
# Feature & label construction
# ----------------------------
BASE_FEATURES = [
    "pig_cx", "pig_cy", "pig_w", "pig_h",
    "left_object_cx", "left_object_cy", "left_object_w", "left_object_h",
    "right_object_cx", "right_object_cy", "right_object_w", "right_object_h",
    "head_cx", "head_cy", "head_w", "head_h",
]

DERIVED_FEATURES = [
    # distances from pig center to objects
    "dist_pig_to_left",
    "dist_pig_to_right",
    # distances from head center to objects
    "dist_head_to_left",
    "dist_head_to_right",
    # relative positions (object vs pig) in x/y
    "dx_head_left", "dy_head_left",
    "dx_head_right", "dy_head_right",
    "dx_pig_left", "dy_pig_left",
    "dx_pig_right", "dy_pig_right",
    # aspect features
    "pig_aspect", "head_aspect",
    "left_object_aspect", "right_object_aspect",
]


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    if any(pd.isna(v) for v in [a[0], a[1], b[0], b[1]]):
        return np.nan
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _aspect(w: float, h: float) -> float:
    if pd.isna(w) or pd.isna(h) or h == 0:
        return np.nan
    return float(w) / float(h)


def make_label(row: pd.Series) -> str:
    has = str(row.get("HasMarker", "")).lower()
    # Accept booleans or strings like 'true'/'false'
    has_marker = (has == "true") or (has == "1") or (has == "yes") or (row.get("HasMarker") is True)
    name = str(row.get("MarkerName", "")).lower()
    if has_marker:
        if "left" in name:
            return "interacting_left"
        if "right" in name:
            return "interacting_right"
    return "not_interacting"


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Ensure columns exist
    for col in BASE_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # Derived geometry
    df["dist_pig_to_left"] = df.apply(
        lambda r: _euclid((r["pig_cx"], r["pig_cy"]), (r["left_object_cx"], r["left_object_cy"])),
        axis=1,
    )
    df["dist_pig_to_right"] = df.apply(
        lambda r: _euclid((r["pig_cx"], r["pig_cy"]), (r["right_object_cx"], r["right_object_cy"])),
        axis=1,
    )
    df["dist_head_to_left"] = df.apply(
        lambda r: _euclid((r["head_cx"], r["head_cy"]), (r["left_object_cx"], r["left_object_cy"])),
        axis=1,
    )
    df["dist_head_to_right"] = df.apply(
        lambda r: _euclid((r["head_cx"], r["head_cy"]), (r["right_object_cx"], r["right_object_cy"])),
        axis=1,
    )

    # Deltas (signed)
    df["dx_head_left"] = df["head_cx"] - df["left_object_cx"]
    df["dy_head_left"] = df["head_cy"] - df["left_object_cy"]
    df["dx_head_right"] = df["head_cx"] - df["right_object_cx"]
    df["dy_head_right"] = df["head_cy"] - df["right_object_cy"]

    df["dx_pig_left"] = df["pig_cx"] - df["left_object_cx"]
    df["dy_pig_left"] = df["pig_cy"] - df["left_object_cy"]
    df["dx_pig_right"] = df["pig_cx"] - df["right_object_cx"]
    df["dy_pig_right"] = df["pig_cy"] - df["right_object_cy"]

    # Aspect ratios
    df["pig_aspect"] = df.apply(lambda r: _aspect(r["pig_w"], r["pig_h"]), axis=1)
    df["head_aspect"] = df.apply(lambda r: _aspect(r["head_w"], r["head_h"]), axis=1)
    df["left_object_aspect"] = df.apply(lambda r: _aspect(r["left_object_w"], r["left_object_h"]), axis=1)
    df["right_object_aspect"] = df.apply(lambda r: _aspect(r["right_object_w"], r["right_object_h"]), axis=1)

    all_feats = BASE_FEATURES + DERIVED_FEATURES
    return df[all_feats], all_feats


# ----------------------------
# Training / Prediction
# ----------------------------
def train(csv_path: str, outdir: str, test_size: float = 0.2, random_state: int = 42):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Build labels
    df["label"] = df.apply(make_label, axis=1)
    label_counts = df["label"].value_counts(dropna=False).to_dict()

    # Features
    X, feat_names = build_features(df.copy())
    y = df["label"]

    # Split
    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Pipeline: impute NaNs → RandomForest (no scaling needed)
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred, labels=["not_interacting", "interacting_left", "interacting_right"])

    # Save artifacts
    model_path = out / "interaction_cls_model.pkl"
    meta_path = out / "interaction_cls_meta.json"
    report_path = out / "report.txt"

    joblib.dump({
        "pipeline": pipeline,
        "features": feat_names,
        "labels": ["not_interacting", "interacting_left", "interacting_right"]
    }, model_path)

    with open(meta_path, "w") as f:
        json.dump({
            "csv": os.path.abspath(csv_path),
            "label_counts": label_counts,
            "features": feat_names
        }, f, indent=2)

    with open(report_path, "w") as f:
        f.write("Label counts:\n")
        f.write(json.dumps(label_counts, indent=2))
        f.write("\n\nClassification report:\n")
        f.write(report)
        f.write("\n\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(pd.DataFrame(cm,
                             index=["not_interacting", "interacting_left", "interacting_right"],
                             columns=["not_interacting", "interacting_left", "interacting_right"]).to_string())
        f.write("\n")

    print(f"[OK] Saved model → {model_path}")
    print(f"[OK] Saved meta  → {meta_path}")
    print(f"[OK] Saved report→ {report_path}")
    return model_path


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train the interaction classifier from a per-frame CSV")
    p_train = p.add_subparsers(dest="cmd")
    # Remove subparsers usage, keep only train parser
    p_train = p.add_parser("train", help="Train a classifier from a per-frame CSV")
    p_train.add_argument("--csv", required=True, help="Path to per-frame CSV")
    p_train.add_argument("--outdir", default="cls_runs", help="Where to save model & reports")
    p_train.add_argument("--test-size", type=float, default=0.2)

    args = p.parse_args()
    # Since only train subparser remains, if no subcommand given, print help and exit
    if not hasattr(args, "csv"):
        p.print_help()
        exit(1)
    return args


def main():
    args = parse_args()
    train(args.csv, args.outdir, test_size=args.test_size)


if __name__ == "__main__":
    # main()
    csv_ = "../ANSC/mark_frame/F003_per_frame.csv"
    outdir_ = "../ANSC/ANSC_pig/models/"
    train(csv_, outdir_)

'''
python ANSC_pig/Train_classification.py train \
  --csv /path/to/per_frame.csv \
  --outdir cls_runs
'''
