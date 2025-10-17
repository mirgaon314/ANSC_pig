

#!/usr/bin/env python3
"""
accuracy_check.py — Simple accuracy analysis for interaction predictions

Computes:
  • overall accuracy
  • not_interacting accuracy (accuracy on GT==not_interacting)
  • left accuracy (accuracy on GT==interacting_left)
  • right accuracy (accuracy on GT==interacting_right)
Also writes a 3×3 confusion matrix and a CSV of mismatches.

Usage (paths have defaults so you can just Run):
  python accuracy_check.py \
    --pred ../ANSC/detected/B004_pred_coor.csv \
    --gt   ../ANSC/mark_frame/B004_per_frame.csv \
    --outdir ../ANSC/detected/accuracy_B004

You can change the defaults below or pass different files via CLI.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------
# Defaults for one-click
# -----------------------
DEFAULT_PRED = Path(__file__).parent / ".." / "detected" / "B004_pred_coor.csv"
DEFAULT_GT   = Path(__file__).parent / ".." / "mark_frame" / "B004_per_frame.csv"
DEFAULT_OUT  = Path(__file__).parent / ".." / "detected" / "accuracy_B004"


def normalize_pred_label(s: str) -> str:
    s = (s or "").strip().lower()
    if "left" in s:  # matches 'interacting_left', 'left', etc.
        return "interacting_left"
    if "right" in s:
        return "interacting_right"
    if "not" in s:
        return "not_interacting"
    # fallback: unknown → not_interacting
    return "not_interacting"


def gt_label_from_row(r: pd.Series) -> str:
    has = str(r.get("HasMarker", "")).strip().lower()
    has_marker = has in {"true", "1", "yes"} or r.get("HasMarker") is True
    name = str(r.get("MarkerName", "")).strip().lower()
    # Ignore explicit trial start/end markers; treat as not_interacting
    if has_marker and ("left" in name):
        return "interacting_left"
    if has_marker and ("right" in name):
        return "interacting_right"
    return "not_interacting"


def pick_time_column(df: pd.DataFrame):
    for c in ["time_sec", "time", "timestamp"]:
        if c in df.columns:
            return c
    return None


def pick_frame_column(df: pd.DataFrame):
    for c in ["frame", "Frame", "frame_index", "idx"]:
        if c in df.columns:
            return c
    return None


def merge_pred_gt(pred: pd.DataFrame, gt: pd.DataFrame, tol_sec: float = 0.05) -> pd.DataFrame:
    # Try frame-based merge first
    pc = pick_frame_column(pred)
    gc = pick_frame_column(gt)
    if pc and gc:
        pred = pred.copy(); gt = gt.copy()
        pred["__frame__"] = pd.to_numeric(pred[pc], errors="coerce").fillna(0).astype(int)
        gt["__frame__"]   = pd.to_numeric(gt[gc], errors="coerce").fillna(0).astype(int)
        merged = gt.merge(pred, left_on="__frame__", right_on="__frame__", how="inner", suffixes=("_gt","_pred"))
        merged.rename(columns={"__frame__":"frame"}, inplace=True)
        return merged

    # Else try time-based nearest merge
    pt = pick_time_column(pred)
    gt_t = pick_time_column(gt)
    if pt and gt_t:
        pred = pred.copy(); gt = gt.copy()
        pred["__t__"] = pd.to_numeric(pred[pt], errors="coerce").ffill().fillna(0.0)
        gt["__t__"]   = pd.to_numeric(gt[gt_t], errors="coerce").ffill().fillna(0.0)
        pred.sort_values("__t__", inplace=True)
        gt.sort_values("__t__", inplace=True)
        merged = pd.merge_asof(gt, pred, on="__t__", direction="nearest", tolerance=tol_sec, suffixes=("_gt","_pred"))
        merged.rename(columns={"__t__":"time_sec"}, inplace=True)
        # Drop rows that failed to find a near match
        merged = merged.dropna(subset=[col for col in merged.columns if col.endswith("_pred")], how="all")
        return merged

    # Fallback: index-merge
    pred = pred.copy(); gt = gt.copy()
    pred["__i__"] = np.arange(len(pred))
    gt["__i__"]   = np.arange(len(gt))
    merged = gt.merge(pred, on="__i__", how="inner", suffixes=("_gt","_pred"))
    merged.rename(columns={"__i__":"row_index"}, inplace=True)
    return merged


def pick_pred_label_column(pred_cols):
    # Try these candidates in order
    candidates = [
        "pred_label", "label", "state", "prediction", "class", "cls"
    ]
    for c in candidates:
        if c in pred_cols:
            return c
    # last resort: find any column containing 'label' or 'state'
    for c in pred_cols:
        lc = c.lower()
        if "label" in lc or "state" in lc:
            return c
    return None


def compute_metrics(merged: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    # Normalize GT labels
    gt_labels = merged.apply(gt_label_from_row, axis=1)

    # Normalize PRED labels
    pred_col = pick_pred_label_column(merged.columns)
    if pred_col is None:
        raise SystemExit("Could not find a prediction label column (looked for pred_label/label/state/prediction/class/cls)")
    pred_labels = merged[pred_col].astype(str).map(normalize_pred_label)

    # Build 3-class set
    classes = ["not_interacting", "interacting_left", "interacting_right"]
    lab2id = {c: i for i, c in enumerate(classes)}
    g = gt_labels.map(lab2id).fillna(0).astype(int).values
    p = pred_labels.map(lab2id).fillna(0).astype(int).values

    N = len(g)
    overall_acc = float((g == p).mean()) if N else 0.0

    def acc_for(label: str) -> float:
        idx = lab2id[label]
        mask = (g == idx)
        if mask.sum() == 0:
            return float("nan")
        return float((p[mask] == idx).mean())

    acc_not   = acc_for("not_interacting")
    acc_left  = acc_for("interacting_left")
    acc_right = acc_for("interacting_right")

    # Confusion matrix 3x3
    cm = np.zeros((3,3), dtype=int)
    for gi, pi in zip(g, p):
        cm[gi, pi] += 1
    cm_df = pd.DataFrame(cm, index=[f"gt_{c}" for c in classes], columns=[f"pd_{c}" for c in classes])

    metrics = {
        "overall_accuracy": overall_acc,
        "not_interacting_accuracy": acc_not,
        "left_accuracy": acc_left,
        "right_accuracy": acc_right,
        "n_frames": int(N),
    }
    return metrics, cm_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(DEFAULT_PRED), help="prediction CSV path")
    ap.add_argument("--gt",   default=str(DEFAULT_GT), help="ground-truth per-frame CSV path")
    ap.add_argument("--outdir", default=str(DEFAULT_OUT), help="output directory for reports")
    ap.add_argument("--tol", type=float, default=0.05, help="time tolerance (sec) for nearest-join when using time columns")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.pred)
    gt   = pd.read_csv(args.gt)

    merged = merge_pred_gt(pred, gt, tol_sec=args.tol)
    if merged.empty:
        raise SystemExit("Merged dataframe is empty — check that your files share a usable join key (frame or time_sec).")

    metrics, cm_df = compute_metrics(merged)

    # Mismatches CSV (for quick QA)
    classes = ["not_interacting", "interacting_left", "interacting_right"]
    lab2id = {c: i for i, c in enumerate(classes)}
    gt_norm  = merged.apply(gt_label_from_row, axis=1)
    pred_col = pick_pred_label_column(merged.columns)
    pred_norm = merged[pred_col].astype(str).map(normalize_pred_label)
    mism = merged.loc[gt_norm != pred_norm].copy()
    mism = mism.assign(gt_label=gt_norm, pred_label=pred_norm)

    # Save
    pd.DataFrame([metrics]).to_csv(outdir / "metrics.csv", index=False)
    cm_df.to_csv(outdir / "confusion_3class.csv")
    mism_cols = [c for c in ["frame","Frame","time_sec","time","HasMarker","MarkerName","gt_label","pred_label"] if c in mism.columns] + [c for c in mism.columns if c.endswith("_cx") or c.endswith("_cy")]
    mism[mism_cols].to_csv(outdir / "mismatches.csv", index=False)

    # Print summary
    print("\n=== Accuracy Summary ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:28s}: {v:.4f}")
        else:
            print(f"{k:28s}: {v}")
    print("\nConfusion matrix saved to:", (outdir / "confusion_3class.csv").resolve())
    print("Mismatches saved to:", (outdir / "mismatches.csv").resolve())

if __name__ == "__main__":
    main()