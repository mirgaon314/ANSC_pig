from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.inference import FolderSequenceDataset
from src.model import InteractionClassifier

DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class SubjectInfo:
    name: str
    root: Path
    csv_path: Path
    img_dir: Path


def get_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> InteractionClassifier:
    model = InteractionClassifier(in_channels=5, temporal_length=5)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def discover_subjects(root_dir: Path, subjects_filter: Optional[Sequence[str]] = None) -> List[SubjectInfo]:
    subjects: List[SubjectInfo] = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if subjects_filter and name not in subjects_filter:
            continue
        img_dir = child / "img"
        if not img_dir.exists():
            continue
        # Find CSV matching <name>_per_frame_with_crops.csv
        csv_candidates = list(child.glob(f"{name}_per_frame_with_crops.csv"))
        if not csv_candidates:
            # fallback: any *_per_frame_with_crops.csv
            csv_candidates = list(child.glob("*_per_frame_with_crops.csv"))
        if not csv_candidates:
            continue
        csv_path = csv_candidates[0]
        subjects.append(SubjectInfo(name=name, root=child, csv_path=csv_path, img_dir=img_dir))
    return subjects


def collect_image_paths(images_dir: Path, extensions: Sequence[str]) -> List[Path]:
    ext_set = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    image_paths = [
        p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in ext_set
    ]
    return image_paths


def build_label_map(csv_path: Path) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    df["MarkerName"] = df["MarkerName"].fillna("").str.lower()
    df["HasMarker"] = df["HasMarker"].astype(bool)
    df["is_interaction"] = df["HasMarker"] & df["MarkerName"].isin({"left", "right"})
    # Use crop_image basename as key
    label_map: Dict[str, int] = {
        Path(str(row["crop_image"]).strip()).name: int(row["is_interaction"]) for _, row in df.iterrows()
    }
    return label_map


def run_predictions(
    model: InteractionClassifier,
    image_paths: List[Path],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> List[Tuple[str, float]]:
    dataset = FolderSequenceDataset(image_paths=image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    sigmoid = torch.nn.Sigmoid()
    outputs: List[Tuple[str, float]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict", leave=False):
            inputs = batch["inputs"].to(device, non_blocking=True)
            logits, _ = model(inputs)
            probs = sigmoid(logits).detach().cpu().numpy().tolist()
            centre_paths = batch["centre_path"]
            if isinstance(centre_paths, str):
                centre_paths = [centre_paths]
            outputs.extend(zip(centre_paths, probs))
    return outputs


# Metrics utilities

def compute_basic_stats(preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    bin_preds = (preds >= threshold).astype(np.int32)
    tp = int(((bin_preds == 1) & (labels == 1)).sum())
    tn = int(((bin_preds == 0) & (labels == 0)).sum())
    fp = int(((bin_preds == 1) & (labels == 0)).sum())
    fn = int(((bin_preds == 0) & (labels == 1)).sum())
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = (tpr + tnr) / 2.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _roc_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Requires both classes present
    if labels.min() == labels.max():
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    # Sort by descending score
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    P = labels.sum()
    N = len(labels) - P
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = tps / (P if P > 0 else 1)
    fpr = fps / (N if N > 0 else 1)
    # Prepend origin
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    return fpr, tpr


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    if labels.min() == labels.max():
        return None
    fpr, tpr = _roc_curve(scores, labels)
    # Trapezoidal rule
    return float(np.trapz(tpr, fpr))


def _precision_recall_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Sort by descending score
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(labels.sum(), 1)
    # Prepend start point
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return precision, recall


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    if labels.min() == labels.max():
        return None
    precision, recall = _precision_recall_curve(scores, labels)
    # Integrate precision over recall
    return float(np.trapz(precision, recall))


def write_rows_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_for_subject(
    model: InteractionClassifier,
    subject: SubjectInfo,
    device: torch.device,
    output_root: Path,
    extensions: Sequence[str],
    batch_size: int,
    num_workers: int,
    threshold: float,
) -> Dict[str, object]:
    image_paths = collect_image_paths(subject.img_dir, extensions)
    label_map = build_label_map(subject.csv_path)

    preds_list = run_predictions(model, image_paths, device, batch_size, num_workers)

    # Build per-frame records with labels
    records: List[Dict[str, object]] = []
    y_true: List[int] = []
    y_score: List[float] = []
    for path_str, prob in preds_list:
        file_name = Path(path_str).name
        label = label_map.get(file_name, None)
        rec: Dict[str, object] = {
            "subject": subject.name,
            "frame_path": path_str,
            "frame_file": file_name,
            "probability": float(prob),
            "prediction": int(prob >= threshold),
        }
        if label is not None:
            rec["label"] = int(label)
            y_true.append(int(label))
            y_score.append(float(prob))
        records.append(rec)

    # Write per-subject predictions CSV inside subject folder
    per_pred_csv = subject.root / f"{subject.name}_predictions.csv"
    write_rows_csv(per_pred_csv, records, fieldnames=list(records[0].keys()))

    # Compute metrics (only on rows where label exists)
    if y_true:
        labels_arr = np.array(y_true, dtype=np.int32)
        scores_arr = np.array(y_score, dtype=np.float32)
        stats = compute_basic_stats(scores_arr, labels_arr, threshold)
        stats["roc_auc"] = roc_auc(scores_arr, labels_arr)
        stats["pr_auc"] = pr_auc(scores_arr, labels_arr)
    else:
        stats = {k: None for k in [
            "accuracy","precision","recall","f1","balanced_accuracy","tp","tn","fp","fn","roc_auc","pr_auc"
        ]}

    # Write per-subject metrics CSV inside subject folder
    metrics_row = {
        "subject": subject.name,
        "num_frames": len(records),
        "num_labeled": len(y_true),
        **{k: (v if v is None else float(v)) for k, v in stats.items()},
    }
    per_metrics_csv = subject.root / f"{subject.name}_metrics.csv"
    write_rows_csv(per_metrics_csv, [metrics_row], fieldnames=list(metrics_row.keys()))

    # Also return for overall aggregation
    return metrics_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch predict across multiple subjects")
    parser.add_argument("--root-dir", type=Path, required=True, help="Root directory containing subject folders")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/full_run/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("predictions/batch"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--extensions", type=str, nargs="*", default=sorted(DEFAULT_EXTENSIONS))
    parser.add_argument("--subjects", type=str, nargs="*", default=None, help="Optional list of subject names to include")
    parser.add_argument("--prefer-mps", action="store_true")
    parser.add_argument("--no-mps", action="store_true")
    parser.add_argument("--max-subjects", type=int, default=0, help="Debug: limit number of subjects")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefer_mps = args.prefer_mps or not args.no_mps
    device = get_device(prefer_mps=prefer_mps)
    print(f"Using device: {device}")

    subjects = discover_subjects(args.root_dir, args.subjects)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]
        print(f"Limiting to {len(subjects)} subjects for debug mode")
    if not subjects:
        raise SystemExit(f"No valid subjects found under {args.root_dir}")

    model = load_checkpoint(args.checkpoint, device)

    overall_rows: List[Dict[str, object]] = []
    for subj in subjects:
        print(f"Processing {subj.name}...")
        metrics_row = run_for_subject(
            model,
            subj,
            device,
            output_root=args.output_dir,
            extensions=args.extensions,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold=args.threshold,
        )
        overall_rows.append(metrics_row)

    # Save overall metrics
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall_csv = args.output_dir / "overall_metrics.csv"
    if overall_rows:
        fieldnames = list(overall_rows[0].keys())
    else:
        fieldnames = [
            "subject","num_frames","num_labeled","accuracy","precision","recall","f1","balanced_accuracy",
            "tp","tn","fp","fn","roc_auc","pr_auc"
        ]
    write_rows_csv(overall_csv, overall_rows, fieldnames)
    print(f"Wrote per-subject predictions/metrics and overall metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
