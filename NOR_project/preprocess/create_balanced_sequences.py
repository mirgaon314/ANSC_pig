"""Create a balanced dataset of 5-frame clips for pig-object interaction detection.

Usage example:
    python preprocess/create_balanced_sequences.py \
        --input-csv NOR_project/B004_per_frame_with_crops.csv \
        --output-csv NOR_project/balanced_sequences.csv \
        --image-root /Users/zimul3/Downloads/img
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SequenceSample:
    """Container describing a 5-frame sequence."""

    sample_id: int
    frame_numbers: List[int]
    frame_files: List[str]
    frame_paths: List[str]
    centre_frame: int
    centre_file: str
    label: int
    transition_on: bool
    prev_positive_count: int
    next_positive_count: int
    missing_image_count: int

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "frame_numbers": "|".join(str(n) for n in self.frame_numbers),
            "frame_files": "|".join(self.frame_files),
            "frame_paths": "|".join(self.frame_paths),
            "centre_frame": self.centre_frame,
            "centre_file": self.centre_file,
            "label": self.label,
            "transition_on": int(self.transition_on),
            "prev_positive_count": self.prev_positive_count,
            "next_positive_count": self.next_positive_count,
            "missing_image_count": self.missing_image_count,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate balanced 5-frame sequences for interaction classification."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to the per-frame metadata CSV (must contain Frame, HasMarker, MarkerName, crop_image columns).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Destination CSV that will hold balanced sequence metadata.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Directory containing extracted frame images. If omitted, the script attempts to resolve paths using crop_dir values in the CSV.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Number of consecutive frames per sample (must be odd; default=5).",
    )
    parser.add_argument(
        "--balance-strategy",
        choices=("downsample", "upsample"),
        default="downsample",
        help="Whether to downsample the majority class or upsample the minority class to achieve class balance.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary. Defaults to <output-csv>.summary.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {"Frame", "HasMarker", "MarkerName", "crop_image"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.sort_values("Frame").reset_index(drop=True)
    df["HasMarker"] = df["HasMarker"].astype(bool)
    df["MarkerName"] = df["MarkerName"].fillna("").str.lower()
    df["is_interaction"] = df["HasMarker"] & df["MarkerName"].isin({"left", "right"})
    return df


def resolve_image_path(
    image_root: Optional[Path],
    row: pd.Series,
    csv_parent: Path,
) -> str:
    filename = Path(str(row["crop_image"]))

    if image_root is not None:
        return str((image_root / filename.name).resolve())

    crop_dir_val = row.get("crop_dir")
    if isinstance(crop_dir_val, str) and crop_dir_val:
        crop_dir_path = Path(crop_dir_val)
        if not crop_dir_path.is_absolute():
            crop_dir_path = (csv_parent / crop_dir_path).resolve()
        return str((crop_dir_path / filename.name).resolve())

    return str((csv_parent / filename).resolve())


def iter_sequences(df: pd.DataFrame, seq_len: int) -> Iterable[tuple[pd.DataFrame, int]]:
    if seq_len % 2 == 0:
        raise ValueError("sequence_length must be odd to have a unique centre frame")

    half = seq_len // 2
    frames = df["Frame"].to_numpy()

    for centre_idx in range(half, len(df) - half):
        window = df.iloc[centre_idx - half : centre_idx + half + 1]
        window_frames = window["Frame"].to_numpy()
        if np.any(np.diff(window_frames) != 1):
            continue
        yield window, centre_idx


def build_sequences(
    df: pd.DataFrame,
    image_root: Optional[Path],
    seq_len: int,
    seed: int,
) -> List[SequenceSample]:
    rng = np.random.default_rng(seed)
    csv_parent = df.attrs.get("csv_parent")
    if csv_parent is None:
        raise RuntimeError("DataFrame missing csv_parent attribute for path resolution")

    samples: List[SequenceSample] = []
    sample_id = 0

    for window, centre_idx in iter_sequences(df, seq_len):
        frame_numbers = window["Frame"].astype(int).tolist()
        frame_files = window["crop_image"].tolist()
        frame_paths = [
            resolve_image_path(image_root, row, csv_parent) for _, row in window.iterrows()
        ]

        missing_count = sum(not Path(path).exists() for path in frame_paths)

        centre_row = window.iloc[seq_len // 2]
        label = int(centre_row["is_interaction"])
        prev_positive = int(window.iloc[: seq_len // 2]["is_interaction"].sum())
        next_positive = int(window.iloc[seq_len // 2 + 1 :]["is_interaction"].sum())
        transition_on = label == 1 and prev_positive == 0

        # Light temporal jitter by swapping neighbours with small probability to diversify sequences.
        if seq_len == 5 and rng.random() < 0.05:
            frame_paths[0], frame_paths[1] = frame_paths[1], frame_paths[0]
            frame_files[0], frame_files[1] = frame_files[1], frame_files[0]
            frame_numbers[0], frame_numbers[1] = frame_numbers[1], frame_numbers[0]

        samples.append(
            SequenceSample(
                sample_id=sample_id,
                frame_numbers=frame_numbers,
                frame_files=frame_files,
                frame_paths=frame_paths,
                centre_frame=int(centre_row["Frame"]),
                centre_file=centre_row["crop_image"],
                label=label,
                transition_on=transition_on,
                prev_positive_count=prev_positive,
                next_positive_count=next_positive,
                missing_image_count=missing_count,
            )
        )
        sample_id += 1

    return samples


def select_with_transition_priority(
    df: pd.DataFrame,
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    if len(df) <= target_size:
        return df.copy()

    transition_df = df[df["transition_on"] == 1]
    remainder_df = df[df["transition_on"] == 0]

    rng = np.random.default_rng(seed)
    if len(transition_df) >= target_size:
        selected_idx = rng.choice(transition_df.index.to_numpy(), size=target_size, replace=False)
        return transition_df.loc[selected_idx].copy()

    selected_indices = list(transition_df.index.to_numpy())
    remaining = target_size - len(selected_indices)
    if remaining > 0 and not remainder_df.empty:
        extra_indices = rng.choice(
            remainder_df.index.to_numpy(),
            size=remaining,
            replace=False,
        )
        selected_indices.extend(extra_indices)

    return df.loc[selected_indices].copy()


def balance_sequences(
    samples: List[SequenceSample],
    balance_strategy: str,
    seed: int,
) -> pd.DataFrame:
    df = pd.DataFrame([sample.to_dict() for sample in samples])
    positive = df[df["label"] == 1]
    negative = df[df["label"] == 0]

    if positive.empty or negative.empty:
        raise RuntimeError(
            "Cannot balance dataset because one of the classes is empty after sequence construction."
        )

    rng = np.random.default_rng(seed)

    if balance_strategy == "downsample":
        target = min(len(positive), len(negative))
        pos_sel = select_with_transition_priority(positive, target, seed)
        neg_sel = negative.sample(n=target, random_state=seed)
    else:  # upsample
        target = max(len(positive), len(negative))
        if len(positive) < target:
            pos_choices = positive.sample(n=target, replace=True, random_state=seed)
        else:
            pos_choices = positive
        if len(negative) < target:
            neg_choices = negative.sample(n=target, replace=True, random_state=seed)
        else:
            neg_choices = negative
        pos_sel = pos_choices
        neg_sel = neg_choices

    balanced = pd.concat([pos_sel, neg_sel], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    balanced.sort_values("centre_frame", inplace=True)
    balanced.reset_index(drop=True, inplace=True)
    balanced["sample_id"] = np.arange(len(balanced))
    return balanced


def summarise_dataset(df: pd.DataFrame) -> dict:
    return {
        "num_samples": int(len(df)),
        "num_positive": int((df["label"] == 1).sum()),
        "num_negative": int((df["label"] == 0).sum()),
        "num_transition_on_positive": int(((df["label"] == 1) & (df["transition_on"] == 1)).sum()),
        "missing_image_samples": int((df["missing_image_count"] > 0).sum()),
        "prev_positive_hist": df["prev_positive_count"].value_counts().to_dict(),
        "next_positive_hist": df["next_positive_count"].value_counts().to_dict(),
    }


def main() -> None:
    args = parse_args()

    df = load_metadata(args.input_csv)
    df.attrs["csv_parent"] = args.input_csv.parent.resolve()

    samples = build_sequences(df, args.image_root, args.sequence_length, args.seed)
    if not samples:
        raise RuntimeError("No valid sequences could be constructed. Check frame continuity.")

    balanced_df = balance_sequences(samples, args.balance_strategy, args.seed)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(args.output_csv, index=False)

    summary_path = args.summary_json or args.output_csv.with_suffix(args.output_csv.suffix + ".summary.json")
    summary = summarise_dataset(balanced_df)
    summary["input_csv"] = str(args.input_csv)
    summary["output_csv"] = str(args.output_csv)
    summary["sequence_length"] = args.sequence_length
    summary["balance_strategy"] = args.balance_strategy

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
