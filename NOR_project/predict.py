from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.inference import FolderSequenceDataset
from src.model import InteractionClassifier

DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interaction predictions on a folder of images")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing frames")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/full_run/best.pt"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("predictions.csv"),
        help="Where to save predictions CSV",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for binary label")
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="*",
        default=sorted(DEFAULT_EXTENSIONS),
        help="File extensions (case-insensitive) to include",
    )
    parser.add_argument("--prefer-mps", action="store_true", help="Run on Apple MPS if available")
    parser.add_argument("--no-mps", action="store_true", help="Disable MPS usage")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit number of frames for debugging")
    return parser.parse_args()


def get_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_image_paths(images_dir: Path, extensions: Sequence[str]) -> List[Path]:
    ext_set = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    image_paths = [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in ext_set
    ]
    if not image_paths:
        raise ValueError(f"No images found in {images_dir} with extensions {sorted(ext_set)}")
    return image_paths


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


def run_predictions(
    model: InteractionClassifier,
    dataset: FolderSequenceDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    threshold: float,
) -> List[dict]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    results: List[dict] = []
    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict", leave=False):
            inputs = batch["inputs"].to(device, non_blocking=True)
            logits, _ = model(inputs)
            probs = sigmoid(logits)
            centre_paths = batch["centre_path"]
            if isinstance(centre_paths, str):
                centre_paths = [centre_paths]
            for path_str, prob in zip(centre_paths, probs):
                probability = float(prob.item())
                results.append(
                    {
                        "frame_path": path_str,
                        "probability": probability,
                        "prediction": int(probability >= threshold),
                    }
                )
    return results


def write_csv(output_path: Path, rows: Iterable[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_path", "probability", "prediction"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    prefer_mps = args.prefer_mps or not args.no_mps
    device = get_device(prefer_mps=prefer_mps)
    print(f"Using device: {device}")

    image_paths = collect_image_paths(args.images_dir, args.extensions)
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
        print(f"Limiting to {len(image_paths)} frames for debug mode")

    dataset = FolderSequenceDataset(image_paths=image_paths)
    model = load_checkpoint(args.checkpoint, device)

    results = run_predictions(
        model,
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        threshold=args.threshold,
    )

    write_csv(args.output_csv, results)
    print(f"Wrote {len(results)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
