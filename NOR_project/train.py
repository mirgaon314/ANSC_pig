from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import InteractionSequenceDataset
from src.model import InteractionClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def binary_focal_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 1.5,
    alpha: float = 0.5,
) -> Tensor:
    probs = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    modulating_factor = (1 - p_t) ** gamma
    loss = alpha_factor * modulating_factor * ce_loss
    return loss.mean()


def attention_margin_penalty(weights: Tensor, centre_index: int, margin: float = 0.2) -> Tensor:
    if weights.ndim != 2:
        raise ValueError("weights tensor must have shape (batch, time)")
    centre_weights = weights[:, centre_index]
    penalty = F.relu(margin - centre_weights)
    return penalty.mean()


def split_sequences(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    df = df.sort_values("centre_frame").reset_index(drop=True)
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def build_dataloaders(
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    temporal_dropout_prob: float,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_df, val_df, test_df = split_sequences(sequences_df)

    train_dataset = InteractionSequenceDataset(
        train_df,
        metadata_df,
        training=True,
        temporal_dropout_prob=temporal_dropout_prob,
    )
    val_dataset = InteractionSequenceDataset(
        val_df,
        metadata_df,
        training=False,
        temporal_dropout_prob=0.0,
    )
    test_dataset = InteractionSequenceDataset(
        test_df,
        metadata_df,
        training=False,
        temporal_dropout_prob=0.0,
    )

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def compute_classification_stats(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (tpr + tnr) / 2

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def warmup_cosine_lr_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    if current_step < warmup_steps:
        return float(current_step) / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(
    model: InteractionClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    centre_index: int,
    gamma: float,
    alpha: float,
    attention_margin: float,
    attention_lambda: float,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_penalty = 0.0
    all_logits: List[Tensor] = []
    all_targets: List[Tensor] = []
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for step, batch in enumerate(pbar, start=1):
        if max_steps and step > max_steps:
            break
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, weights = model(inputs)
        logits = logits.view(-1, 1)
        loss = binary_focal_loss_with_logits(logits, targets, gamma=gamma, alpha=alpha)
        penalty = attention_margin_penalty(weights, centre_index=centre_index, margin=attention_margin)
        total = loss + attention_lambda * penalty
        total.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_penalty += penalty.item()
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "pen": f"{penalty.item():.4f}"})

    if not all_logits:
        return {"loss": 0.0, "penalty": 0.0, "balanced_accuracy": 0.0}

    logits_tensor = torch.cat(all_logits)
    targets_tensor = torch.cat(all_targets)
    stats = compute_classification_stats(logits_tensor, targets_tensor)
    steps = len(all_logits)
    return {
        "loss": total_loss / max(1, steps),
        "penalty": total_penalty / max(1, steps),
        **stats,
    }


@torch.no_grad()
def evaluate(
    model: InteractionClassifier,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    centre_index: int,
    split_name: str,
    gamma: float,
    alpha: float,
    attention_margin: float,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_penalty = 0.0
    all_logits: List[Tensor] = []
    all_targets: List[Tensor] = []
    pbar = tqdm(loader, desc=f"Epoch {epoch} [{split_name}]", leave=False)
    for step, batch in enumerate(pbar, start=1):
        if max_steps and step > max_steps:
            break
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        logits, weights = model(inputs)
        logits = logits.view(-1, 1)
        loss = binary_focal_loss_with_logits(logits, targets, gamma=gamma, alpha=alpha)
        penalty = attention_margin_penalty(
            weights, centre_index=centre_index, margin=attention_margin
        )
        total_loss += loss.item()
        total_penalty += penalty.item()
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    if not all_logits:
        return {"loss": 0.0, "penalty": 0.0, "balanced_accuracy": 0.0}

    logits_tensor = torch.cat(all_logits)
    targets_tensor = torch.cat(all_targets)
    stats = compute_classification_stats(logits_tensor, targets_tensor)
    steps = len(all_logits)
    return {
        "loss": total_loss / max(1, steps),
        "penalty": total_penalty / max(1, steps),
        **stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train interaction classifier")
    parser.add_argument("--balanced-csv", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.5, help="Focal loss gamma")
    parser.add_argument("--alpha", type=float, default=0.5, help="Focal loss alpha")
    parser.add_argument("--attention-margin", type=float, default=0.2)
    parser.add_argument("--attention-lambda", type=float, default=0.1)
    parser.add_argument("--temporal-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="Limit training steps per epoch (0 = no limit)")
    parser.add_argument("--max-val-steps", type=int, default=0, help="Limit validation steps (0 = no limit)")
    parser.add_argument("--prefer-mps", action="store_true", help="Prefer MPS device if available")
    parser.add_argument("--no-mps", action="store_true", help="Disable MPS usage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    prefer_mps = args.prefer_mps or not args.no_mps
    device = get_device(prefer_mps=prefer_mps)
    print(f"Using device: {device}")

    sequences_df = pd.read_csv(args.balanced_csv)
    metadata_df = pd.read_csv(args.metadata_csv)

    train_loader, val_loader, test_loader = build_dataloaders(
        sequences_df,
        metadata_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
        temporal_dropout_prob=args.temporal_dropout,
    )

    model = InteractionClassifier(in_channels=5, temporal_length=5)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_metric = -float("inf")
    history: List[Dict[str, float]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"

    max_steps = args.max_steps if args.max_steps > 0 else None
    max_val_steps = args.max_val_steps if args.max_val_steps > 0 else None

    effective_steps = max_steps or len(train_loader)
    total_steps = args.epochs * max(1, effective_steps)
    warmup_steps = max(1, int(0.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_lr_lambda(step, warmup_steps, total_steps),
    )

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            centre_index=model.centre_index,
            gamma=args.gamma,
            alpha=args.alpha,
            attention_margin=args.attention_margin,
            attention_lambda=args.attention_lambda,
            scheduler=scheduler,
            max_steps=max_steps,
        )
        val_stats = evaluate(
            model,
            val_loader,
            device,
            epoch,
            centre_index=model.centre_index,
            split_name="val",
            gamma=args.gamma,
            alpha=args.alpha,
            attention_margin=args.attention_margin,
            max_steps=max_val_steps,
        )
        epoch_record = {"epoch": epoch, "phase": "train", **train_stats}
        history.append(epoch_record)
        history.append({"epoch": epoch, "phase": "val", **val_stats})

        torch.save(model.state_dict(), args.output_dir / "last.pt")
        if val_stats["balanced_accuracy"] > best_metric:
            best_metric = val_stats["balanced_accuracy"]
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, best_metric)

        with metrics_path.open("w") as f:
            json.dump(history, f, indent=2)

        print(
            f"Epoch {epoch}: train_bal_acc={train_stats['balanced_accuracy']:.3f} "
            f"val_bal_acc={val_stats['balanced_accuracy']:.3f}"
        )
        if max_steps is not None and epoch == 1:
            print(f"Debug mode: limited to {max_steps} train batches per epoch.")

    test_stats = evaluate(
        model,
        test_loader,
        device,
        epoch=args.epochs,
        centre_index=model.centre_index,
        split_name="test",
        gamma=args.gamma,
        alpha=args.alpha,
        attention_margin=args.attention_margin,
        max_steps=max_val_steps,
    )
    history.append({"epoch": args.epochs, "phase": "test", **test_stats})
    with metrics_path.open("w") as f:
        json.dump(history, f, indent=2)

    save_checkpoint(args.output_dir / "final.pt", model, optimizer, args.epochs, best_metric)
    print(f"Training complete. Test balanced accuracy: {test_stats['balanced_accuracy']:.3f}")


if __name__ == "__main__":
    main()
