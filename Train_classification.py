#!/usr/bin/env python3
"""
Train_multimodal (image + tabular)

Multimodal training that combines:
  - Tabular coordinates/features from per-frame CSVs (same features as Train_classification.py)
  - Pig crop images (480x480 gray-canvas crops you generated)

Target (2-class):
  - "not_interacting"
  - "interacting"

How it works:
  * Loads 1..N CSV files (either pass --csvs paths directly, or pass --names and --base to auto-resolve).
  * Builds the same tabular features as in Train_classification.py.
  * Loads the pig crop image for each row (uses first filename if multiple are listed).
  * Trains a late-fusion model: ResNet18 image encoder (pretrained) + MLP on tabular features -> fused classifier.

Example runs:

# Option A: pass names to auto-resolve CSVs in ../ANSC/mark_frame_crops/{NAME}/{NAME}_per_frame_with_crops.csv
python ANSC_pig/Train_multimodal.py \
  --names B001 B002 F003 \
  --base ../ANSC/mark_frame_crops \
  --outdir cls_runs_mm \
  --epochs 15 --batch 32 --device mps

# Option B: pass CSV paths directly
python ANSC_pig/Train_multimodal.py \
  --csvs ../ANSC/mark_frame_crops/B001/B001_per_frame_with_crops.csv ../ANSC/mark_frame_crops/F003/F003_per_frame_with_crops.csv \
  --outdir cls_runs_mm \
  --epochs 15 --batch 32 --device mps
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # safe on Mac

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import json
import math
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

# ----------------------------
# Defaults for "Run button" usage (no CLI args)
# ----------------------------
# Set DEFAULT_NAMES to e.g. ["B001","F003"] to use specific recordings,
# or leave it empty to auto-discover all *_per_frame_with_crops.csv under DEFAULT_BASE.
DEFAULT_NAMES: List[str] = ["B001","B002","B003","B004","B005","C003","C004","C005", "F002","F003","F004","F005","FB001","FB002","FB003","FB004","FB005"]
DEFAULT_BASE: Path = (Path(__file__).parent / ".." / "mark_frame_crops").resolve()

def autodiscover_csvs(base: Path) -> List[Path]:
    return sorted(base.rglob("*_per_frame_with_crops.csv"))


# ----------------------------
# Tabular feature construction
# ----------------------------
BASE_FEATURES = [
    "pig_cx", "pig_cy", "pig_w", "pig_h",
    "left_object_cx", "left_object_cy", "left_object_w", "left_object_h",
    "right_object_cx", "right_object_cy", "right_object_w", "right_object_h",
    "head_cx", "head_cy", "head_w", "head_h",
]

DERIVED_FEATURES = [
    "dist_pig_to_left", "dist_pig_to_right",
    "dist_head_to_left", "dist_head_to_right",
    "dx_head_left", "dy_head_left", "dx_head_right", "dy_head_right",
    "dx_pig_left", "dy_pig_left", "dx_pig_right", "dy_pig_right",
    "pig_aspect", "head_aspect", "left_object_aspect", "right_object_aspect",
]

ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES

# --- Trial window helpers -----------------------------------------------------

def _boolish(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return False


def filter_to_trial_window(df: pd.DataFrame, rec_name: str = "") -> pd.DataFrame:
    """
    Keep only frames within the trial window: from the first marker row that contains 'start'
    (e.g., 'Trial,<name>,start') to the marker row that contains 'end' (if any).
    This ensures NOT_INTERACTING negatives are sampled *after* the trial has begun.
    If no start marker is found, returns the original df unchanged.
    """
    if "HasMarker" not in df.columns or "MarkerName" not in df.columns:
        # Nothing to filter by; return as-is
        return df

    m = df.copy()
    has_marker = m["HasMarker"].apply(_boolish)
    name = m["MarkerName"].astype(str).str.lower()

    start_idx = None
    end_idx = None

    # Prefer explicit 'start'/'end' tokens
    starts = m.index[ has_marker & name.str.contains("start", na=False) ].tolist()
    ends   = m.index[ has_marker & name.str.contains("end",   na=False) ].tolist()

    if starts:
        start_idx = min(starts)
    if ends:
        # choose the last end marker after start if both present
        end_idx = max([e for e in ends if (start_idx is None or e >= start_idx)] or ends)

    if start_idx is None:
        # No explicit start; try to infer from the first interaction marker
        first_inter = m.index[ has_marker & name.str.contains("interacting_left|interacting_right|left|right", regex=True, na=False) ].tolist()
        if first_inter:
            start_idx = min(first_inter)

    if start_idx is None:
        # Still nothing; keep original
        print(f"[TRIAL] No start marker found for '{rec_name}'. Using full file without trimming.")
        return df

    # Slice between start and end (inclusive)
    if end_idx is not None and end_idx >= start_idx:
        sliced = m.loc[start_idx:end_idx].copy()
    else:
        sliced = m.loc[start_idx:].copy()

    print(f"[TRIAL] '{rec_name}': kept rows {len(sliced)} (from idx {start_idx} to {('end '+str(end_idx)) if end_idx is not None else 'EOF'})")
    return sliced

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
    has_marker = (has == "true") or (has == "1") or (has == "yes") or (row.get("HasMarker") is True)
    name = str(row.get("MarkerName", "")).lower()
    if has_marker and ("left" in name or "right" in name):
        return "interacting"
    return "not_interacting"

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    for col in BASE_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

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

    df["dx_head_left"] = df["head_cx"] - df["left_object_cx"]
    df["dy_head_left"] = df["head_cy"] - df["left_object_cy"]
    df["dx_head_right"] = df["head_cx"] - df["right_object_cx"]
    df["dy_head_right"] = df["head_cy"] - df["right_object_cy"]

    df["dx_pig_left"] = df["pig_cx"] - df["left_object_cx"]
    df["dy_pig_left"] = df["pig_cy"] - df["left_object_cy"]
    df["dx_pig_right"] = df["pig_cx"] - df["right_object_cx"]
    df["dy_pig_right"] = df["pig_cy"] - df["right_object_cy"]

    df["pig_aspect"] = df.apply(lambda r: _aspect(r["pig_w"], r["pig_h"]), axis=1)
    df["head_aspect"] = df.apply(lambda r: _aspect(r["head_w"], r["head_h"]), axis=1)
    df["left_object_aspect"] = df.apply(lambda r: _aspect(r["left_object_w"], r["left_object_h"]), axis=1)
    df["right_object_aspect"] = df.apply(lambda r: _aspect(r["right_object_w"], r["right_object_h"]), axis=1)

    return df[ALL_FEATURES], ALL_FEATURES


# ----------------------------
# Dataset
# ----------------------------
LABEL2ID = {"not_interacting": 0, "interacting": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

class PigInteractionDataset(Dataset):
    def __init__(self, frames: pd.DataFrame, feature_names: List[str], transform: T.Compose):
        self.df = frames.reset_index(drop=True)
        self.feat_names = feature_names
        self.transform = transform

        # Build label column
        self.df["label"] = self.df.apply(make_label, axis=1)
        # Keep only rows with an image path
        if "crop_dir" not in self.df.columns or "crop_image" not in self.df.columns:
            raise KeyError("CSV must include 'crop_dir' and 'crop_image' columns. Use your crop_pigs.py first.")
        self.df["crop_image"] = self.df["crop_image"].fillna("")
        self.df = self.df[self.df["crop_image"].str.len() > 0].copy()

        # Resolve absolute image paths (take first if multiple separated by ';')
        def build_path(row):
            ci = row["crop_image"]
            if ";" in ci:
                ci = ci.split(";")[0]
            return str(Path(row["crop_dir"]) / ci)
        self.df["img_path"] = self.df.apply(build_path, axis=1)

        # Basic existence check
        self.df = self.df[self.df["img_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

        # Prepare tabular features (NaN safe; we replace NaN with 0, rely on BatchNorm later)
        feats, _ = build_features(self.df.copy())
        self.feats = feats.fillna(0.0).astype(np.float32).values
        self.labels = self.df["label"].map(LABEL2ID).astype(int).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.iloc[idx]["img_path"]
        with Image.open(p) as im:
            im = im.convert("RGB")
        img = self.transform(im)
        feat = torch.from_numpy(self.feats[idx])
        label = int(self.labels[idx])
        return img, feat, label


# ----------------------------
# Model
# ----------------------------
class FusionNet(nn.Module):
    def __init__(self, n_tab: int, n_classes: int = 3, img_embed: int = 256, tab_embed: int = 128):
        super().__init__()
        # Image encoder
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.img_head = nn.Linear(in_feats, img_embed)

        # Tabular tower
        self.tab_mlp = nn.Sequential(
            nn.Linear(n_tab, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.10),
            nn.Linear(256, tab_embed),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(tab_embed),
        )

        # Fusion head
        self.classifier = nn.Sequential(
            nn.Linear(img_embed + tab_embed, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, img, tab):
        # Image path
        z_img = self.backbone(img)           # [B, 512]
        z_img = self.img_head(z_img)         # [B, img_embed]
        # Tabular path
        z_tab = self.tab_mlp(tab)            # [B, tab_embed]
        # Fuse
        z = torch.cat([z_img, z_tab], dim=1) # [B, img_embed+tab_embed]
        logits = self.classifier(z)
        return logits


# ----------------------------
# Utils
# ----------------------------
def make_dataloaders(csv_paths: List[Path], img_size: int = 224, batch: int = 32, num_workers: int = 4, val_split: float = 0.2, test_split: float = 0.1):
    """
    Build DataLoaders with a **group-aware** split: no frames from the same recording (CSV) appear in multiple splits.
    Groups are inferred from the CSV parent folder name: base/{NAME}/{NAME}_per_frame_with_crops.csv
    """
    # Build transforms
    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(0.5),
        T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.03),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        T.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
    ])
    eval_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Load each CSV and tag with its recording group
    dfs = []
    rec_lengths = {}  # recording -> number of rows
    for p in csv_paths:
        rec = p.parent.name  # recording/group name
        df_i = pd.read_csv(p)
        # Trim to trial window (after start, up to end if present)
        df_i = filter_to_trial_window(df_i, rec_name=rec)
        df_i["__recording__"] = rec
        dfs.append(df_i)
        rec_lengths[rec] = rec_lengths.get(rec, 0) + len(df_i)

    if not dfs:
        raise RuntimeError("No CSVs loaded.")

    df_all = pd.concat(dfs, ignore_index=True)

    # Prepare group-aware split
    recordings = list(rec_lengths.keys())
    # deterministic shuffle
    rng = np.random.RandomState(42)
    rng.shuffle(recordings)

    total_rows = int(sum(rec_lengths.values()))
    target_test = int(round(total_rows * max(0.0, test_split)))
    target_val  = int(round(total_rows * max(0.0, val_split)))

    test_recs, val_recs, train_recs = [], [], []
    acc_test, acc_val = 0, 0
    for r in recordings:
        r_len = rec_lengths[r]
        if acc_test < target_test:
            test_recs.append(r); acc_test += r_len
        elif acc_val < target_val:
            val_recs.append(r);  acc_val  += r_len
        else:
            train_recs.append(r)

    # In case either split ended up empty due to tiny dataset, rebalance minimally
    if not train_recs and val_recs:
        train_recs.append(val_recs.pop())  # move one rec from val to train
    if not train_recs and test_recs:
        train_recs.append(test_recs.pop())  # move one rec from test to train
    if not val_recs and train_recs and len(train_recs) > 1:
        val_recs.append(train_recs.pop())
    if test_split > 0 and not test_recs and train_recs and len(train_recs) > 1:
        test_recs.append(train_recs.pop())

    df_train = df_all[df_all["__recording__"].isin(train_recs)].copy()
    df_val   = df_all[df_all["__recording__"].isin(val_recs)].copy()
    df_test  = df_all[df_all["__recording__"].isin(test_recs)].copy()

    # Datasets
    ds_train = PigInteractionDataset(df_train, ALL_FEATURES, transform=train_tf)
    ds_val   = PigInteractionDataset(df_val,   ALL_FEATURES, transform=eval_tf)
    ds_test  = PigInteractionDataset(df_test,  ALL_FEATURES, transform=eval_tf) if len(df_test) else None

    # Debug: show split sizes and class counts
    def _hist(labels_np):
        bins = np.bincount(labels_np, minlength=2)
        return {ID2LABEL[i]: int(bins[i]) for i in range(2)}
    print(f"[DATA] rows: train={len(df_train)} val={len(df_val)} test={len(df_test)}")
    print(f"[DATA] samples (with images): train={len(ds_train)} val={len(ds_val)} test={(len(ds_test) if ds_test is not None else 0)}")
    print(f"[DATA] train class counts: {_hist(ds_train.labels)}")
    print(f"[DATA] val class counts:   {_hist(ds_val.labels)}")
    if ds_test is not None:
        print(f"[DATA] test class counts:  {_hist(ds_test.labels)}")

    # Class balancing via WeightedRandomSampler on TRAIN
    y = ds_train.labels
    if len(y) == 0:
        raise RuntimeError("No training samples found (do your CSVs have crop_dir & crop_image filled?)")
    class_counts = np.bincount(y, minlength=2)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y]
    sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.float), num_samples=len(y), replacement=True)

    # DataLoaders
    train_loader = DataLoader(ds_train, batch_size=batch, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(ds_test,  batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True) if ds_test is not None else None

    print(f"[SPLIT] recordings: train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")
    return train_loader, val_loader, test_loader, len(ALL_FEATURES), class_counts.tolist()


def train(args):
    # Resolve CSV list
    csv_paths: List[Path] = []
    if args.csvs:
        csv_paths = [Path(p) for p in args.csvs]
    elif args.names:
        base = Path(args.base)
        for n in args.names:
            csv_paths.append(base / n / f"{n}_per_frame_with_crops.csv")
    else:
        raise ValueError("Provide either --csvs or --names with --base")

    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

    train_loader, val_loader, test_loader, n_tab, class_counts = make_dataloaders(
        csv_paths, img_size=args.imgsz, batch=args.batch, num_workers=args.workers, val_split=args.val_split, test_split=args.test_split
    )

    # --- Standardize tabular features using TRAIN stats (z-score) ---
    ds_train = train_loader.dataset
    ds_val = val_loader.dataset
    ds_test = test_loader.dataset if test_loader is not None else None

    feat_mean = ds_train.feats.mean(axis=0, dtype=np.float64)
    feat_std = ds_train.feats.std(axis=0, dtype=np.float64) + 1e-6

    ds_train.feats = ((ds_train.feats - feat_mean) / feat_std).astype(np.float32)
    ds_val.feats = ((ds_val.feats - feat_mean) / feat_std).astype(np.float32)
    if ds_test is not None:
        ds_test.feats = ((ds_test.feats - feat_mean) / feat_std).astype(np.float32)

    print("[NORM] Applied z-score normalization to tabular features (using TRAIN stats).")

    # Device (respect CUDA index strings like "0","1","2")
    if args.device in ("0", "1", "2") and torch.cuda.is_available():
        torch.cuda.set_device(int(args.device))
        device = torch.device("cuda")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # Model
    model = FusionNet(n_tab=n_tab, n_classes=2)
    model.to(device)

    # Freeze backbone for first few epochs, then unfreeze with a lower LR
    freeze_epochs = getattr(args, "freeze_epochs", 3)
    lr_backbone = getattr(args, "lr_backbone", 3e-5)

    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]

    for p in backbone_params:
        p.requires_grad = (freeze_epochs == 0)

    # Loss (label smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr},
        {"params": backbone_params, "lr": 0.0 if freeze_epochs > 0 else lr_backbone},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / "mm_fusion_best.pt"
    meta_path = outdir / "mm_fusion_meta.json"

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone at the boundary epoch
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            for p in backbone_params:
                p.requires_grad = True
            optimizer.param_groups[1]["lr"] = lr_backbone
            print(f"[UNFREEZE] Enabled backbone with lr={lr_backbone}")
        # Train
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for imgs, feats, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs, feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * labels.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        scheduler.step()

        # Validate
        model.eval()
        v_total, v_correct, v_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for imgs, feats, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                feats = feats.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(imgs, feats)
                loss = criterion(logits, labels)

                v_loss_sum += loss.item() * labels.size(0)
                preds = logits.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_loss = v_loss_sum / max(v_total, 1)
        val_acc = v_correct / max(v_total, 1)
        print(f"Epoch {epoch:02d}/{args.epochs}  train_loss={train_loss:.4f} acc={train_acc:.3f}  val_loss={val_loss:.4f} acc={val_acc:.3f}")

        # Save best
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "n_tab": n_tab,
                "label_map": ID2LABEL,
                "feature_names": ALL_FEATURES,
                "img_size": args.imgsz,
                "feat_mean": feat_mean.astype(float).tolist(),
                "feat_std": feat_std.astype(float).tolist(),
            }, ckpt_path)
            with open(meta_path, "w") as f:
                json.dump({
                    "csvs": [str(p) for p in csv_paths],
                    "class_counts": class_counts,
                    "best_val_acc": float(best_val_acc),
                    "feature_names": ALL_FEATURES
                }, f, indent=2)
            print(f"[OK] Saved best checkpoint to {ckpt_path} (val_acc={best_val_acc:.3f})")

    # -------- Evaluate BEST checkpoint on TEST set --------
    if test_loader is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        t_total, t_correct, t_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for imgs, feats, labels in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                feats = feats.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs, feats)
                loss = criterion(logits, labels)
                t_loss_sum += loss.item() * labels.size(0)
                preds = logits.argmax(1)
                t_correct += (preds == labels).sum().item()
                t_total += labels.size(0)
        test_loss = t_loss_sum / max(t_total, 1)
        test_acc = t_correct / max(t_total, 1)
        print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.3f}")

    print(f"[DONE] Best val acc: {best_val_acc:.3f}. Checkpoint: {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train a multimodal (image + tabular) interaction classifier")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csvs", nargs="+", help="List of per_frame_with_crops CSV paths")
    g.add_argument("--names", nargs="+", help="Recording names (auto-resolve CSVs under --base)")

    p.add_argument("--base", default="../ANSC/mark_frame_crops", help="Base folder when using --names")
    p.add_argument("--outdir", default="cls_runs_mm", help="Where to save model & meta")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--test-split", type=float, default=0.1)
    p.add_argument("--device", default="mps", choices=["cpu","mps","cuda","0","1","2"], help="Device to use")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--freeze-epochs", type=int, default=3, help="freeze image backbone for N epochs before unfreezing")
    p.add_argument("--lr-backbone", type=float, default=3e-5, help="learning rate for image backbone after unfreeze")
    return p.parse_args()


if __name__ == "__main__":
    # If no CLI args are provided, build args from defaults so you can just press "Run"
    if len(sys.argv) == 1:
        if DEFAULT_NAMES:
            args = argparse.Namespace(
                csvs=None,
                names=DEFAULT_NAMES,
                base=str(DEFAULT_BASE),
                outdir="../ANSC/ANSC_pig/models/cls_runs_mm",
                epochs=15,
                batch=32,
                imgsz=224,
                workers=4,
                val_split=0.2,
                test_split=0.1,
                device="mps",
                lr=1e-4,
                freeze_epochs=3,
                lr_backbone=3e-5,
            )
            print(f"[AUTO] Using DEFAULT_NAMES={DEFAULT_NAMES} under base={DEFAULT_BASE}")
        else:
            discovered = autodiscover_csvs(DEFAULT_BASE)
            if not discovered:
                raise FileNotFoundError(f"No CSVs found under {DEFAULT_BASE}. "
                                        f"Expected pattern '*_per_frame_with_crops.csv'. "
                                        f"Set DEFAULT_NAMES to a list like ['B001','F003'] or run with CLI.")
            args = argparse.Namespace(
                csvs=[str(p) for p in discovered],
                names=None,
                base=str(DEFAULT_BASE),
                outdir="/ANSC_pig/models/cls_runs_mm",
                epochs=15,
                batch=32,
                imgsz=224,
                workers=4,
                val_split=0.2,
                test_split=0.1,
                device="mps",
                lr=1e-4,
                freeze_epochs=3,
                lr_backbone=3e-5,
            )
            print(f"[AUTO] Discovered {len(discovered)} CSV(s) under {DEFAULT_BASE}")
    else:
        args = parse_args()

    train(args)
