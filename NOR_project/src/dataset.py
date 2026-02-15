from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as img:
        return img.convert("RGB")


@dataclass
class SequenceRecord:
    frame_numbers: List[int]
    frame_paths: List[Path]
    label: int
    transition_on: bool
    prev_positive_count: int
    next_positive_count: int
    centre_frame: int


class SequenceAugmenter:
    """Applies consistent augmentations across all frames in a clip."""

    def __init__(
        self,
        hflip_prob: float = 0.5,
        color_jitter: transforms.ColorJitter | None = None,
        max_affine_deg: float = 5.0,
    ) -> None:
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter or transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
        )
        self.max_affine_deg = max_affine_deg

    def __call__(self, frames: List[Image.Image]) -> List[Image.Image]:
        out = [frame.copy() for frame in frames]

        if random.random() < self.hflip_prob:
            out = [F.hflip(frame) for frame in out]

        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.8, 1.2)
        hue_factor = random.uniform(-0.02, 0.02)

        out = [F.adjust_brightness(frame, brightness_factor) for frame in out]
        out = [F.adjust_contrast(frame, contrast_factor) for frame in out]
        out = [F.adjust_saturation(frame, saturation_factor) for frame in out]
        out = [F.adjust_hue(frame, hue_factor) for frame in out]

        angle, translate, scale, shear = transforms.RandomAffine.get_params(
            degrees=(-self.max_affine_deg, self.max_affine_deg),
            translate=(0.02, 0.02),
            scale_ranges=(0.95, 1.05),
            shears=None,
            img_size=out[0].size,
        )
        out = [
            F.affine(frame, angle=angle, translate=translate, scale=scale, shear=shear)
            for frame in out
        ]

        return out


class InteractionSequenceDataset(Dataset[Dict[str, Tensor]]):
    """Dataset yielding 5-frame clips with auxiliary channels."""

    def __init__(
        self,
        sequences_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        training: bool = True,
        seq_len: int = 5,
        temporal_dropout_prob: float = 0.1,
        image_loader: Callable[[Path], Image.Image] | None = None,
    ) -> None:
        if seq_len % 2 == 0:
            raise ValueError("seq_len must be odd")
        self.seq_len = seq_len
        self.training = training
        self.temporal_dropout_prob = temporal_dropout_prob
        self.image_loader = image_loader or _load_image

        self.records: List[SequenceRecord] = []
        for _, row in sequences_df.iterrows():
            frame_numbers = [int(v) for v in str(row["frame_numbers"]).split("|")]
            frame_paths = [Path(p) for p in str(row["frame_paths"]).split("|")]
            self.records.append(
                SequenceRecord(
                    frame_numbers=frame_numbers,
                    frame_paths=frame_paths,
                    label=int(row["label"]),
                    transition_on=bool(row.get("transition_on", 0)),
                    prev_positive_count=int(row.get("prev_positive_count", 0)),
                    next_positive_count=int(row.get("next_positive_count", 0)),
                    centre_frame=int(row["centre_frame"]),
                )
            )

        metadata_df = metadata_df.copy()
        metadata_df["MarkerName"] = metadata_df["MarkerName"].fillna("").str.lower()
        metadata_df["HasMarker"] = metadata_df["HasMarker"].astype(bool)
        metadata_df["is_interaction"] = metadata_df["HasMarker"] & metadata_df[
            "MarkerName"
        ].isin({"left", "right"})
        self.frame_label_map: Dict[int, int] = {
            int(frame): int(is_interaction)
            for frame, is_interaction in metadata_df[["Frame", "is_interaction"]].values
        }

        self.augment = SequenceAugmenter() if training else None
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.records)

    def _load_frames(self, record: SequenceRecord) -> List[Image.Image]:
        return [self.image_loader(path) for path in record.frame_paths]

    def _build_auxiliary_channels(
        self, record: SequenceRecord, height: int, width: int, device: torch.device
    ) -> Dict[str, Tensor]:
        temporal_positions = torch.linspace(
            0.0, 1.0, steps=self.seq_len, dtype=torch.float32, device=device
        ).view(1, self.seq_len, 1, 1)
        temporal_positions = temporal_positions.expand(1, self.seq_len, height, width)

        prior = torch.tensor(
            [
                float(self.frame_label_map.get(frame, 0))
                for frame in record.frame_numbers
            ],
            dtype=torch.float32,
            device=device,
        ).view(1, self.seq_len, 1, 1)
        prior = prior.expand(1, self.seq_len, height, width)

        return {
            "temporal_positions": temporal_positions,
            "interaction_prior": prior,
        }

    def _temporal_dropout(self, frames: Tensor) -> Tensor:
        if self.seq_len != 5:
            return frames
        if random.random() >= self.temporal_dropout_prob:
            return frames
        neighbour_indices = [0, 1, 3, 4]
        target_idx = random.choice(neighbour_indices)
        centre_idx = self.seq_len // 2
        frames[:, target_idx] = frames[:, centre_idx]
        return frames

    def __getitem__(self, index: int) -> Dict[str, Tensor | int]:
        record = self.records[index]
        frames = self._load_frames(record)

        if self.augment is not None:
            frames = self.augment(frames)

        frame_tensors: List[Tensor] = []
        for frame in frames:
            tensor = F.pil_to_tensor(frame).float() / 255.0
            tensor = F.normalize(tensor, mean=self.mean, std=self.std)
            frame_tensors.append(tensor)

        rgb = torch.stack(frame_tensors, dim=0)  # (T, C, H, W)
        rgb = rgb.permute(1, 0, 2, 3)  # (C, T, H, W)
        rgb = self._temporal_dropout(rgb)

        aux = self._build_auxiliary_channels(
            record, height=rgb.shape[-2], width=rgb.shape[-1], device=rgb.device
        )
        inputs = torch.cat([rgb, aux["temporal_positions"], aux["interaction_prior"]], dim=0)

        label = torch.tensor([float(record.label)], dtype=torch.float32)
        return {
            "inputs": inputs,
            "label": label,
            "centre_frame": torch.tensor(record.centre_frame, dtype=torch.int32),
            "transition_on": torch.tensor(record.transition_on, dtype=torch.int32),
        }
