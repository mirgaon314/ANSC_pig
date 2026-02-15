from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD = torch.tensor([0.229, 0.224, 0.225])


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as img:
        return img.convert("RGB")


@dataclass(frozen=True)
class WindowSample:
    centre_index: int
    frame_paths: List[Path]
    centre_path: Path


class FolderSequenceDataset(Dataset[Dict[str, Tensor]]):
    """Dataset yielding 5-frame clips centred on each image in a folder."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        seq_len: int = 5,
        temporal_positions: bool = True,
        include_prior: bool = True,
    ) -> None:
        if seq_len % 2 == 0:
            raise ValueError("seq_len must be odd")
        self.seq_len = seq_len
        self.half = seq_len // 2
        self.temporal_positions = temporal_positions
        self.include_prior = include_prior
        self.image_paths = list(image_paths)
        if not self.image_paths:
            raise ValueError("No images provided for inference")

        self.samples: List[WindowSample] = []
        for idx, centre_path in enumerate(self.image_paths):
            frame_paths = self._collect_window(idx)
            self.samples.append(
                WindowSample(
                    centre_index=idx,
                    frame_paths=frame_paths,
                    centre_path=centre_path,
                )
            )

    def _collect_window(self, centre_idx: int) -> List[Path]:
        paths: List[Path] = []
        last_idx = len(self.image_paths) - 1
        for offset in range(-self.half, self.half + 1):
            idx = min(max(centre_idx + offset, 0), last_idx)
            paths.append(self.image_paths[idx])
        return paths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        sample = self.samples[index]
        frames = [_load_image(path) for path in sample.frame_paths]

        frame_tensors: List[Tensor] = []
        for frame in frames:
            tensor = F.pil_to_tensor(frame).float() / 255.0
            tensor = F.normalize(tensor, mean=_MEAN, std=_STD)
            frame_tensors.append(tensor)

        rgb = torch.stack(frame_tensors, dim=1)  # (C, T, H, W)

        channels: List[Tensor] = [rgb]

        if self.temporal_positions:
            positions = torch.linspace(0.0, 1.0, steps=self.seq_len, dtype=torch.float32)
            temporal = positions.view(1, self.seq_len, 1, 1).expand(1, self.seq_len, rgb.shape[2], rgb.shape[3])
            channels.append(temporal)

        if self.include_prior:
            prior = torch.zeros((1, self.seq_len, rgb.shape[2], rgb.shape[3]), dtype=torch.float32)
            channels.append(prior)

        inputs = torch.cat(channels, dim=0)
        return {
            "inputs": inputs,
            "centre_path": str(sample.centre_path),
        }
