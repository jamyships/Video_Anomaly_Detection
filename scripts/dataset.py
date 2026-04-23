from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


LabelName = Literal["normal", "abnormal"]


def list_video_ids(frames_root: Path, label: LabelName) -> list[str]:
    """
    The KaggleHub TAD dump is organized as:
      frames/{abnormal,normal}/{video_name.mp4}/frame_000001.jpg ...

    We treat `{video_name.mp4}` directory name as the video id.
    """
    d = frames_root / label
    if not d.exists():
        raise FileNotFoundError(f"Missing directory: {d}")
    return sorted([p.name for p in d.iterdir() if p.is_dir()])


class TADFeatureDataset(Dataset):
    """
    Loads pre-extracted features: (N=32, F=1024) float32 per video.

    File layout (recommended):
      data/features/abnormal/<video_id>.npy
      data/features/normal/<video_id>.npy
    """

    def __init__(self, features_root: Path, label: LabelName):
        self.features_dir = features_root / label
        if not self.features_dir.exists():
            raise FileNotFoundError(
                f"Features directory not found: {self.features_dir}. "
                "Run `scripts/i3d_extractor.py` (or provide features) first."
            )
        self.items = sorted([p for p in self.features_dir.glob("*.npy")])
        if not self.items:
            raise FileNotFoundError(f"No .npy feature files found in {self.features_dir}")
        self.label = 1.0 if label == "abnormal" else 0.0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        p = self.items[idx]
        arr = np.load(p).astype(np.float32)  # (32,1024)
        x = torch.from_numpy(arr)
        y = torch.tensor(self.label, dtype=torch.float32)
        return {"x": x, "y": y, "video_id": p.stem}


@dataclass(frozen=True)
class FrameSample:
    """
    One sampled clip represented as a tensor of frames.
    Shape matches common video backbones: (T, C, H, W).
    """

    frames: torch.Tensor
    video_id: str
    label: float


class TADFramesDataset(Dataset):
    """
    Raw frames dataset.

    This is used for feature extraction (offline). It does NOT return (32,1024)
    features; instead it returns uniformly sampled frames per segment.
    """

    def __init__(
        self,
        frames_root: Path,
        label: LabelName,
        num_segments: int = 32,
        frames_per_segment: int = 16,
        image_size: int = 224,
    ):
        self.frames_root = frames_root
        self.label_name: LabelName = label
        self.num_segments = int(num_segments)
        self.frames_per_segment = int(frames_per_segment)
        self.image_size = int(image_size)

        self.video_ids = list_video_ids(frames_root, label)
        self.label = 1.0 if label == "abnormal" else 0.0

    def __len__(self) -> int:
        return len(self.video_ids)

    def _load_frame(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        x = torch.from_numpy(np.array(img)).to(torch.float32) / 255.0  # (H,W,C)
        x = x.permute(2, 0, 1)  # (C,H,W)
        return x

    def __getitem__(self, idx: int) -> FrameSample:
        video_id = self.video_ids[idx]
        d = self.frames_root / self.label_name / video_id
        frames = sorted([p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
        if len(frames) < self.num_segments:
            raise RuntimeError(f"Not enough frames in {d} ({len(frames)}) for {self.num_segments} segments")

        # split into segments and uniformly sample frames_per_segment each
        seg_edges = np.linspace(0, len(frames), num=self.num_segments + 1, dtype=int)
        sampled: list[torch.Tensor] = []
        for i in range(self.num_segments):
            a, b = seg_edges[i], seg_edges[i + 1]
            seg = frames[a:b] if b > a else frames[a : a + 1]
            if not seg:
                seg = frames[a : a + 1]
            idxs = np.linspace(0, len(seg) - 1, num=self.frames_per_segment, dtype=int)
            for j in idxs:
                sampled.append(self._load_frame(seg[int(j)]))

        # (N*T, C, H, W) -> treat as (T_total, C, H, W) for backbone processing
        clip = torch.stack(sampled, dim=0)
        return FrameSample(frames=clip, video_id=video_id, label=self.label)

