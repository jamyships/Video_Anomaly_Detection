from __future__ import annotations

"""
Offline feature extraction.

The architecture documentation expects per-video features shaped (32, 1024).
This script produces that cache from raw frames.

Note:
- We support two backbones:
  - `baseline`: torchvision `r3d_18` + projection to 1024 (fast, always available)
  - `i3d`: Kinetics-pretrained I3D via `pytorchvideo` (closer to the paper)

Both backbones write cached `.npy` features shaped `(32, 1024)` so the downstream
TCA/MLP/CC + MIL/PEL training code is unchanged.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.config import Config, ensure_dirs
from scripts.dataset import TADFramesDataset


class BaselineBackbone(nn.Module):
    """
    A lightweight, reproducible stand-in for I3D feature extraction.

    - Uses a 3D CNN from torchvision to get a clip embedding
    - Projects embedding to 1024 dims
    """

    def __init__(self, out_dim: int = 1024):
        super().__init__()
        from torchvision.models.video import r3d_18, R3D_18_Weights

        m = r3d_18(weights=R3D_18_Weights.DEFAULT)
        # remove classification head
        self.backbone = nn.Sequential(*(list(m.children())[:-1]))  # -> (B, 512, 1, 1, 1)
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        z = self.backbone(x).flatten(1)  # (B, 512)
        return self.proj(z)  # (B, out_dim)


class I3DBackbone(nn.Module):
    """
    I3D feature extractor via PyTorchVideo hub.

    We keep the contract: output embedding is projected to 1024 dims.
    """

    def __init__(self, out_dim: int = 1024):
        super().__init__()
        try:
            from pytorchvideo.models.hub import i3d_r50
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "pytorchvideo is required for --backbone i3d. Install dependencies from requirements.txt."
            ) from e

        m = i3d_r50(pretrained=True)  # Kinetics-400 pretrained
        # Remove classification head; keep feature trunk.
        # PyTorchVideo hub models expose `blocks`; last block is usually classification.
        if hasattr(m, "blocks") and len(m.blocks) >= 2:
            self.trunk = nn.Sequential(*list(m.blocks[:-1]))
            trunk_out = 2048
        else:  # pragma: no cover
            # Fallback: treat everything except final projection as trunk
            self.trunk = nn.Sequential(*(list(m.children())[:-1]))
            trunk_out = 2048

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Linear(trunk_out, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        z = self.trunk(x)
        if z.dim() == 5:
            z = self.pool(z).flatten(1)
        else:  # pragma: no cover
            z = z.flatten(1)
        return self.proj(z)


def _segment_mean(features: torch.Tensor, num_segments: int, frames_per_segment: int) -> torch.Tensor:
    """
    Convert per-frame/clip features into per-segment features by averaging
    each segment's frames_per_segment features.

    features: (num_segments*frames_per_segment, D)
    returns:  (num_segments, D)
    """
    d = features.shape[-1]
    x = features.view(num_segments, frames_per_segment, d).mean(dim=1)
    return x


@torch.no_grad()
def extract_for_label(cfg: Config, label: str, backbone: str, batch_size: int = 1) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ds = TADFramesDataset(
        frames_root=cfg.dataset_root,
        label=label,  # type: ignore[arg-type]
        num_segments=cfg.num_segments,
        frames_per_segment=cfg.frames_per_segment,
        image_size=224,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    if backbone == "i3d":
        model = I3DBackbone(out_dim=cfg.i3d_feature_dim).to(device)
    else:
        model = BaselineBackbone(out_dim=cfg.i3d_feature_dim).to(device)
    model.eval()

    out_dir = cfg.features_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(dl, desc=f"Extracting {label}"):
        # FrameSample is a dataclass; DataLoader will collate into dict-like structure
        frames = batch["frames"].to(device)  # (B, T_total, C, H, W)
        video_ids = batch["video_id"]

        b, t_total, c, h, w = frames.shape
        if backbone == "i3d":
            # Feed per-segment clips (16 frames) to I3D.
            # frames: (B, N*F, C, H, W) -> (B, N, F, C, H, W)
            xseg = frames.view(b, cfg.num_segments, cfg.frames_per_segment, c, h, w)
            # (B,N,F,C,H,W) -> (B*N, C, F, H, W)
            x = xseg.permute(0, 1, 3, 2, 4, 5).reshape(b * cfg.num_segments, c, cfg.frames_per_segment, h, w)
            feat_seg = model(x).view(b, cfg.num_segments, -1)  # (B,N,1024)
            feat = feat_seg
        else:
            # Baseline: process each sampled frame as a tiny 3D clip of length 1:
            # (B*T_total, C, 1, H, W) and average per segment.
            x = frames.reshape(b * t_total, c, h, w).unsqueeze(2)
            feat = model(x).view(b, t_total, -1)

        for i in range(b):
            if backbone == "i3d":
                seg = feat[i].cpu().numpy().astype(np.float32)
            else:
                seg = _segment_mean(feat[i], cfg.num_segments, cfg.frames_per_segment).cpu().numpy().astype(np.float32)
            np.save(out_dir / f"{Path(video_ids[i]).stem}.npy", seg)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract (32,1024) cached features from TAD frames.")
    p.add_argument(
        "--backbone",
        choices=["baseline", "i3d"],
        default="baseline",
        help="Feature extractor backbone. Use `i3d` for Kinetics-pretrained I3D via pytorchvideo.",
    )
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Config()
    ensure_dirs(cfg)
    extract_for_label(cfg, "normal", backbone=args.backbone, batch_size=int(args.batch_size))
    extract_for_label(cfg, "abnormal", backbone=args.backbone, batch_size=int(args.batch_size))


if __name__ == "__main__":
    main()

