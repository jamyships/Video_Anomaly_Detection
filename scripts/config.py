from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """
    Central configuration for the PEL-style MIL VAD pipeline described in
    `PROJECT_ARCHITECTURE.md`.

    Key design choice:
    - Training runs on pre-extracted per-segment features of shape (N=32, F=1024).
      Feature extraction can be run once from the raw frames and cached.
    """

    # ---- Dataset (raw frames) ----
    dataset_root: Path = Path(
        r"C:\Users\backupadmin\.cache\kagglehub\datasets\nikanvasei\traffic-anomaly-dataset-tad\versions\1\TAD\frames"
    )

    # ---- Cached features (recommended for training speed) ----
    features_root: Path = Path("data") / "features"
    # expected layout:
    # data/features/{abnormal,normal}/{video_id}.npy  -> (32, 1024) float32

    # ---- Temporal segmentation ----
    num_segments: int = 32
    frames_per_segment: int = 16  # uniformly sampled per segment for backbone

    # ---- Model dims ----
    i3d_feature_dim: int = 1024
    tca_dim: int = 1024
    mlp_dim: int = 512

    # ---- TCA (self-attention) ----
    tca_heads: int = 8
    tca_layers: int = 1
    tca_dropout: float = 0.1

    # ---- CC (causal conv) ----
    cc_kernel: int = 5

    # ---- MIL / training ----
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    topk: int = 3  # top-k segments for MIL cross-entropy style supervision
    lambda_kd: float = 0.1
    smoothness_weight: float = 8e-4
    sparsity_weight: float = 8e-4

    # ---- Device ----
    device: str = "cuda"

    # ---- PEL / CLIP ----
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    prompt_anomaly: tuple[str, ...] = (
        "traffic accident",
        "car crash",
        "collision",
        "illegal turn",
        "road spill",
        "pedestrian on road",
        "vehicle retrograde",
    )
    prompt_normal: tuple[str, ...] = (
        "normal traffic",
        "cars driving normally",
        "normal road scene",
        "safe driving",
        "traffic flow",
    )


def ensure_dirs(cfg: Config) -> None:
    (Path("models")).mkdir(parents=True, exist_ok=True)
    (Path("results")).mkdir(parents=True, exist_ok=True)
    (Path("data") / "features" / "abnormal").mkdir(parents=True, exist_ok=True)
    (Path("data") / "features" / "normal").mkdir(parents=True, exist_ok=True)

