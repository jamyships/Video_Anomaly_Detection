from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalContextAggregation(nn.Module):
    """
    TCA: Temporal Context Aggregation via self-attention.

    Input:  (B, N, D)
    Output: (B, N, D)
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FeatureMLP(nn.Module):
    """
    MLP: projects TCA features into a task-specific space.

    Input:  (B, N, D_in)
    Output: (B, N, D_out)
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalConv1d(nn.Module):
    """
    CC: causal convolution over time.

    Input:  (B, N, C)
    Output: (B, N, C_out)
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) -> (B, C, N)
        x = x.transpose(1, 2)
        left_pad = self.kernel_size - 1
        x = F.pad(x, (left_pad, 0))  # pad time dimension on the left only
        y = self.conv(x)  # (B, C_out, N)
        return y.transpose(1, 2)


class VADHead(nn.Module):
    """
    Full head (excluding I3D feature extractor):
    TCA -> MLP -> CC -> anomaly score per segment.

    Inputs are assumed to be pre-extracted I3D-like features of shape (B, N, 1024).
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        tca_heads: int = 8,
        tca_layers: int = 1,
        tca_dropout: float = 0.1,
        mlp_dim: int = 512,
        cc_kernel: int = 5,
    ):
        super().__init__()
        self.tca = TemporalContextAggregation(
            d_model=feature_dim, nhead=tca_heads, num_layers=tca_layers, dropout=tca_dropout
        )
        self.mlp = FeatureMLP(d_in=feature_dim, d_hidden=feature_dim, d_out=mlp_dim, dropout=0.5)
        self.cc = CausalConv1d(c_in=mlp_dim, c_out=1, kernel_size=cc_kernel)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns:
        - proj: (B, N, 512) projected features (used by PEL KD)
        - logits: (B, N) raw anomaly logits per segment
        - scores: (B, N) sigmoid scores in [0,1]
        """
        x_ctx = self.tca(x)
        proj = self.mlp(x_ctx)
        logits = self.cc(proj).squeeze(-1)
        scores = torch.sigmoid(logits)
        return {"proj": proj, "logits": logits, "scores": scores}


def score_smoothing(scores: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    SS: simple moving-average smoothing (test-time only).

    scores: (B, N) or (N,)
    """
    if scores.dim() == 1:
        scores_ = scores[None, :, None]  # (1,N,1)
        squeeze = True
    elif scores.dim() == 2:
        scores_ = scores[:, :, None]  # (B,N,1)
        squeeze = False
    else:
        raise ValueError("scores must be 1D or 2D")

    k = int(kernel_size)
    if k <= 1:
        out = scores_
    else:
        pad = k // 2
        x = scores_.transpose(1, 2)  # (B,1,N)
        x = F.pad(x, (pad, pad), mode="replicate")
        w = torch.ones(1, 1, k, device=scores.device, dtype=scores.dtype) / k
        out = F.conv1d(x, w).transpose(1, 2)

    out = out.squeeze(-1)
    return out[0] if squeeze else out

