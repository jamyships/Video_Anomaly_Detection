from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MilLossConfig:
    topk: int = 3
    smoothness_weight: float = 8e-4
    sparsity_weight: float = 8e-4


class MilTopKBCELoss(nn.Module):
    """
    MIL loss used for weak labels.

    This implements a pragmatic "L_ce" variant:
    - For each video bag (N segments), select top-k segment logits
    - Aggregate (mean) top-k logits into a bag logit
    - Apply BCEWithLogitsLoss with the video-level label (0 normal, 1 abnormal)

    Regularization:
    - temporal smoothness on scores
    - sparsity on scores (rare anomalies assumption)
    """

    def __init__(self, cfg: MilLossConfig):
        super().__init__()
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        logits: (B, N) raw logits per segment
        label:  (B,) 0/1 float
        """
        if logits.dim() != 2:
            raise ValueError("logits must be (B, N)")
        if label.dim() != 1:
            raise ValueError("label must be (B,)")

        b, n = logits.shape
        k = min(int(self.cfg.topk), n)
        topk_logits, _ = torch.topk(logits, k=k, dim=1)
        bag_logit = topk_logits.mean(dim=1)  # (B,)

        label = label.to(dtype=bag_logit.dtype)
        ce = self.bce(bag_logit, label)

        scores = torch.sigmoid(logits)
        smooth = ((scores[:, 1:] - scores[:, :-1]) ** 2).mean()
        sparse = scores.mean()

        total = ce + self.cfg.smoothness_weight * smooth + self.cfg.sparsity_weight * sparse
        return {"total": total, "ce": ce, "smooth": smooth, "sparse": sparse}


class PelKDLoss(nn.Module):
    """
    PEL (L_kd): Knowledge distillation aligning projected visual features to CLIP text prompts.

    - Takes projected features (B, N, 512)
    - Computes similarity to anomaly vs normal prompt embeddings
    - Encourages abnormal videos' top-k segments to align with anomaly prompts (and away from normal)
      and normal videos to align with normal prompts.

    This is a lightweight, reproducible approximation that keeps the intent of the paper:
    use CLIP's semantic space as weak supervision without frame-level labels.
    """

    def __init__(self, anomaly_text: torch.Tensor, normal_text: torch.Tensor, temperature: float = 0.07):
        super().__init__()
        self.register_buffer("anomaly_text", F.normalize(anomaly_text, dim=-1))
        self.register_buffer("normal_text", F.normalize(normal_text, dim=-1))
        self.temperature = temperature

    def forward(self, proj: torch.Tensor, video_label: torch.Tensor, topk: int = 3) -> torch.Tensor:
        """
        proj: (B, N, D) projected visual features (expected D=512)
        video_label: (B,) float {0,1} where 1 means abnormal
        """
        if proj.dim() != 3:
            raise ValueError("proj must be (B, N, D)")
        b, n, d = proj.shape
        k = min(int(topk), n)

        v = F.normalize(proj, dim=-1)  # (B,N,D)
        # Prompt prototypes: average across prompts
        a = self.anomaly_text.mean(dim=0, keepdim=True)  # (1,D)
        nrm = self.normal_text.mean(dim=0, keepdim=True)  # (1,D)
        a = F.normalize(a, dim=-1)
        nrm = F.normalize(nrm, dim=-1)

        # Similarities per segment
        sim_a = (v @ a.t()).squeeze(-1) / self.temperature  # (B,N)
        sim_n = (v @ nrm.t()).squeeze(-1) / self.temperature  # (B,N)
        # Build 2-class logits [normal, anomaly]
        logits2 = torch.stack([sim_n, sim_a], dim=-1)  # (B,N,2)

        # Select segments with highest anomaly score to focus KD on likely anomalous regions
        topk_idx = torch.topk(sim_a, k=k, dim=1).indices  # (B,k)
        gather = topk_idx[..., None].expand(-1, -1, 2)  # (B,k,2)
        top_logits2 = torch.gather(logits2, dim=1, index=gather)  # (B,k,2)
        top_logits2 = top_logits2.reshape(b * k, 2)

        target = video_label.to(dtype=torch.long).repeat_interleave(k)  # 0/1
        return F.cross_entropy(top_logits2, target)

