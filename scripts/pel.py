from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ClipTextEmbeddings:
    anomaly: torch.Tensor  # (P_a, D)
    normal: torch.Tensor  # (P_n, D)


def build_clip_text_embeddings(
    prompts_anomaly: Iterable[str],
    prompts_normal: Iterable[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cuda",
) -> ClipTextEmbeddings:
    """
    Loads CLIP (via `open_clip`) and returns normalized text embeddings.

    Why `open_clip`:
    - Reproducible, pip-installable, works on Windows
    - Supports OpenAI-pretrained CLIP weights
    """

    try:
        import open_clip
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "open_clip_torch is required for PEL. Install with `pip install open_clip_torch`."
        ) from e

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        ta = tokenizer(list(prompts_anomaly)).to(device)
        tn = tokenizer(list(prompts_normal)).to(device)
        ea = model.encode_text(ta)
        en = model.encode_text(tn)
        ea = ea / ea.norm(dim=-1, keepdim=True)
        en = en / en.norm(dim=-1, keepdim=True)
    return ClipTextEmbeddings(anomaly=ea, normal=en)

