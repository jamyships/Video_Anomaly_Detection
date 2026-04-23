from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.config import Config
from scripts.dataset import TADFeatureDataset
from scripts.model import VADHead, score_smoothing


@torch.no_grad()
def main() -> None:
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load latest checkpoint if present
    ckpts = sorted(Path("models").glob("pel_vad_epoch*.pt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in `models/`. Run training first.")
    ckpt_path = ckpts[-1]
    # Checkpoints are created locally by this repo; we explicitly allow full load.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = VADHead(
        feature_dim=cfg.i3d_feature_dim,
        tca_heads=cfg.tca_heads,
        tca_layers=cfg.tca_layers,
        tca_dropout=cfg.tca_dropout,
        mlp_dim=cfg.mlp_dim,
        cc_kernel=cfg.cc_kernel,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    ds_n = TADFeatureDataset(cfg.features_root, "normal")
    ds_a = TADFeatureDataset(cfg.features_root, "abnormal")
    dl = DataLoader(ds_n, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    dla = DataLoader(ds_a, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    y_true: list[float] = []
    y_score: list[float] = []

    def run(loader):
        for batch in tqdm(loader, desc="eval"):
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy().tolist()
            out = model(x)
            # Bag score = max segment score after optional smoothing
            s = score_smoothing(out["scores"], kernel_size=5)
            bag = s.max(dim=1).values.detach().cpu().numpy().tolist()
            y_true.extend(y)
            y_score.extend(bag)

    run(dl)
    run(dla)

    auc = roc_auc_score(np.array(y_true), np.array(y_score))
    fpr, tpr, _ = roc_curve(np.array(y_true), np.array(y_score))
    print(f"Checkpoint: {ckpt_path}")
    print(f"ROC-AUC (video-level, max over segments): {auc:.4f}")

    Path("results").mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (video-level)")
    plt.legend()
    out_path = Path("results") / "roc_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

