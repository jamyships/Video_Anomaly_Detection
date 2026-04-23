from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.config import Config, ensure_dirs
from scripts.dataset import TADFeatureDataset
from scripts.losses import MilLossConfig, MilTopKBCELoss, PelKDLoss
from scripts.model import VADHead
from scripts.pel import build_clip_text_embeddings


def cycle(dl):
    while True:
        for x in dl:
            yield x


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PEL-style MIL VAD on cached features.")
    p.add_argument("--device", default=None, help="Override device (e.g. cpu, cuda).")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    p.add_argument("--no-pel", action="store_true", help="Disable CLIP PEL loss (L_kd).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Config()
    ensure_dirs(cfg)

    device_name = args.device or (cfg.device if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    epochs = int(args.epochs) if args.epochs is not None else cfg.epochs
    batch_size = int(args.batch_size) if args.batch_size is not None else cfg.batch_size

    # Datasets are separated by class (normal/abnormal) to enable paired MIL steps
    ds_n = TADFeatureDataset(cfg.features_root, "normal")
    ds_a = TADFeatureDataset(cfg.features_root, "abnormal")
    dl_n = DataLoader(ds_n, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    dl_a = DataLoader(ds_a, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = VADHead(
        feature_dim=cfg.i3d_feature_dim,
        tca_heads=cfg.tca_heads,
        tca_layers=cfg.tca_layers,
        tca_dropout=cfg.tca_dropout,
        mlp_dim=cfg.mlp_dim,
        cc_kernel=cfg.cc_kernel,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mil = MilTopKBCELoss(MilLossConfig(cfg.topk, cfg.smoothness_weight, cfg.sparsity_weight)).to(device)

    pel = None
    if not args.no_pel:
        clip_text = build_clip_text_embeddings(
            cfg.prompt_anomaly,
            cfg.prompt_normal,
            model_name=cfg.clip_model,
            pretrained=cfg.clip_pretrained,
            device=str(device),
        )
        pel = PelKDLoss(clip_text.anomaly, clip_text.normal).to(device)

    it_n = cycle(dl_n)
    it_a = cycle(dl_a)
    steps_per_epoch = min(len(dl_n), len(dl_a))

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch+1}/{epochs}")
        running = {"loss": 0.0, "ce": 0.0, "kd": 0.0}

        for _ in pbar:
            bn = next(it_n)
            ba = next(it_a)

            x = torch.cat([bn["x"], ba["x"]], dim=0).to(device)  # (2B,N,F)
            y = torch.cat([bn["y"], ba["y"]], dim=0).to(device)  # (2B,)

            out = model(x)
            mil_out = mil(out["logits"], y)
            kd = torch.tensor(0.0, device=device)
            loss = mil_out["total"]
            if pel is not None:
                kd = pel(out["proj"], y, topk=cfg.topk)
                loss = loss + cfg.lambda_kd * kd

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running["loss"] += float(loss.item())
            running["ce"] += float(mil_out["ce"].item())
            running["kd"] += float(kd.item())
            pbar.set_postfix(
                loss=running["loss"] / (pbar.n + 1),
                ce=running["ce"] / (pbar.n + 1),
                kd=running["kd"] / (pbar.n + 1),
            )

        ckpt = {
            "model": model.state_dict(),
            "cfg": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
            "epoch": epoch,
        }
        torch.save(ckpt, Path("models") / f"pel_vad_epoch{epoch+1}.pt")


if __name__ == "__main__":
    main()

