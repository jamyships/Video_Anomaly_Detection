## TAD Prompt-Enhanced Learning (PEL) — Implementation

This repo implements the architecture described in `PROJECT_ARCHITECTURE.md`:

- **I3D (feature extractor)**: done offline, cached as `(32, 1024)` features per video.
- **TCA**: temporal context aggregation via self-attention.
- **MLP**: projection to a task-specific embedding space (512-d).
- **CC**: causal temporal convolution producing per-segment anomaly logits/scores.
- **PEL (L_kd)**: CLIP-guided knowledge distillation using anomaly/normal text prompts.
- **MIL (L_ce)**: weakly supervised training with video-level labels.
- **SS**: score smoothing at inference time.

### Documentation

- `docs/PROJECT_ARCHITECTURE.md`: the conceptual architecture (diagram-level)
- `docs/IMPLEMENTATION_GUIDE.md`: comprehensive mapping of every component to code (what/why/how)

### Dataset path (KaggleHub)

Your notebook downloads TAD frames to:

- `C:\Users\backupadmin\.cache\kagglehub\datasets\nikanvasei\traffic-anomaly-dataset-tad\versions\1\TAD\frames`

This path is the default in `scripts/config.py`.

### Repo layout

- `scripts/config.py`: all hyperparameters + paths
- `scripts/dataset.py`: TAD frames loader and cached-feature loader
- `scripts/i3d_extractor.py`: offline feature extraction into `data/features/`
- `scripts/model.py`: TCA + MLP + CC (+ SS helper)
- `scripts/pel.py`: CLIP text embedding builder (open_clip)
- `scripts/losses.py`: MIL loss + PEL KD loss
- `scripts/train.py`: training loop
- `scripts/eval.py`: video-level ROC-AUC and ROC curve plot

### Quickstart

If you use the included `venv` as-is on Windows with Python 3.13, PyTorch will likely
install as **CPU-only** (because CUDA wheels may not be available for that Python yet).
The code still runs correctly on CPU; to use your GPU, create an environment with a
CUDA-supported Python (commonly 3.10–3.12) and install the CUDA-enabled PyTorch build.

Create features (run once):

```bash
python -m scripts.i3d_extractor
```

To use a Kinetics-pretrained I3D backbone (closer to the paper):

```bash
python -m scripts.i3d_extractor --backbone i3d
```

Train:

```bash
python -m scripts.train
```

Eval:

```bash
python -m scripts.eval
```

### Notes on I3D

The documentation calls for **I3D** features (1024-d, Kinetics-pretrained). This repo
keeps the **same interface** (cached `(32,1024)` features) and supports:

- `--backbone baseline`: torchvision 3D CNN + projection (fast baseline)
- `--backbone i3d`: PyTorchVideo Kinetics-pretrained I3D + projection (paper-aligned)

If you later add true I3D weights/extractor, you only need to ensure the saved `.npy`
features remain `(32,1024)` to keep everything downstream unchanged.

