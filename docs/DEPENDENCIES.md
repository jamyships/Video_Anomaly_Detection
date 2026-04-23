# Dependencies (What we use, where, and why)

This repo is intentionally small: it implements a PEL-style weakly supervised VAD pipeline
on cached per-segment features. This document explains every library in `requirements.txt`,
what functionality it provides, and **exactly where it’s used** in the code.

---

## Core deep learning

### `torch`
**What it provides**
- Tensors, GPU execution (when CUDA build is installed), autograd, optimizers, neural network layers.

**Where we use it**
- Everywhere in the ML stack:
  - Model blocks: `scripts/model.py`
  - Losses: `scripts/losses.py`
  - Training: `scripts/train.py`
  - Evaluation: `scripts/eval.py`
  - CLIP text embedding extraction: `scripts/pel.py`

**Why it’s relevant**
- This project is a PyTorch implementation of the architecture. Every trainable module and loss is PyTorch.

### `torchvision`
**What it provides**
- Pretrained models, including 3D video backbones; image/video utilities.

**Where we use it**
- Feature extraction (baseline backbone):
  - `scripts/i3d_extractor.py` uses `torchvision.models.video.r3d_18`.

**Why it’s relevant**
- Provides a dependable baseline feature extractor that works on Windows via pip.

---

## Data + numerics

### `numpy`
**What it provides**
- Fast CPU arrays and file format I/O for `.npy`.

**Where we use it**
- Loading/saving cached features:
  - `scripts/dataset.py` (`np.load`)
  - `scripts/i3d_extractor.py` (`np.save`)
- Frame sampling utilities:
  - `scripts/dataset.py` uses `np.linspace` to slice segments and pick frames.

**Why it’s relevant**
- Cached feature files are `.npy`, and segment sampling is simpler with numpy utilities.

### `Pillow`
**What it provides**
- Image decoding and resizing.

**Where we use it**
- Raw frame loading for feature extraction:
  - `scripts/dataset.py` → `TADFramesDataset._load_frame()`

**Why it’s relevant**
- The Kaggle TAD dump is stored as extracted frame images; we need to decode them to tensors.

---

## Training/eval ergonomics

### `tqdm`
**What it provides**
- Progress bars.

**Where we use it**
- Feature extraction and training/eval loops:
  - `scripts/i3d_extractor.py`
  - `scripts/train.py`
  - `scripts/eval.py`

**Why it’s relevant**
- Makes long-running extraction/training visible and debuggable.

### `scikit-learn`
**What it provides**
- Metrics and ROC tools.

**Where we use it**
- ROC-AUC + ROC curve:
  - `scripts/eval.py` uses `roc_auc_score`, `roc_curve`

**Why it’s relevant**
- Standard, well-tested implementation for reporting AUC.

### `matplotlib`
**What it provides**
- Plotting.

**Where we use it**
- Save ROC curve plot:
  - `scripts/eval.py` writes `results/roc_curve.png`

**Why it’s relevant**
- Quick visual sanity check for scoring separation.

---

## PEL / CLIP

### `open_clip_torch`
**What it provides**
- A widely used open-source CLIP implementation with pretrained weights and tokenizers.

**Where we use it**
- Text embedding generation:
  - `scripts/pel.py` → `build_clip_text_embeddings()`

**Why it’s relevant**
- PEL’s core idea is “semantic guidance” from CLIP text embeddings during training.

---

## I3D (paper-aligned backbone)

### `pytorchvideo`
**What it provides**
- Video model zoo and utilities, including Kinetics-pretrained I3D variants.

**Where we use it**
- Optional I3D feature extraction backend:
  - `scripts/i3d_extractor.py` → `I3DBackbone` uses `pytorchvideo.models.hub.i3d_r50(pretrained=True)`

**Why it’s relevant**
- This is the closest “pip-installable” path to a Kinetics-pretrained I3D-style extractor on PyTorch.

### `fvcore`, `iopath`
**What they provide**
- Utility dependencies used internally by PyTorchVideo/Detectron2-family tooling (config, IO, checkpoints).

**Where we use them**
- Indirectly required when using `pytorchvideo`.

**Why they’re relevant**
- They enable `pytorchvideo` to load models/weights reliably.

---

## GPU note (Windows + Python version)

If your environment uses Python 3.13, pip may install a **CPU-only** PyTorch build.
GPU usage requires installing a CUDA-enabled PyTorch build for a Python version with
CUDA wheels available (commonly 3.10–3.12).

