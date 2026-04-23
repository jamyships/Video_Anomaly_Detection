# Implementation Guide (Comprehensive)

This file is the **“make sense of everything”** document: it ties the **architecture diagram**
to the **actual code in this repo**, explains **what each component does**, **why it’s used**
in Video Anomaly Detection (VAD), and how data flows end-to-end.

If you’re starting from scratch, read in this order:
- `docs/PROJECT_ARCHITECTURE.md` (what we’re building conceptually)
- `docs/IMPLEMENTATION_GUIDE.md` (how it’s implemented here, file-by-file)

---

## 1. The big picture (diagram → implementation)

### 1.1 Diagram recap (left → right)

Both branches share weights (the “abnormal” and “normal” paths are the same model):

```
V  ──► I3D ──► TCA ──► MLP ──► CC ──► S  ──► (SS at test) ──► MIL (L_ce)
                       ↘
                        PEL (L_kd)  [training only]
```

### 1.2 Where each block lives in this repo

- **I3D (feature extractor; offline caching)**: `scripts/i3d_extractor.py`
- **TCA (Temporal Context Aggregation)**: `scripts/model.py` → `TemporalContextAggregation`
- **MLP projection head**: `scripts/model.py` → `FeatureMLP`
- **CC (Causal Convolution)**: `scripts/model.py` → `CausalConv1d`
- **Scores \(S\)**: `scripts/model.py` → `VADHead.forward()` returns `logits` + `scores`
- **SS (Score Smoothing; inference only)**: `scripts/model.py` → `score_smoothing`
- **MIL loss \(L_ce\)**: `scripts/losses.py` → `MilTopKBCELoss`
- **PEL \(L_kd\)**:
  - CLIP text embeddings: `scripts/pel.py` → `build_clip_text_embeddings`
  - KD loss: `scripts/losses.py` → `PelKDLoss`
- **Training loop**: `scripts/train.py`
- **Evaluation**: `scripts/eval.py`
- **Central config (paths, hyperparams)**: `scripts/config.py`

---

## 2. Data: what the project uses and why

### 2.1 Dataset location (your KaggleHub download)

Your `model.ipynb` downloads to:

- `C:\Users\backupadmin\.cache\kagglehub\datasets\nikanvasei\traffic-anomaly-dataset-tad\versions\1`

Inside that root, this repo uses the **frames directory**:

- `...\versions\1\TAD\frames`

This is the default `dataset_root` in `scripts/config.py`.

### 2.2 What’s actually on disk

The Kaggle dump is organized as:

- `frames/normal/<video_id>/frame_000001.jpg ...`
- `frames/abnormal/<video_id>/frame_000001.jpg ...`

Where `<video_id>` is a directory name such as:
- `Normal_001.mp4`
- `01_Accident_001.mp4`

The code treats that directory name as the **video id**.

### 2.3 Why we cache features (`data/features/`)

Running a 3D video backbone over raw frames is expensive. MIL-style VAD training is
typically done on **pre-extracted per-segment features**.

This repo’s contract is:

- **Per video**: `(N=32, F=1024)` float32 array
- Saved as: `data/features/{normal,abnormal}/{video_id}.npy`

Once cached, training is fast: it only trains **TCA + MLP + CC** (and the losses).

### 2.4 Feature extraction vs training responsibilities

- **Feature extraction**: converts raw frames → cached features (run once)
  - `python -m scripts.i3d_extractor ...`
- **Training**: cached features → anomaly scoring model
  - `python -m scripts.train`

This separation matches how weakly-supervised VAD papers are typically run.

---

## 3. Feature extraction (“I3D”) in this repo

### 3.1 What the paper expects

The paper/architecture doc assumes:
- I3D pretrained on Kinetics
- outputs 1024-d features per segment

### 3.2 What we implement (two backends, same contract)

In `scripts/i3d_extractor.py`, you can choose:

- **`--backbone baseline`** (default)
  - uses torchvision `r3d_18` and projects to 1024
  - pro: easiest to run everywhere
  - con: not literally I3D

- **`--backbone i3d`** (paper-aligned)
  - uses PyTorchVideo’s Kinetics-pretrained I3D (`i3d_r50(pretrained=True)`)
  - then pools + projects to 1024
  - pro: much closer to the paper’s “I3D” intent
  - con: larger dependencies, heavier compute

**Both produce identical downstream inputs**: `.npy` files of shape `(32,1024)`.

### 3.3 How frames are sampled into segments

From `scripts/dataset.py` (`TADFramesDataset`):

- a video’s frames are split into **32 non-overlapping segments**
- per segment we uniformly sample **16 frames**

These sampled frames are what the backbone sees.

---

## 4. Model implementation (TCA → MLP → CC)

All model code lives in `scripts/model.py`.

### 4.1 Input / output contracts

Input to the trainable head (cached feature tensor):
- `x`: shape **(B, 32, 1024)**

Outputs:
- `proj`: shape **(B, 32, 512)** (used for PEL/CLIP KD)
- `logits`: shape **(B, 32)** (raw anomaly logits)
- `scores`: shape **(B, 32)** (`sigmoid(logits)` in \([0,1]\))

### 4.2 TCA: Temporal Context Aggregation

Implemented as a Transformer encoder:
- each segment feature attends to every other segment
- result: each segment gets a context-enriched representation

**Why it matters for TAD:** many traffic anomalies are only “anomalies” relative to
what happened earlier/later (context). Attention helps model long-range dependencies.

### 4.3 MLP projection

The MLP is a learned adapter:
- it converts generic backbone features into a space useful for anomaly scoring
- in this repo it maps **1024 → 512**

**Why 512?**
- CLIP text embeddings for common CLIP models are 512-d, so a 512-d projection is a
  practical bridge for PEL-style alignment.

### 4.4 CC: Causal Convolution

Implemented with left-padding so each time step depends only on past steps.

**Why causal?**
- It prevents “peeking into the future,” making streaming/online deployment plausible.

### 4.5 SS: Score smoothing (test-time only)

Implemented as a moving average over the temporal axis.

**Why only at test time?**
- Training uses raw segment scores so the loss can focus sharply on discriminating
  segments; smoothing at train time can blur the learning signal.

---

## 5. Training losses (MIL + PEL)

All losses live in `scripts/losses.py`.

### 5.1 MIL \(L_ce\): weakly supervised video labels → segment scoring

**Problem:** training labels are video-level (normal vs abnormal), but we need segment-level scores.

**Approach here (Top‑k MIL BCE):**
- compute per-segment logits \(z_{t}\) for \(t=1..N\)
- select top‑k logits in each video
- average them to a single “bag logit”
- apply BCEWithLogitsLoss against the video label

This matches the intent described in `docs/PROJECT_ARCHITECTURE.md`: learn segment-level
scores from video-level supervision by focusing on the most suspicious segments.

**Regularizers included (standard in MIL-VAD):**
- **smoothness**: penalizes rapid changes between adjacent segment scores
- **sparsity**: penalizes too many segments being high (anomalies are rare)

### 5.2 PEL \(L_kd\): CLIP-guided semantic alignment

PEL in this repo is implemented as:

1. Build CLIP text embeddings for:
   - anomaly prompts (e.g., “traffic accident”, “collision”, …)
   - normal prompts (e.g., “normal traffic”, …)
2. Take projected visual features `proj` (B, N, 512) and normalize them
3. Compute similarity to the anomaly prototype vs normal prototype
4. Apply cross-entropy on the **top‑k anomaly-like segments** using the video label

**Why top‑k here too?**
- In abnormal videos, only some segments are truly anomalous.
- Focusing KD on the most anomaly-like segments reduces noise.

**Where the CLIP embeddings come from:**
- `scripts/pel.py` uses `open_clip_torch` to load a CLIP model and encode prompts.

**Training-time only:** PEL affects gradients during training; inference uses only the VAD head.

---

## 6. Training loop (how weight sharing is realized)

Training code is `scripts/train.py`.

### 6.1 What “weight sharing” means in practice

The diagram shows two branches (abnormal vs normal), but they share weights.
In code, that’s implemented by:

- sampling a batch of normal videos and a batch of abnormal videos
- concatenating them into one batch
- running them through **one** `VADHead`

So both normal and abnormal examples update the same parameters.

### 6.2 End-to-end step (what happens each iteration)

1. Load cached features:
   - normal batch: `(B, 32, 1024)` with label 0
   - abnormal batch: `(B, 32, 1024)` with label 1
2. Forward pass → `logits`, `scores`, `proj`
3. Compute:
   - `MIL loss` on `logits`
   - optional `PEL KD loss` on `proj` (if enabled)
4. Backprop + optimizer step
5. Save checkpoint to `models/`

---

## 7. Evaluation (what we measure right now)

Evaluation code is `scripts/eval.py`.

Current evaluation is **video-level ROC-AUC**:
- Convert segment scores → one score per video using `max(scores)`
- Compute ROC-AUC on normal vs abnormal
- Save ROC curve plot to `results/roc_curve.png`

**Why video-level?**
- The Kaggle dump we downloaded provides the normal/abnormal split by folder, but
  does not ship frame-level annotations in the extracted tree we saw.
- Frame-level evaluation can be added once we have ground-truth temporal labels.

---

## 8. Practical notes / pitfalls

### 8.1 “Why is `data/features/` empty?”

Because it’s generated by feature extraction and ignored by git.
Populate it with:

```bash
python -m scripts.i3d_extractor
```

or paper-aligned:

```bash
python -m scripts.i3d_extractor --backbone i3d
```

### 8.2 GPU note (Windows + Python 3.13)

If PyTorch installs as CPU-only in your environment, it’s typically because CUDA wheels
aren’t available for that Python version. The code still works; to use GPU, install a CUDA
PyTorch build in a Python version with supported wheels (commonly 3.10–3.12).


