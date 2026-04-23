# Code-level map (Architecture → files → functions)

This document is the “traceability” layer: for every architecture component, it shows
the concrete code entry points and the functions/classes involved, plus **why those
functions exist** in this particular project.

---

## 1. Data flow: end-to-end execution path

### 1.1 Feature extraction (frames → cached features)

**Entry point**
- `python -m scripts.i3d_extractor`

**Code path**
- `scripts/i3d_extractor.py`
  - `main()`: reads `Config`, ensures output dirs, runs extraction for both labels
  - `_parse_args()`: selects backbone (`baseline` vs `i3d`) and batch size
  - `extract_for_label(cfg, label, backbone, batch_size)`:
    - creates a `TADFramesDataset` for `label`
    - runs a video backbone to generate embeddings
    - saves one `.npy` per video to `data/features/<label>/...`

**Why this is needed**
- The rest of the architecture expects `(32,1024)` features. Extracting once and caching
  dramatically reduces training cost.

**Where frames come from**
- `scripts/config.py` → `Config.dataset_root`

**Which dataset code is used**
- `scripts/dataset.py` → `TADFramesDataset`
  - `list_video_ids(frames_root, label)`: discovers available videos by directory name
  - `_load_frame(path)`: decodes + resizes a frame image (Pillow)
  - `__getitem__(idx)`: segments frames into 32 bins, samples 16 frames each

---

### 1.2 Training (cached features → model parameters)

**Entry point**
- `python -m scripts.train`

**Code path**
- `scripts/train.py`
  - `_parse_args()`: allows overriding device/epochs/batch-size; `--no-pel` disables CLIP KD
  - `cycle(dl)`: makes an infinite iterator so we can always pair normal + abnormal batches
  - `main()`:
    - loads `Config`
    - builds datasets:
      - `TADFeatureDataset(features_root, "normal")`
      - `TADFeatureDataset(features_root, "abnormal")`
    - instantiates the model: `VADHead(...)`
    - instantiates losses:
      - `MilTopKBCELoss(...)`
      - optionally `PelKDLoss(...)` (if not `--no-pel`)
    - performs paired-batch training steps (weight sharing)
    - saves checkpoints to `models/pel_vad_epoch*.pt`

**Why normal/abnormal are loaded separately**
- It mirrors the diagram’s two branches (shared weights) while ensuring each step sees
  both classes (MIL is defined over weak video labels).

**Which dataset code is used**
- `scripts/dataset.py` → `TADFeatureDataset`
  - `__getitem__`: loads `.npy` → returns `{"x": (32,1024), "y": 0/1, "video_id": ...}`

---

### 1.3 Inference / evaluation (scores → ROC-AUC plot)

**Entry point**
- `python -m scripts.eval`

**Code path**
- `scripts/eval.py`
  - `main()`:
    - loads latest checkpoint from `models/`
    - rebuilds `VADHead`, loads weights
    - runs the model on normal and abnormal feature sets
    - converts segment scores → video score via `max(scores)`
    - computes ROC-AUC + saves ROC curve plot

**Why we use `max` over segments**
- In MIL VAD, a video is abnormal if **at least one** segment is abnormal; max is the
  simplest “at least one” aggregator and matches classic MIL intuition.

---

## 2. Model blocks (diagram components)

All model code is in `scripts/model.py`.

### 2.1 TCA — `TemporalContextAggregation`

**Function/class used**
- `class TemporalContextAggregation(nn.Module)`
  - `__init__(d_model, nhead, num_layers, dropout)`
  - `forward(x)`

**Inputs/outputs**
- input `x`: `(B, N, D)` where `N=32`, `D=1024`
- output: `(B, N, D)` same shape, context-enriched

**Why these functions exist**
- `__init__`: wires up a `TransformerEncoder` with the right attention dimension
- `forward`: makes TCA a reusable block that can be swapped/extended later

---

### 2.2 MLP — `FeatureMLP`

**Function/class used**
- `class FeatureMLP(nn.Module)`
  - `__init__(d_in, d_hidden, d_out, dropout)`
  - `forward(x)`

**Inputs/outputs**
- `(B, N, 1024)` → `(B, N, 512)`

**Why this exists**
- It adapts generic backbone features into a task- and PEL-friendly embedding space.

---

### 2.3 CC — `CausalConv1d`

**Function/class used**
- `class CausalConv1d(nn.Module)`
  - `__init__(c_in, c_out, kernel_size)`
  - `forward(x)`

**Inputs/outputs**
- input `(B, N, C)` → output `(B, N, C_out)`

**Why we left-pad**
- Ensures causal dependency (time \(t\) never uses future \(t+1\)).

---

### 2.4 Score head — `VADHead`

**Function/class used**
- `class VADHead(nn.Module)`
  - `__init__(...)`: constructs `tca`, `mlp`, `cc`
  - `forward(x)` returns a dict:
    - `proj`: `(B,N,512)` (for PEL)
    - `logits`: `(B,N)` (for MIL)
    - `scores`: `(B,N)` sigmoid(logits)

**Why `forward` returns a dict**
- Training needs both:
  - `logits` for MIL
  - `proj` for PEL KD

Returning both avoids recomputation and keeps training code explicit.

---

### 2.5 SS — `score_smoothing`

**Function used**
- `score_smoothing(scores, kernel_size=5)`

**Why it’s a standalone function**
- It has no trainable parameters and is only applied at inference/eval time.

---

## 3. Losses (diagram labels)

All losses are in `scripts/losses.py`.

### 3.1 MIL — `MilTopKBCELoss`

**Functions/classes used**
- `MilLossConfig`: holds top-k + regularizer weights
- `MilTopKBCELoss.forward(logits, label)` returns:
  - `total`, `ce`, `smooth`, `sparse`

**Why top-k aggregation exists**
- Because weak labels do not tell which segments are anomalous; top-k approximates
  “the abnormal video contains some abnormal segments.”

---

### 3.2 PEL KD — `PelKDLoss`

**Functions/classes used**
- `PelKDLoss.__init__(anomaly_text, normal_text, temperature)`
- `PelKDLoss.forward(proj, video_label, topk)`

**Why CLIP prompts are encoded once**
- CLIP encoding is expensive; embeddings are reused across batches.

---

## 4. CLIP prompt embedding (PEL support)

All CLIP support code is in `scripts/pel.py`.

### 4.1 `build_clip_text_embeddings(...)`

**What it does**
- Loads CLIP via `open_clip_torch`
- Tokenizes prompt strings
- Encodes them to normalized 512-d vectors

**Why this function exists**
- It isolates CLIP loading/tokenization so the training loop stays focused on VAD logic.

---

## 5. Configuration (why a single `Config` object)

Configuration is in `scripts/config.py`.

**Functions/classes used**
- `Config`: dataclass holding:
  - dataset paths
  - segmentation parameters (N=32, frames/segment=16)
  - model dims and hyperparameters
  - CLIP prompts
- `ensure_dirs(cfg)`: creates `models/`, `results/`, and `data/features/...`

**Why it exists**
- Keeps the implementation consistent with the architecture doc and prevents “magic numbers”
  scattered across files.

