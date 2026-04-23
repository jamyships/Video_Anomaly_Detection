# TAD Anomaly Detection — Full Architecture Documentation
## Based on Pu et al. (2023): Prompt-Enhanced Learning (PEL) Framework

---

## 1. What We Are Building and Why

This project implements the **Prompt-Enhanced Learning (PEL)** framework for
video anomaly detection on the **TAD (Traffic Anomaly Detection)** dataset.

The architecture originates from the paper:
> "Learning Prompt-Enhanced Context Features for Weakly-Supervised Video Anomaly Detection"
> Pu, Wu & Wang (2023), arXiv:2306.14451

### Why this architecture?
The architecture in the diagram (Fig. 4 from the paper) is the **most complete
and modern** VAD pipeline described in the blog post. It combines:
- A strong video feature extractor (I3D)
- Temporal context modeling (TCA)
- Causal convolution for sequence prediction (CC)
- CLIP-based semantic guidance (PEL)
- Score smoothing to reduce false alarms (SS)
- Weakly supervised training via Multiple Instance Learning (MIL)

---

## 2. Architecture — Component by Component

Reading the diagram LEFT → RIGHT (both the Abnormal Vₐ and Normal Vₙ branches
share identical structure via **weight sharing**):

```
Abnormal Vₐ ──► I3D ──► TCA ──► MLP ──► CC ──► Sₐ ──►
                                   ↘                    SS ──► MIL ──► L_ce
                                   PEL (L_kd)
Normal Vₙ   ──► I3D ──► TCA ──► MLP ──► CC ──► Sₙ ──►
```

### 2.1 I3D — Video Feature Extractor
**What it is:** Inflated 3D ConvNet pre-trained on Kinetics-400.
**What it does:** Takes a clip of T frames and produces a 1024-dim feature
  vector capturing BOTH spatial (what is happening) and temporal (motion)
  information.
**Why here:** Raw pixels are too noisy and high-dimensional for anomaly scoring.
  We need compact, semantically rich representations. I3D, pre-trained on a large
  action recognition dataset, already "understands" human and vehicle motions.
**In TAD context:** It will encode "car turning normally" vs "car having an accident"
  into different regions of the 1024-dim space.
**Output shape:** (B, N, 1024) where N = number of segments per video.

### 2.2 TCA — Temporal Context Aggregation
**What it is:** A multi-head self-attention block (Transformer encoder layer).
**What it does:** Each of the N segments attends to ALL other segments in the
  same video. This lets the model understand a segment in the context of what
  came before and after it.
**Why here:** Anomalies in traffic scenes are temporal — a car driving normally
  for 5 seconds then suddenly swerving is only anomalous when you consider
  the temporal trajectory. Pure per-segment scoring misses this.
**Why self-attention specifically:** Unlike RNNs, attention is parallelizable and
  captures long-range dependencies without vanishing gradients.
**Output shape:** (B, N, D) — same shape as input, but each feature now
  encodes context from the full temporal window.

### 2.3 MLP — Multilayer Perceptron (Feature Projection)
**What it is:** Two linear layers with ReLU and dropout.
**What it does:** Projects the TCA output into a lower-dimensional space (512-d)
  that is more suitable for anomaly scoring.
**Why here:** Acts as a learned adapter between the I3D feature space and the
  task-specific anomaly scoring space. Also regularizes via dropout.
**Output shape:** (B, N, 512)

### 2.4 CC — Causal Convolution
**What it is:** 1D causal convolution over the temporal dimension.
**What it does:** Applies a convolution where each position can only attend to
  PAST segments (not future). This outputs a per-segment anomaly score.
**Why causal:** In a real deployment, you cannot look into the future. Causal
  convolutions ensure the model is deployable in streaming/real-time settings.
**Why convolution (not attention again):** Convolutions are computationally
  efficient for local temporal smoothing — they aggregate over a sliding window
  of k past segments, enforcing that anomaly scores vary smoothly over time.
**Output shape:** (B, N, 1) — one anomaly score per segment.

### 2.5 PEL — Prompt-Enhanced Learning (CLIP guidance)
**What it is:** A knowledge distillation loss using CLIP text embeddings.
**What it does:**
  - Queries ConceptNet (or uses hard-coded synonyms) for words related to
    traffic anomalies (e.g., "accident", "collision", "crash", "overturn")
  - Uses CLIP's text encoder to embed these words into a 512-d semantic space
  - Trains the MLP output features of anomalous segments to be CLOSE to
    anomaly text embeddings and FAR from normal text embeddings
  - This is the L_kd (knowledge distillation) loss
**Why here:** CLIP provides "free" semantic supervision without needing
  frame-level labels. It tells the model: "a feature that looks like an accident
  should be similar to the word 'accident'". This significantly boosts
  performance, especially for rare anomaly types.
**Why only between the two branches (not split):** PEL compares the MLP
  features of abnormal and normal videos, using the CLIP semantic space as a
  reference anchor.

### 2.6 SS — Score Smoothing
**What it is:** A Gaussian/moving-average temporal smoother applied at TEST time.
**What it does:** Averages anomaly scores over a small window (±k frames) to
  eliminate sharp spikes from individual noisy frames.
**Why here:** Frame jitters (lighting changes, compression artifacts) can cause
  brief high-error frames that trigger false alarms. Smoothing removes these
  without hurting recall for real anomalies (which persist for many frames).
**Only at test time:** During training, raw scores are used for the MIL loss.

### 2.7 MIL — Multiple Instance Learning
**What it is:** A ranking-based loss function (not a module with weights).
**What it does:** 
  - Treats each video as a "bag" of N segments
  - Positive bag = anomalous video (at least one anomalous segment)
  - Negative bag = normal video (all segments are normal)
  - MIL ranking loss: max_score(positive_bag) > max_score(negative_bag)
  - Also adds temporal smoothness + sparsity regularization
**Why not cross-entropy?** We don't have frame-level labels — only video-level
  labels (this whole video has/doesn't have an anomaly). MIL allows us to train
  a frame-level predictor using only these weaker video-level labels.
**Loss:** L_ce in the diagram = cross-entropy variant of MIL (Pu et al. use
  a cross-entropy loss on top-k segment scores, unlike Sultani's ranking loss)

---

## 3. Training Strategy

### Weight Sharing
Both the Abnormal (Vₐ) and Normal (Vₙ) branches share ALL weights
(I3D, TCA, MLP, CC). This means:
- The model processes one video at a time (not pairs)
- During a training step, both an anomalous and a normal video are fed
- Gradients flow through both paths, updating the same weights

### Two Losses
1. **L_kd (PEL loss):** Knowledge distillation from CLIP. Aligns MLP features
   of anomalous segments toward anomaly text embeddings.
2. **L_ce (MIL loss):** Cross-entropy on predicted scores. Pushes scores of
   anomalous videos higher than normal videos.
Total loss = L_ce + λ * L_kd

### Test / Inference
- Only the I3D → TCA → MLP → CC path is used (no PEL)
- Score Smoothing (SS) is applied
- Threshold = 0.5 on scores ∈ [0, 1]

---

## 4. Dataset: TAD (Traffic Anomaly Detection)

- 500 videos total: 250 abnormal, 250 normal
- Each clip ~1,075 frames on average
- Anomalies span ~80 frames
- Only VIDEO-LEVEL labels for training (not frame-level) — perfect for MIL
- Frame-level labels available for TEST SET only — for evaluation

### Why TAD fits this architecture perfectly
- Traffic anomalies (accidents, illegal turns) are SHORT events in otherwise
  normal videos — exactly the scenario MIL was designed for
- The temporal context of TCA is crucial: you need to know a car was driving
  normally before judging an event as anomalous
- CLIP prompts can be traffic-specific: "car accident", "collision", "illegal turn"

---

## 5. Feature Extraction Strategy

**Problem:** Running I3D on all 500 videos × ~1075 frames is computationally
heavy but only needs to be done ONCE.

**Solution:** Pre-extract I3D features offline → save as .npy files → 
train only the lightweight TCA + MLP + CC pipeline.

This is the standard approach used in all MIL-based VAD papers and is why
they can train on a single GPU in minutes rather than hours.

**Segment strategy:**
- Each video is divided into T=32 non-overlapping segments
- Each segment = (total_frames / 32) consecutive frames
- From each segment, 16 frames are uniformly sampled → fed to I3D
- I3D output: 1024-dim vector per segment
- Final shape per video: (32, 1024)

---

## 6. File Structure

```
TAD_PEL/
├── scripts/
│   ├── config.py           ← All hyperparameters in one place
│   ├── dataset.py          ← TAD dataset loader + feature extraction prep
│   ├── i3d_extractor.py    ← I3D feature extraction (run once)
│   ├── model.py            ← TCA + MLP + CC + PEL modules
│   ├── losses.py           ← MIL loss + PEL loss
│   ├── train.py            ← Training loop
│   └── eval.py             ← Evaluation + ROC-AUC + visualizations
├── notebooks/
│   └── TAD_PEL_full.ipynb  ← Complete guided notebook
├── models/                 ← Saved checkpoints
├── results/                ← Plots, metrics
├── docs/
│   └── PROJECT_ARCHITECTURE.md  ← This file
└── requirements.txt
```
