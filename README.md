# AMAG Neural Forecasting

Forecasting neural activity from micro-electrocorticography (µECoG) recordings using **AMAG** (Additive, Multiplicative and Adaptive Graph Neural Network) with deep multi-head transformers, channel attention, and cross-session domain adaptation.

**Competition:** NSF HDR Hackathon 2025 — Neural Forecasting Challenge
**Paper:** Li et al., "AMAG: Additive, Multiplicative and Adaptive Graph Neural Network For Forecasting Neural Activity", NeurIPS 2023

## Architecture

```
Input (B, 20, C, 9)
  → Feature Pathways (LMP + Spectral split embedding)
  → TE (2-layer, 4-head Transformer with pre-norm)
  → Session Embedding (additive, for cross-day adaptation)
  → SI (Add + Modulator with learnable adjacency)
  → Channel Attention (iTransformer-style cross-channel mixing)
  → TR (2-layer, 4-head Transformer with pre-norm)
  → Output (B, 20, C)
```

- **TE (Temporal Encoder):** Split LMP/spectral feature embedding → sinusoidal PE → 2 stacked multi-head self-attention blocks (4 heads, d=128, pre-norm, GELU)
- **SI (Spatial Interaction):** Additive message-passing with learnable adjacency + multiplicative (Hadamard) message-passing
- **Channel Attention:** iTransformer-style cross-channel self-attention per timestep — sample-dependent spatial mixing
- **TR (Temporal Readout):** Same structure as TE with separate parameters, projects to LMP predictions
- **RevIN:** Context-only instance normalization (stats from first 10 timesteps only)

### OOD / Domain Adaptation Methods

| Method | Citation | Purpose |
|--------|----------|---------|
| **RevIN** (context-only) | Kim et al., ICLR 2022 | Instance normalization using context window stats only |
| **MMD** | Gretton et al., JMLR 2012 | Multi-kernel Maximum Mean Discrepancy for domain alignment |
| **Spectral Loss** | — | FFT magnitude MSE auxiliary loss for frequency content |
| **Session Embeddings** | Ye et al., NeurIPS 2023 | Learnable per-session embeddings for cross-day adaptation |
| **Channel Attention** | Liu et al., ICLR 2024 | iTransformer-style sample-dependent spatial mixing |
| **Feature Pathways** | — | Separate LMP and spectral band embedding |
| **Freq Augmentation** | Xu et al., ICLR 2024 | Phase perturbation in frequency domain |
| **Huber Loss** | Huber, 1964 | Robust regression, less sensitive to outliers |
| **EMA** | Polyak & Juditsky, 1992 | Smoothed weight averaging for stable generalization |
| **Snapshot Ensemble** | Huang et al., ICLR 2017 | Averages predictions from 5 cosine cycle checkpoints |
| **Mixup** | Zhang et al., ICLR 2018 | Interpolates training samples for regularization |

## Setup

Requires Python >= 3.10 and a CUDA GPU.

```bash
# Install uv (if not installed)
pip install uv

# Install dependencies (PyTorch with CUDA 12.4)
uv sync
```

## Data

Place dataset files in the `dataset/` directory:

```
dataset/
├── train/
│   ├── train_data_beignet.npz
│   ├── train_data_beignet_2022-06-01_private.npz
│   ├── train_data_beignet_2022-06-02_private.npz
│   ├── train_data_affi.npz
│   └── train_data_affi_2024-03-20_private.npz
└── test/
    ├── test_data_beignet_masked.npz
    ├── test_data_beignet_2022-06-01_private_masked.npz
    ├── test_data_beignet_2022-06-02_private_masked.npz
    ├── test_data_affi_masked.npz
    └── test_data_affi_2024-03-20_private_masked.npz
```

Each `.npz` file contains `arr_0` with shape `(N, 20, C, 9)`:
- **N** = number of trials
- **20** = timesteps (10 context + 10 forecast, 30ms bins)
- **C** = channels (89 for beignet, 239 for affi)
- **9** = features (LMP + 8 frequency bands)

## Training

### Phase 1: Paper-Faithful Reproduction

Matches the original AMAG paper settings (Adam, StepLR, single-head single-layer d=64, adaptor enabled, no augmentation).

```bash
uv run python run_train.py beignet --phase paper
uv run python run_train.py affi --phase paper
```

### Phase 2: Competition (v3 — Full OOD Extensions)

Deep multi-head transformer, channel attention, feature pathways, MMD, spectral loss, Huber loss, session embeddings, frequency augmentation, snapshot ensemble.

```bash
# Train beignet (89 channels)
uv run python run_train.py beignet

# Train affi (239 channels)
uv run python run_train.py affi

# Override epochs
uv run python run_train.py beignet --epochs 200

# Multi-seed ensemble (trains 3 separate runs, checkpoints in checkpoints/seed_<N>/)
uv run python run_train.py beignet --seeds 42 123 456
```

### Options

```bash
uv run python run_train.py <monkey> --phase <paper|compete> [--epochs N] [--seeds S1 S2 ...]
```

| Argument | Values | Default |
|----------|--------|---------|
| `monkey` | `beignet`, `affi` | `beignet` |
| `--phase` | `paper`, `compete` | `compete` |
| `--epochs` | any integer | 500 (paper), 300 (compete) |
| `--seeds` | space-separated ints | `42` |

### Configuration Summary

| Parameter | Phase 1 (paper) | Phase 2 (compete) |
|-----------|-----------------|-------------------|
| Hidden dim | 64 | 128 |
| Attention heads | 1 | 4 |
| Transformer layers | 1 | 2 |
| d_ff | 256 | 512 |
| Dropout | 0.0 | 0.1 |
| Optimizer | Adam | AdamW |
| Scheduler | StepLR (x0.95/50ep) | CosineWarmRestarts (T0=60) |
| Weight decay | 1e-5 | 1e-4 |
| Epochs | 500 | 300 |
| Warmup | None | 10 epochs (linear) |
| Patience | 50 | 40 |
| Adaptor | Enabled | Disabled |
| Feature pathways | Off | On (LMP + spectral split) |
| Channel attention | Off | On (4-head cross-channel) |
| Session embeddings | Off | On |
| RevIN | Off | On (context-only stats) |
| MMD lambda | 0 | 0.05 |
| Spectral lambda | 0 | 0.1 |
| Loss | MSE | Huber (SmoothL1) |
| EMA | Off | On (decay=0.999) |
| Mixup | Off | On (alpha=0.3) |
| Augmentation | Off | Jitter + scale + channel dropout + freq phase |
| Snapshots | None | 5 (at cycle minima) |

## Evaluation

Evaluate trained models on the training data (since test data is masked):

```bash
uv run python -m src.amag.evaluate
```

This uses the submission model (snapshot ensemble + TTA normalization) and reports MSE on steps 10-19 for each dataset, split by same-day vs cross-date.

## Submission

After training, prepare the Codabench submission:

```bash
# Checkpoints are automatically copied to submission/ during evaluation
# Create the zip (files only, no directory structure)
cd submission
zip -j ../submission.zip model.py amag_*_snap*.pth amag_*_best.pth
```

Upload `submission.zip` to Codabench. The scoring script evaluates MSE on steps 10-19 across 5 test datasets (2 same-day + 3 cross-date), and the final score is their average.

### Submission Model Interface

The `submission/model.py` is self-contained (no imports from `src/`). It implements:

```python
class Model:
    def __init__(self, monkey_name):  # 'beignet' or 'affi'
    def load(self):                    # Auto-detects config from checkpoint, loads ensemble
    def predict(self, x):              # (N, 20, C, 9) → (N, 20, C)
```

Inference pipeline: TTA normalization (context-window stats) → snapshot ensemble (up to 5 models averaged) → denormalization.

## Project Structure

```
├── run_train.py                  # Training entry point (--phase paper|compete, --seeds)
├── pyproject.toml                # Dependencies (numpy, scipy, tqdm, torch)
├── src/amag/
│   ├── config.py                 # TrainConfig + phase1_config() + phase2_config()
│   ├── model.py                  # AMAG model (RevIN, channel attn, session embed)
│   ├── train.py                  # Training loop (EMA, mixup, MMD, spectral, warmup)
│   ├── data.py                   # Dataset loading, normalization, augmentation
│   ├── adjacency.py              # Correlation matrix initialization
│   ├── losses.py                 # MMD loss + spectral loss
│   ├── evaluate.py               # Local evaluation using submission model
│   └── modules/
│       ├── temporal_encoding.py  # TE: feature pathways + PE + multi-head transformer
│       ├── spatial_interaction.py # SI: Add + Modulator + adjacency
│       ├── temporal_readout.py   # TR: multi-head transformer + output FC
│       ├── channel_attention.py  # iTransformer-style cross-channel attention
│       └── revin.py              # RevIN with context-only normalization
├── submission/
│   └── model.py                  # Self-contained submission (no src/ imports)
├── CLAUDE.md                     # Paper reproduction guide
└── TASK.md                       # Challenge task description
```

## GPU Thermal Management

Designed for sustained training on laptop GPUs (tested on RTX 4090 Laptop):

- **AMP bfloat16** — reduced compute without loss scaling overhead
- **Inter-batch cooldown** — configurable sleep between batches (default 50ms)
- **Thermal pause** — auto-pauses at 82°C, resumes when cooled
- **Gradient checkpointing** — used in Adaptor MLP for memory efficiency
- **cudnn.benchmark** — autotuner for fixed-size inputs

Approximate VRAM usage (Phase 2):
- Beignet (89ch, batch=8): ~730 MB
- Affi (239ch, batch=4): ~1030 MB

## References

- Li et al., "AMAG: Additive, Multiplicative and Adaptive Graph Neural Network For Forecasting Neural Activity", NeurIPS 2023
- Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift", ICLR 2022
- Gretton et al., "A Kernel Two-Sample Test", JMLR 2012
- Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024
- Ye et al., "Neural Data Transformer 2", NeurIPS 2023
- Xu et al., "FITS: Modeling Time Series with 10k Parameters", ICLR 2024
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
- Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
- Huang et al., "Snapshot Ensembles: Train 1, Get M for Free", ICLR 2017
- Polyak & Juditsky, "Acceleration of Stochastic Approximation by Averaging", 1992
