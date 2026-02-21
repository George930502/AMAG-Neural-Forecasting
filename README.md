# AMAG Neural Forecasting

Forecasting neural activity from micro-electrocorticography (µECoG) recordings using **AMAG** (Additive, Multiplicative and Adaptive Graph Neural Network) with **RevIN** and **CORAL** for cross-session generalization.

**Competition:** NSF HDR Hackathon 2025 — Neural Forecasting Challenge
**Paper:** Li et al., "AMAG: Additive, Multiplicative and Adaptive Graph Neural Network For Forecasting Neural Activity", NeurIPS 2023

## Architecture

```
Input (B, 20, C, 9) → TE (Transformer) → SI (Add + Modulator) → TR (Transformer) → Output (B, 20, C)
```

- **TE (Temporal Encoder):** FC embedding + sinusoidal positional encoding + single-head self-attention
- **SI (Spatial Interaction):** Additive message-passing with learnable adjacency + multiplicative (Hadamard) message-passing + optional Adaptor MLP gating
- **TR (Temporal Readout):** Same structure as TE with separate parameters, projects to LMP predictions

### OOD Extensions

| Method | Citation | Purpose |
|--------|----------|---------|
| **RevIN** | Kim et al., ICLR 2022 | Instance normalization to handle distribution shift across sessions |
| **CORAL** | Sun & Saenko, ECCV 2016 | Aligns covariance of hidden features between sessions |
| **EMA** | Polyak & Juditsky, 1992 | Smoothed weight averaging for stable generalization |
| **Snapshot Ensemble** | Huang et al., ICLR 2017 | Averages predictions from 3 cosine cycle checkpoints |
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

Matches the original AMAG paper settings (Adam, StepLR, adaptor enabled, no augmentation).

```bash
uv run python run_train.py beignet --phase paper
uv run python run_train.py affi --phase paper
```

### Phase 2: Competition (OOD Extensions)

Adds RevIN, CORAL, EMA, mixup, snapshot ensemble, and augmentation for cross-date generalization.

```bash
uv run python run_train.py beignet --phase compete
uv run python run_train.py affi --phase compete
```

### Options

```bash
uv run python run_train.py <monkey> --phase <paper|compete> [--epochs N]
```

| Argument | Values | Default |
|----------|--------|---------|
| `monkey` | `beignet`, `affi` | `beignet` |
| `--phase` | `paper`, `compete` | `compete` |
| `--epochs` | any integer | 500 (paper), 150 (compete) |

### Configuration Summary

| Parameter | Phase 1 (paper) | Phase 2 (compete) |
|-----------|-----------------|-------------------|
| Optimizer | Adam | AdamW |
| Scheduler | StepLR (x0.95/50ep) | CosineWarmRestarts (T0=50) |
| Weight decay | 1e-5 | 1e-4 |
| Epochs | 500 | 150 |
| Adaptor | Enabled | Disabled |
| RevIN | Off | On |
| CORAL lambda | 0 | 0.1 |
| EMA | Off | On (decay=0.999) |
| Mixup | Off | On (alpha=0.3) |
| Augmentation | Off | Jitter + scale + channel dropout |
| Snapshots | None | 3 (at cycle minima) |

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
    def load(self):                    # Load snapshot checkpoints
    def predict(self, x):              # (N, 20, C, 9) → (N, 20, C)
```

Inference pipeline: TTA normalization (stats from context window) → snapshot ensemble (3 models averaged) → denormalization.

## Project Structure

```
├── run_train.py                  # Training entry point (--phase paper|compete)
├── pyproject.toml                # Dependencies (numpy, scipy, tqdm, torch)
├── src/amag/
│   ├── config.py                 # TrainConfig + phase1_config() + phase2_config()
│   ├── model.py                  # AMAG model with optional RevIN
│   ├── train.py                  # Training loop (EMA, mixup, CORAL, thermal mgmt)
│   ├── data.py                   # Dataset loading, normalization, augmentation
│   ├── adjacency.py              # Correlation matrix initialization
│   ├── losses.py                 # CORAL loss (Sun & Saenko, ECCV 2016)
│   ├── evaluate.py               # Local evaluation using submission model
│   └── modules/
│       ├── temporal_encoding.py  # TE: FC + positional encoding + transformer
│       ├── spatial_interaction.py # SI: Add + Modulator + Adaptor + adjacency
│       ├── temporal_readout.py   # TR: transformer + output FC
│       └── revin.py              # RevIN (Kim et al., ICLR 2022)
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

## References

- Li et al., "AMAG: Additive, Multiplicative and Adaptive Graph Neural Network For Forecasting Neural Activity", NeurIPS 2023
- Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift", ICLR 2022
- Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV 2016
- Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
- Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
- Huang et al., "Snapshot Ensembles: Train 1, Get M for Free", ICLR 2017
