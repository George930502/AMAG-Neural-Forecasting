# AMAG v3 — Detailed Implementation Documentation

**Branch:** `v3`
**Base paper:** Li et al., "AMAG: Additive, Multiplicative and Adaptive Graph Neural Network For Forecasting Neural Activity", NeurIPS 2023
**Competition:** NSF HDR Hackathon 2025 — Neural Forecasting Challenge

---

## Table of Contents

1. [Motivation and Root Cause Analysis](#1-motivation-and-root-cause-analysis)
2. [Architecture Changes](#2-architecture-changes)
   - 2.1 [Multi-Head Multi-Layer Transformer](#21-multi-head-multi-layer-transformer)
   - 2.2 [Feature-Specific Input Pathways](#22-feature-specific-input-pathways)
   - 2.3 [Channel-Mixing Attention](#23-channel-mixing-attention)
   - 2.4 [Session Embeddings](#24-session-embeddings)
   - 2.5 [Context-Only RevIN](#25-context-only-revin)
3. [Loss Functions](#3-loss-functions)
   - 3.1 [Huber Loss](#31-huber-loss)
   - 3.2 [Multi-Kernel MMD Loss](#32-multi-kernel-mmd-loss)
   - 3.3 [Spectral Loss](#33-spectral-loss)
4. [Data Augmentation](#4-data-augmentation)
   - 4.1 [Frequency-Domain Phase Perturbation](#41-frequency-domain-phase-perturbation)
   - 4.2 [Existing Augmentations](#42-existing-augmentations)
5. [Training Pipeline](#5-training-pipeline)
   - 5.1 [Warmup + Cosine Annealing Schedule](#51-warmup--cosine-annealing-schedule)
   - 5.2 [Snapshot Ensemble](#52-snapshot-ensemble)
   - 5.3 [EMA (Exponential Moving Average)](#53-ema-exponential-moving-average)
   - 5.4 [Mixup Regularization](#54-mixup-regularization)
6. [Inference Pipeline](#6-inference-pipeline)
   - 6.1 [TTA Normalization](#61-tta-normalization)
   - 6.2 [Snapshot Ensemble Averaging](#62-snapshot-ensemble-averaging)
   - 6.3 [Auto-Config Detection](#63-auto-config-detection)
7. [Full Forward Pass Walkthrough](#7-full-forward-pass-walkthrough)
8. [Hyperparameter Summary](#8-hyperparameter-summary)
9. [File-by-File Change Log](#9-file-by-file-change-log)
10. [Model Parameter Counts](#10-model-parameter-counts)

---

## 1. Motivation and Root Cause Analysis

The v2 baseline achieved a total MSR of 50,044 against the leader's 40,396. The gap was most severe on cross-day (OOD) datasets:

| Dataset | v2 Score | Leader | Gap |
|---|---|---|---|
| MSE_affi (same-day) | 51,989 | 40,001 | 30% |
| MSE_beignet (same-day) | 65,497 | 56,106 | 17% |
| MSE_affi_private (cross-day) | 45,046 | 32,113 | **40%** |
| MSE_beignet_D2 (cross-day) | 40,389 | 33,855 | 19% |
| MSE_beignet_D3 (cross-day) | 47,302 | 39,907 | 19% |

Root causes identified:
- **Weak OOD generalization** — 3 of 5 test sets are cross-day; v2 had static spatial interaction and weak normalization
- **Shallow architecture** — single transformer block, single-head attention, d=64 limited temporal modeling capacity
- **Fixed graph at inference** — Adaptor was disabled in compete mode, so spatial interaction was entirely static
- **Frequency features underused** — LMP and 8 power bands fed through a single linear layer
- **Weak normalization at test time** — RevIN computed stats over all 20 timesteps including masked future

---

## 2. Architecture Changes

### 2.1 Multi-Head Multi-Layer Transformer

**Files:** `src/amag/modules/temporal_encoding.py`, `src/amag/modules/temporal_readout.py`
**References:** Vaswani et al. 2017; Ye & Pandarinath 2021 (NDT uses 3-layer decoder)

**v2 (before):**
- 1 TransformerBlock, 1 attention head, d_model=64, d_ff=256
- Post-norm (LayerNorm after attention), ReLU activation

**v3 (after):**
- 2 stacked TransformerBlocks in both TE and TR (separate parameters)
- 4 attention heads, d_model=128, head_dim=32, d_ff=512
- **Pre-norm** architecture (LayerNorm before attention — more stable for deeper stacks)
- GELU activation instead of ReLU
- Output projection (`out_proj`) after multi-head concatenation
- Separate residual dropout on attention output and FFN output

**TransformerBlock forward pass (pre-norm):**
```
x_norm = LayerNorm(x)
Q, K, V = split_heads(W_q(x_norm), W_k(x_norm), W_v(x_norm))  # 4 heads, dim 32 each
attn = softmax(Q @ K^T / sqrt(32)) @ V                          # (B, 4, T, 32)
attn = concat_heads(attn)                                        # (B, T, 128)
attn = W_out(attn)                                               # output projection
x = x + Dropout(attn)                                            # residual 1

x = x + Dropout(FFN(LayerNorm(x)))                               # residual 2
  where FFN = Linear(128→512) → GELU → Dropout → Linear(512→128)
```

**Why pre-norm:** Post-norm places LayerNorm after the residual addition, which can cause gradient explosion in deeper stacks. Pre-norm normalizes before attention/FFN, keeping the residual path clean. This is the standard in GPT-2, LLaMA, and modern transformers.

### 2.2 Feature-Specific Input Pathways

**File:** `src/amag/modules/temporal_encoding.py` (TemporalEncoder)
**Reference:** STNDT (Le & Shlizerman, 2022) separates spatial/temporal processing

**Problem:** The input has 9 features per channel: 1 LMP signal + 8 frequency power bands (0.5–4Hz, 4–8Hz, ..., 200–400Hz). LMP has near-zero correlation with power bands. A single `Linear(9, 128)` forces the model to learn this separation implicitly.

**Solution:** Split embedding into two pathways:
```python
# LMP pathway: 1 → 64
embed_lmp = Linear(1, 64)

# Spectral pathway: 8 → 64
embed_spec = Linear(8, 64)

# Fuse: 128 → 128
h = embed_fuse(cat([embed_lmp(x[:,:,:1]), embed_spec(x[:,:,1:])], dim=-1))
```

This allows the model to learn independent representations for LMP amplitude and spectral power content, then fuse them with a learned linear combination.

**Config flag:** `use_feature_pathways: bool` (True for compete, False for paper)

### 2.3 Channel-Mixing Attention

**File:** `src/amag/modules/channel_attention.py` (NEW)
**Reference:** Liu et al., "iTransformer", ICLR 2024 Spotlight

**Problem:** v2 disabled the Adaptor MLP in compete mode (it overfits on small cross-day sets), leaving spatial interaction entirely static via learned adjacency matrices `A_a` and `A_m`. This means every test sample gets the same spatial mixing regardless of its content.

**Solution:** Add a lightweight cross-channel attention layer between SI and TR. For each timestep, channels become tokens and undergo multi-head self-attention:

```
Input:  z: (B, T, C, D)
Reshape: (B*T, C, D)        — each timestep independently
Pre-norm: LayerNorm(z)
Q, K, V: 4-head attention across C channels (C tokens, D=128 dim)
Output: z + Dropout(W_out(MultiHead(Q, K, V)))
Reshape back: (B, T, C, D)
```

**Why this works:** Unlike the Adaptor MLP which requires O(C^2 * T * D) pairwise computations, channel attention is O(C^2 * D) per timestep — much cheaper. It's also sample-dependent: each trial gets different attention weights based on its channel activation patterns.

**Placement in pipeline:** After SI (which provides fixed adjacency-weighted mixing), before TR. This gives the model both stable learned spatial structure (SI) and adaptive per-sample spatial refinement (channel attention).

### 2.4 Session Embeddings

**File:** `src/amag/model.py`
**Reference:** NDT2 (Ye et al., NeurIPS 2023); POYO (Azabou et al., NeurIPS 2023)

**Problem:** Cross-day recordings have different baseline neural activity patterns. Without session-specific information, the model can't distinguish which recording day a trial comes from.

**Solution:** Learnable embedding per session, added to TE output:
```python
session_embed = nn.Embedding(num_sessions, hidden_dim)  # (3, 128) for beignet

# During forward:
h = self.te(x)                              # (B, T, C, 128)
se = self.session_embed(session_ids)        # (B, 128)
h = h + se[:, None, None, :]               # broadcast to (B, T, C, 128)
```

- **Training:** Each sample has a session_id (0=primary, 1=cross-day-1, 2=cross-day-2). The embedding is added after TE output.
- **Inference for known sessions:** Use the corresponding learned embedding.
- **Inference for unknown sessions:** The submission model passes `session_ids=None`, so no session embedding is added. The model must generalize via other OOD mechanisms (RevIN, TTA normalization).

### 2.5 Context-Only RevIN

**File:** `src/amag/modules/revin.py`
**Reference:** Kim et al., "Reversible Instance Normalization", ICLR 2022

**Problem (v2):** RevIN computed instance statistics over all 20 timesteps: `mean = x.mean(dim=1)`. But timesteps 10-19 are masked (copied from step 9), so the statistics are contaminated by repeated values and don't represent the true distribution of the forecast window.

**Fix (v3):** Compute statistics from context window only (steps 0-9):
```python
def normalize(self, x):      # x: (B, T, C)
    context = x[:, :10]      # only use first 10 timesteps
    self._mean = context.mean(dim=1, keepdim=True)   # (B, 1, C)
    self._std = (context.var(dim=1, keepdim=True) + eps).sqrt()
    return (x - self._mean) / self._std * weight + bias
```

The same context-derived statistics are stored and used for denormalization of the full output sequence. This prevents future information leakage and gives more accurate per-sample normalization for cross-day data.

---

## 3. Loss Functions

### 3.1 Huber Loss

**File:** `src/amag/train.py`
**Reference:** Huber (1964)

Replaces MSE with Smooth L1 (Huber) loss:
```python
criterion = nn.SmoothL1Loss(beta=1.0)
```

**Why:** Neural signals have occasional extreme values (artifacts, high-amplitude transients). MSE squares these, giving them disproportionate influence on gradients. Huber loss transitions from quadratic (near 0) to linear (far from 0), reducing sensitivity to outliers while maintaining smooth gradients near the optimum.

**Config:** `loss_type: str = "huber"` (v3 default) or `"mse"` (paper default)

### 3.2 Multi-Kernel MMD Loss

**File:** `src/amag/losses.py`
**Reference:** Gretton et al., "A Kernel Two-Sample Test", JMLR 2012

**Replaces CORAL** (v2 used CORAL which only matches 2nd-order statistics). MMD matches all moments of the distribution via kernel embedding.

**Algorithm:**
1. Compute TE hidden states: `h = model.get_te_hidden(x)` → `(B, T, C, D)`
2. Mean-pool to `(B, D)`: `h_pooled = h.mean(dim=(1,2))`
3. Split by session: `source = h_pooled[session==0]`, `target = h_pooled[session>0]`
4. Compute MMD with **median-heuristic bandwidth selection**:
   - Compute all pairwise squared distances
   - Set `median_dist = median(dist[dist > 0])`
   - Use 5 Gaussian kernels at bandwidths: `[0.2, 0.5, 1.0, 2.0, 5.0] * median_dist`
   - `MMD = mean(K_ss) + mean(K_tt) - 2*mean(K_st)`

**Why median heuristic:** Fixed bandwidths (v2's approach with CORAL) don't adapt to the scale of hidden representations. The median heuristic automatically selects bandwidths that are appropriate for the actual feature magnitudes, preventing numerical overflow/underflow.

**Config:** `mmd_lambda: float = 0.05`

### 3.3 Spectral Loss

**File:** `src/amag/losses.py`

Auxiliary loss that operates in the frequency domain:
```python
def spectral_loss(pred, target):        # (B, T, C)
    pred_fft = torch.fft.rfft(pred, dim=1)
    target_fft = torch.fft.rfft(target, dim=1)
    return F.mse_loss(pred_fft.abs(), target_fft.abs())
```

**Why:** Neural networks tend to learn low-frequency components first (spectral bias). An explicit FFT magnitude loss encourages the model to match the full frequency spectrum of the target signal, including higher-frequency oscillatory components that are important in neural data.

**Config:** `spectral_lambda: float = 0.1`

**Total loss:**
```
L = Huber(pred, target) + 0.05 * MMD(h_source, h_target) + 0.1 * Spectral(pred, target)
```

---

## 4. Data Augmentation

### 4.1 Frequency-Domain Phase Perturbation

**File:** `src/amag/data.py`
**Reference:** Xu et al., "FITS", ICLR 2024

```python
def _apply_freq_augmentation(self, x):   # x: (T, C, F)
    X_fft = np.fft.rfft(x, axis=0)       # FFT along time
    phase_noise = np.random.uniform(-0.1*pi, 0.1*pi, X_fft.shape)
    X_fft = X_fft * np.exp(1j * phase_noise)
    return np.fft.irfft(X_fft, n=T, axis=0)
```

**Why:** Standard Gaussian jitter adds noise uniformly across frequencies. Phase perturbation preserves the **exact power spectrum** (magnitude) while varying the phase relationships. This creates augmented samples that have the same spectral content as real data but with different temporal patterns — more realistic diversity for OOD generalization.

**Phase range:** ±0.1π (±18°) — small enough to preserve temporal structure, large enough to create meaningful diversity.

### 4.2 Existing Augmentations (unchanged from v2)

| Augmentation | Parameter | Effect |
|---|---|---|
| Jittering | `aug_jitter_std=0.02` | Additive Gaussian noise on all features |
| Scaling | `aug_scale_std=0.1` | Per-channel multiplicative noise (simulates impedance drift) |
| Channel dropout | `aug_channel_drop_p=0.1` | Randomly zero out 10% of channels |
| Mixup | `mixup_alpha=0.3` | Beta-distributed interpolation between training pairs |

All augmentations are applied on-the-fly during training only (not during validation or inference).

---

## 5. Training Pipeline

### 5.1 Warmup + Cosine Annealing Schedule

**File:** `src/amag/train.py`

```
Epochs 1-10:   Linear warmup from 0.01*lr to lr
Epochs 11-300: CosineAnnealingWarmRestarts (T_0=60, T_mult=1)
                → 5 cosine cycles of 60 epochs each
```

**Why warmup:** Large learning rates at initialization (when weights are random) cause unstable gradients, especially with pre-norm transformers and multi-head attention. Linear warmup from 1% of target LR allows the model to find a reasonable loss basin before applying the full learning rate.

**Implementation:**
```python
if warmup_epochs > 0 and epoch <= warmup_epochs:
    warmup_scheduler.step()    # LinearLR: 0.01*lr → lr over 10 epochs
else:
    scheduler.step()           # CosineAnnealingWarmRestarts
```

### 5.2 Snapshot Ensemble

**Reference:** Huang et al., "Snapshot Ensembles: Train 1, Get M for Free", ICLR 2017

At each cosine cycle minimum (epochs 60, 120, 180, 240, 300), the model checkpoint is saved as a snapshot. At inference, predictions from all snapshots are averaged.

- **v2:** 3 snapshots (cycles of 50 epochs, 150 total)
- **v3:** 5 snapshots (cycles of 60 epochs, 300 total)

Snapshots capture diverse solutions from different regions of the loss landscape. The cosine cycle minimum is chosen because the model has the lowest learning rate (and thus most refined weights) at that point.

If EMA is active and past the start epoch, the EMA shadow model is saved instead of the raw model.

### 5.3 EMA (Exponential Moving Average)

**Reference:** Polyak & Juditsky, 1992

```python
shadow_param = decay * shadow_param + (1 - decay) * model_param
```

- `ema_decay = 0.999` — shadow weights update slowly, averaging ~1000 recent optimizer steps
- `ema_start_epoch = 10` — begins after warmup completes (no point averaging random early weights)
- Updated once per optimizer step (not per micro-batch in gradient accumulation)

### 5.4 Mixup Regularization

**Reference:** Zhang et al., ICLR 2018

```python
lam = Beta(0.3, 0.3).sample()
x_mixed = lam * x1 + (1 - lam) * x2
target_mixed = lam * target1 + (1 - lam) * target2
```

A separate DataLoader provides random second batches for mixing. Both the input features and the targets are interpolated with the same lambda. This acts as strong regularization, especially for small datasets.

---

## 6. Inference Pipeline

**File:** `submission/model.py`

### 6.1 TTA Normalization

Test-Time Adaptation computes normalization statistics from the context window of the test data itself (not training statistics):

```python
context = x[:, :10]                                    # (N, 10, C, 9)
flat = context.reshape(N*10, C*9)
mean, std = flat.mean(axis=0), flat.std(axis=0)        # per-channel-feature
x_norm = 2 * (x - (mean - 4*std)) / (8*std) - 1       # normalize to [-1, 1]
```

This handles cross-day distribution shifts at test time without any model parameter updates.

### 6.2 Snapshot Ensemble Averaging

```python
all_preds = []
for model in self.models:                              # up to 5 snapshot models
    pred = model(x_norm)                               # (N, 20, C)
    all_preds.append(pred)
pred_final = mean(all_preds, axis=0)                   # average across snapshots
pred_raw = denormalize(pred_final, mean, std)           # back to raw scale
```

### 6.3 Auto-Config Detection

The submission model automatically detects the architecture configuration from the checkpoint state dict:

| Detected from | Config field |
|---|---|
| `"revin."` key prefix exists | `use_revin` |
| `"channel_attn."` key prefix exists | `use_channel_attn` |
| `"session_embed."` key prefix exists | `use_session_embed` |
| `"te.embed_lmp.weight"` exists | `use_feature_pathways` |
| `te.embed_fuse.weight.shape[0]` or `te.embed.weight.shape[0]` | `hidden_dim` |
| max layer index in `"te.layers.{i}."` keys | `num_layers` |
| `"te.layers.0.ffn.0.weight"` shape | `d_ff` |
| `"session_embed.weight"` shape | `num_sessions` |

This means the submission model can load checkpoints from both v2 (single-head, d=64) and v3 (multi-head, d=128) without manual configuration.

---

## 7. Full Forward Pass Walkthrough

Input: `x: (B, 20, C, 9)` where C=89 (beignet) or C=239 (affi)

```
1. RevIN normalize (context-only):
   lmp = x[:,:,:,0]                           # (B, 20, C)
   mean, std = stats from lmp[:,:10,:]        # context window only
   lmp_norm = (lmp - mean) / std * weight + bias
   x[:,:,:,0] = lmp_norm

2. Temporal Encoder (TE):
   Reshape: (B*C, 20, 9)
   Feature pathways:
     lmp_embed = Linear(1→64)(x[:,:,:1])      # (B*C, 20, 64)
     spec_embed = Linear(8→64)(x[:,:,1:])     # (B*C, 20, 64)
     h = Linear(128→128)(cat(lmp_embed, spec_embed))
   Positional encoding: h += PE               # sinusoidal
   Transformer layer 1: h = Block(h)          # 4-head, pre-norm
   Transformer layer 2: h = Block(h)          # 4-head, pre-norm
   Reshape: (B, 20, C, 128)

3. Session Embedding (training only):
   h = h + session_embed[session_id]          # (B, 1, 1, 128) broadcast

4. Spatial Interaction (SI):
   Add:  a = A_a @ h                          # (C,C) learnable adjacency
   a = FC(a)                                  # Linear(128→128)
   Mod:  m = (A_m @ h) ⊙ h                   # Hadamard product
   m = FC(m)                                  # Linear(128→128)
   z = β₁*h + β₂*a + β₃*m                    # learnable scalars

5. Channel Attention:
   Reshape: (B*20, C, 128)                    # channels as tokens
   Pre-norm + 4-head self-attention across C channels
   z = z + Dropout(W_out(Attn(Q,K,V)))        # residual
   Reshape: (B, 20, C, 128)

6. Temporal Readout (TR):
   Reshape: (B*C, 20, 128)
   Positional encoding: r += PE
   Transformer layer 1: r = Block(r)          # 4-head, pre-norm
   Transformer layer 2: r = Block(r)          # 4-head, pre-norm
   Output: pred = Linear(128→1)(r)            # (B*C, 20, 1)
   Reshape: (B, 20, C)

7. RevIN denormalize:
   pred = (pred - bias) / weight * std + mean

Output: pred: (B, 20, C) — LMP predictions for all 20 timesteps
Loss computed on steps 10-19 only (forecast window)
```

---

## 8. Hyperparameter Summary

### Phase 2 (Compete) — v3 Defaults

| Category | Parameter | Value |
|---|---|---|
| **Architecture** | hidden_dim | 128 |
| | d_ff | 512 |
| | num_heads | 4 |
| | num_layers | 2 |
| | dropout | 0.1 |
| | use_channel_attn | True |
| | use_feature_pathways | True |
| | use_session_embed | True |
| | use_revin | True (context-only) |
| **Optimizer** | type | AdamW |
| | lr | 5e-4 |
| | weight_decay | 1e-4 |
| | warmup_epochs | 10 (linear, 1%→100%) |
| **Scheduler** | type | CosineAnnealingWarmRestarts |
| | T_0 (cycle length) | 60 epochs |
| | T_mult | 1 (constant cycle length) |
| **Training** | epochs | 300 |
| | batch_size | 32 (beignet) / 16 (affi) |
| | patience | 40 |
| | val_every | 5 epochs |
| | grad_clip_norm | 5.0 |
| **Losses** | primary | Huber (SmoothL1, beta=1.0) |
| | mmd_lambda | 0.05 |
| | spectral_lambda | 0.1 |
| **Regularization** | EMA decay | 0.999 (from epoch 10) |
| | mixup_alpha | 0.3 |
| | aug_jitter_std | 0.02 |
| | aug_scale_std | 0.1 |
| | aug_channel_drop_p | 0.1 |
| | freq_augment | True |
| **Ensemble** | num_snapshots | 5 |
| | snapshot_cycle_len | 60 |

### Phase 1 (Paper) — Unchanged

| Parameter | Value |
|---|---|
| hidden_dim | 64 |
| d_ff | 256 |
| num_heads | 1 |
| num_layers | 1 |
| dropout | 0.0 |
| optimizer | Adam |
| lr | 5e-4, StepLR ×0.95/50ep |
| epochs | 500 |
| All v3 features | Disabled |

---

## 9. File-by-File Change Log

| File | Lines Changed | Summary |
|---|---|---|
| `src/amag/config.py` | +72 | Added `d_ff`, `num_heads`, `num_layers`, `use_channel_attn`, `use_feature_pathways`, `mmd_lambda`, `spectral_lambda`, `loss_type`, `use_session_embed`, `num_sessions`, `warmup_epochs`. Updated defaults. Removed `cooldown_ms`. |
| `src/amag/modules/temporal_encoding.py` | +80 | Rewrote `TransformerBlock` with multi-head attention, pre-norm, GELU, output projection. `TemporalEncoder` now has `nn.ModuleList` of stacked layers and optional feature pathways. |
| `src/amag/modules/temporal_readout.py` | +12 | Added `num_heads`, `num_layers` params. Uses `nn.ModuleList` of stacked `TransformerBlock`s. |
| `src/amag/modules/channel_attention.py` | **NEW (70 lines)** | iTransformer-style cross-channel multi-head self-attention with pre-norm and residual connection. |
| `src/amag/modules/revin.py` | +16 | Changed `normalize()` to compute mean/std from context window only (`x[:, :context_len]`). Added `context_len` parameter. |
| `src/amag/model.py` | +52 | Integrated ChannelAttention, session embeddings, feature pathways. Forward pass accepts `session_ids`. New `get_te_hidden()` also supports session embeddings. |
| `src/amag/losses.py` | +88 (rewrite) | Replaced CORAL with multi-kernel MMD using median-heuristic bandwidths. Added spectral loss (FFT magnitude MSE). Removed `coral_loss` and `compute_coral_from_hidden`. |
| `src/amag/train.py` | +99 | New loss computation (Huber + MMD + spectral), linear warmup scheduler, session IDs passed to model, freq augmentation enabled, multi-worker DataLoaders. Removed all thermal management code. |
| `src/amag/data.py` | +27 | Added `_apply_freq_augmentation()` method (FFT phase perturbation). Added `freq_augment` flag to Dataset and `prepare_datasets()`. |
| `submission/model.py` | +263 (rewrite) | Full mirror of all architecture changes. Auto-detects config from checkpoint state dict. Supports up to 5 snapshot ensemble. Context-only RevIN. |
| `run_train.py` | +21 | Added `--seeds` flag for multi-seed ensemble training. Updated batch sizes (32/16). Removed cooldown. |

---

## 10. Model Parameter Counts

### Beignet (89 channels, v3 compete config)

| Module | Parameters |
|---|---|
| TE (embed_lmp + embed_spec + embed_fuse) | 1×64 + 8×64 + 128×128 = 17,024 |
| TE (PE) | 0 (buffer) |
| TE (2× TransformerBlock) | 2 × (4×128×128 + 128×128 + 128×512 + 512×128 + biases + LN) ≈ 2 × 198,400 = 396,800 |
| SI (A_a + A_m) | 2 × 89×89 = 15,842 |
| SI (fc_add + fc_mod + betas) | 2 × 128×128 + 3 = 32,771 |
| Channel Attention | 4×128×128 + 128×128 + LN ≈ 82,432 |
| TR (PE + 2× TransformerBlock + output_fc) | 396,800 + 128×1 = 396,929 |
| RevIN (affine) | 2 × 89 = 178 |
| Session Embedding | 3 × 128 = 384 |
| **Total** | **~926,000** |

### Affi (239 channels, v3 compete config)

Same as above except:
- SI adjacency: 2 × 239×239 = 114,242 (vs 15,842)
- RevIN: 2 × 239 = 478
- Session Embedding: 2 × 128 = 256 (affi has 2 sessions)
- **Total: ~1,025,000**

### Comparison with v2

| | v2 (d=64, 1 layer) | v3 (d=128, 2 layers) |
|---|---|---|
| Beignet | ~100,000 | ~926,000 |
| Affi | ~200,000 | ~1,025,000 |
| Factor | — | ~5-9× larger |

---

## References

1. Li et al., "AMAG: Additive, Multiplicative and Adaptive Graph Neural Network For Forecasting Neural Activity", NeurIPS 2023
2. Vaswani et al., "Attention Is All You Need", NeurIPS 2017
3. Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift", ICLR 2022
4. Gretton et al., "A Kernel Two-Sample Test", JMLR 2012
5. Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024
6. Ye et al., "Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity", NeurIPS 2023
7. Xu et al., "FITS: Modeling Time Series with 10k Parameters", ICLR 2024
8. Huber, "Robust Estimation of a Location Parameter", Annals of Mathematical Statistics, 1964
9. Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
10. Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
11. Huang et al., "Snapshot Ensembles: Train 1, Get M for Free", ICLR 2017
12. Polyak & Juditsky, "Acceleration of Stochastic Approximation by Averaging", SIAM Journal on Control and Optimization, 1992
