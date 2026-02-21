# AMAG: Reproduction Guide — Monkey Affogato (A) & Beignet (B)
**Paper:** "AMAG: Additive, Multiplicative and Adaptive Graph Neural Network For Forecasting Neural Activity"  
**Venue:** NeurIPS 2023  
**Authors:** Jingyuan Li, Leo Scholl, Trung Le, Pavithra Rajeswaran, Amy Orsborn, Eli Shlizerman  
**Code:** https://github.com/shlizee/AMAG

---

## 1. Problem Formulation

**Task:** Forecast future neural activity from past observations.

**Notation:**
- Input: $X \in \mathbb{R}^{T \times C \times D}$ (T timesteps, C channels, D feature dims)
- Goal: $\hat{X}_{t+1:t+\tau} = f_\theta(X_{0:t})$
- Loss: $\min_{f_\theta} \mathbb{E}_X[\mathcal{L}(f_\theta(X_{0:t}), X_{t+1:t+\tau})]$
- Training loss: MSE — $\mathcal{L} = \mathbb{E}_X[\|\hat{X}_{t:T} - X_{t:T}\|^2]$

---

## 2. Model Architecture (AMAG)

Three sequential modules: **TE → SI → TR**

### 2.1 Spatial Interaction (SI) Module

Graph $G = (V, E, A)$ where $|V| = C$.

**Add Module (additive message-passing):**
$$a_t^{(v)} = \sum_{u \in N_a(v)} S^{(u,v)} A_a^{(u,v)} h_t^{(u)}$$

- $A_a \in \mathbb{R}^{C \times C}$: learnable adjacency (initialized from correlation matrix)
- $S^{(u,v)} = \sigma(\text{MLP}([H^{(u)}, H^{(v)}]))$: sample-dependent Adaptor, range [0,1]
- MLP: 4 FC layers, input dims $64\times t$, $64\times 2$, $64\times 4$, $64$

**Modulator Module (multiplicative message-passing):**
$$m_t^{(v)} = \sum_{u \in N_m(v)} A_m^{(u,v)} h_t^{(u)} \odot h_t^{(v)}$$

- $A_m \in \mathbb{R}^{C \times C}$: learnable adjacency (also initialized from correlation matrix)
- $\odot$: Hadamard product

**SI Output:**
$$z_t^{(v)} = \beta_1 h_t^{(v)} + \beta_2 FC(a_t^{(v)}) + \beta_3 FC(m_t^{(v)})$$

where $\beta_1, \beta_2, \beta_3$ control self-connection, Add, and Modulator contributions. The paper treats these as scalar coefficients (learnable parameters). The $FC$ layers here are single-layer fully-connected networks mapping from hidden dim $d$ to $d$. TE and TR GRUs use **separate** weight parameters (not shared between TE and TR).

### 2.2 Adjacency Matrix Initialization

$$\text{Corr}(u,v) = \frac{X^{(u)} X^{(v)T}}{\sqrt{|X^{(u)}||X^{(v)}|}}$$

Both $A_a^{(u,v)}$ and $A_m^{(u,v)}$ initialized as $\text{Corr}(u,v)$.  
Flatten features: $X^{(u)}, X^{(v)} \in \mathbb{R}^{TD}$ before computing correlation.

### 2.3 TE and TR — One-step Forecasting (GRU-based)

```
h_t^(v) = GRU_TE(X_t^(v), h_{t-1}^(v)),   h_0 = 0
z_t^(v) = SI(h_t^(v), {h_t^(u) | u ∈ N(v)}, Aa, Am)
r_t^(v) = GRU_TR(z_t^(v), r_{t-1}^(v)),    r_0 = 0
X_{t+1}^(v) = FC(r_t^(v))
```

### 2.4 TE and TR — Multi-step Forecasting (Transformer-based)

**Embedding:**
$$E = FC(\text{mask}(X)) + PE$$

- `mask`: zeros out future inputs after context window $t$
- $PE(\text{pos}, 2i) = \sin(\text{pos}/10000^{2i/d})$
- $PE(\text{pos}, 2i+1) = \sin(\text{pos}/10000^{(2i+1)/d})$ *(Note: the paper writes `sin` for both 2i and 2i+1; standard Vaswani 2017 uses `cos` for odd indices. Follow standard implementation: `cos` for 2i+1.)*

**Temporal attention (single-head):**
$$ATT^{(v)} = \text{Softmax}\left(\frac{Q_e^{(v)}(K_e^{(v)})^T}{\sqrt{d}}\right) V_e^{(v)}$$
$$H^{(v)} = LN(MLP(E^{(v)} + ATT^{(v)}))$$

Then SI applied per timestep. TR identical structure to TE but takes $Z^{(v)}$ as input and outputs $\hat{X}^{(v)}$.  
Note: $W_k, W_q, W_v$ are **separate** parameter sets in TE and TR. Parameters shared **across channels**.

---

## 3. Datasets — Monkey A & B Only

### 3.1 Recording Setup
- **Recording type:** µECoG (surface-level micro-electrocorticography)
- **Implant location:** Motor cortices
- **Regions covered:** Primary Motor Cortex (M1), Premotor Cortex (PM), Frontal Eye Field (FEF), Supplementary Motor Area (SMA), Dorsolateral Prefrontal Cortex (DLPFC)
- **Total effective electrodes:** 239 µECoG electrode arrays (previously used in Trumpis et al., 2021)

### 3.2 Monkey Affogato (A)
- **Channels used:** All 239 electrodes (all 5 regions)
- **Sessions:** 13 sessions (combined into one dataset)
- **Train / Val / Test split:** 985 / 122 / 122 trials (80/10/10)

### 3.3 Monkey Beignet (B)
- **Channels used:** 87 electrodes from **M1 region only**
- **Sessions:** 13 sessions (combined into one dataset)
- **Train / Val / Test split:** 700 / 87 / 87 trials (80/10/10)

### 3.4 Behavioral Task
- **Task:** Center-out reaching toward one of **8 directions**
- **Tracking:** Marker-based camera system (OptiTrack) maps hand position to cursor
- **Trial structure:**
  1. Subject reaches center target
  2. Peripheral (surrounding) target appears
  3. Delay period: ~50–150ms
  4. Subject moves from center to peripheral target
  5. Successful reach → reward

### 3.5 Trial Segmentation & Alignment
- **Alignment point:** Appearance of peripheral target
- **Sample window:** 600ms after peripheral target onset
  - First **150ms** = preparatory context (input to model)
  - Next **450ms** = forecast target (model output)
- **Exclusion criterion:** Trials shorter than 600ms after peripheral target onset are excluded
- **Timestep arithmetic:** 1 bin = 30ms → total T = 600ms / 30ms = **20 timesteps** per trial; context t = 5 steps; forecast τ = 15 steps

### 3.6 Raw Data Acquisition
- **Sampling rate:** 25 kHz (full broadband)

### 3.7 Preprocessing Steps
1. Extract **LMP** (Local Motor Potential) from broadband signal
2. Compute power in **8 frequency bands:**
   - 0.5–4 Hz, 4–8 Hz, 8–12 Hz, 12–25 Hz, 25–50 Hz, 50–100 Hz, 100–200 Hz, 200–400 Hz
3. Downsample to **30ms bins**
4. Stack LMP + 8 power bands → **D = 9 features per channel**
5. **Normalize** per channel using training set statistics:
   - Compute mean ($\mu$) and std ($\sigma$) per channel on training set only
   - Linearly scale values in $[\mu - 4\sigma, \mu + 4\sigma]$ to $[-1, 1]$

### 3.8 Data Availability
- Will be shared via project page: https://github.com/shlizee/AMAG
- Includes: neural time series for all train/test samples + metadata of relative electrode locations within array
- Ethics: All procedures approved by University of Washington Institutional Animal Care and Use Committee

---

## 4. Experiment Setup (Monkey A & B)

| Setting | Value |
|---|---|
| Context window | ≥5 steps (150ms) |
| One-step TE/TR | GRU |
| Multi-step TE/TR | Transformer |
| Optimizer | Adam |
| LR (GNN methods incl. AMAG) | 5e-4 |
| LR (non-GNN methods) | 1e-4 |
| LR decay | ×0.95 every 50 epochs |
| Training epochs | 500 (most); 1000 for AMAG one-step and STNDT |
| Validation frequency | Every 10 epochs |
| GPU | Titan X |
| AMAG hidden size | 64 |
| Runs per result | 3 (std reported) |
| Evaluation metrics | R², Pearson Correlation, MSE |

**Metric definitions:**
- **R²:** Standard coefficient of determination: $R^2 = 1 - \frac{\sum(X - \hat{X})^2}{\sum(X - \bar{X})^2}$, computed over all channels and timepoints in the forecast window, then reported as mean ± std across 3 runs.
- **Correlation (Corr):** Pearson correlation between ground truth and predicted signals (per-channel, then averaged across channels).
- **MSE:** Mean squared error $\frac{1}{N}\sum(X - \hat{X})^2$ over all channels and forecast timesteps.

---

## 5. AMAG Hyperparameters (Monkey A & B)

| Setting | Value |
|---|---|
| Hidden size (GRU / Transformer) | 64 |
| Weight decay | 1e-5 |
| Learning rate | 5e-4 decaying ×0.95 every 50 epochs |
| Adjacency matrix init | Correlation matrix |
| Adaptor MLP layers | 4 FC layers: dims $64\times t$, $64\times 2$, $64\times 4$, $64$ |

---

## 6. Baseline Hyperparameters (Monkey A & B)

| Model | Monkey A & B Specific Config |
|---|---|
| **LFADS** | Uni-directional GRU (constrained from default bi-directional). Factor dim=256; inferred input dim=256; enc/ctrl/gen dim=512; LR=1e-4 ×0.95/50ep; WD=1e-4 |
| **LRNN** | 1-layer linear RNN (no nonlinearity, no gating). 1-step: hidden=1024, no WD; Multi: hidden=2048, WD=1e-5; LR=1e-4 ×0.95/50ep |
| **RNNf** | 1-layer GRU (nonlinear activation + gating). 1-step: hidden=1024, no WD; Multi: hidden=2048, WD=0; LR=1e-4 ×0.95/50ep |
| **RNN PSID** | Behavior hidden=64, neuron hidden=512; LR=1e-4; no WD. *1-step only.* |
| **NDT** | 3-layer decoder-based Transformer. Attn dim=1024 (both 1-step and multi); 1-step: WD=0; multi: WD=1e-5; LR=1e-4 ×0.95/50ep |
| **STNDT** | 10 spatio-temporal blocks (spatial attn dim=context length, temporal attn dim=C); WD=1e-5 (1-step), 1e-4 (multi); LR=1e-4 ×0.95/50ep |
| **TERN** | 1-layer GRU encoder + 1 Transformer decoder layer; hidden=1024 for both; 1-step: WD=0; multi: WD=1e-4; LR=1e-4 ×0.95/50ep. *(Also called "Neural RoBERTa" in some references.)* |
| **GWNet** | 3 layers, 2 blocks/layer + 1 end CNN + 1 read-out CNN; res/dil ch=64, skip ch=128, end ch=64; kernel=2 all; LR=5e-4 ×0.95/50ep; no WD |
| **DCRNN** | 2 diffusion conv layers, 2 diffusion steps each; hidden=64; LR=5e-4 ×0.95/50ep; no WD; adjacency = KNN-pruned correlation matrix (predefined, not learned) |
| **GS4-G / GS4-S** | Hidden=256; WD=1e-5; LR=5e-4 ×0.95/50ep. *Multi-step only.* |

**Edge pruning for DCRNN/GraphS4mer:** KNN — each neuron connected to $\lfloor C/2 \rfloor$ nearest neighbors.

---

## 7. Quantitative Results (Monkey A & B)

### One-step Forecasting

| Model | Monkey B R² | Monkey B Corr | Monkey B MSE | Monkey A R² | Monkey A Corr | Monkey A MSE |
|---|---|---|---|---|---|---|
| STNDT | 0.857±2e-3 | 0.932±8e-4 | 0.0089±2e-4 | 0.879±1e-3 | 0.939±7e-4 | 0.0074±7e-5 |
| NDT | 0.897±3e-4 | 0.950±1e-4 | 0.0058±2e-5 | 0.924±6e-4 | 0.962±2e-4 | 0.0046±4e-5 |
| LFADS | 0.846±1e-4 | 0.923±1e-4 | 0.0088±9e-5 | 0.903±7e-4 | 0.951±2e-4 | 0.0060±5e-5 |
| RNNf | 0.909±6e-4 | 0.954±2e-4 | 0.0052±4e-5 | 0.926±7e-4 | 0.963±2e-4 | 0.0045±4e-5 |
| RNN PSID | 0.915±7e-4 | 0.957±3e-4 | 0.0048±5e-5 | 0.908±3e-4 | 0.953±4e-4 | 0.0049±1e-5 |
| LRNN | 0.916±8e-4 | 0.957±4e-4 | 0.0047±4e-5 | 0.927±3e-4 | 0.963±8e-5 | 0.0045±2e-5 |
| TERN | 0.888±1e-3 | 0.945±7e-4 | 0.0067±2e-4 | 0.929±4e-4 | 0.964±2e-4 | 0.0043±2e-5 |
| DCRNN | 0.964±1e-4 | 0.982±8e-5 | 0.0020±4e-6 | 0.977±2e-3 | 0.988±8e-4 | 0.0014±1e-4 |
| GWNet | 0.942±2e-3 | 0.971±1e-3 | 0.0033±1e-4 | 0.949±2e-3 | 0.974±1e-3 | 0.0031±1e-4 |
| **AMAG** | **0.973±2e-3** | **0.986±1e-3** | **0.0015±1e-4** | **0.979±7e-4** | **0.990±4e-4** | **0.0013±4e-5** |

### Multi-step Forecasting

| Model | Monkey B R² | Monkey B Corr | Monkey B MSE | Monkey A R² | Monkey A Corr | Monkey A MSE |
|---|---|---|---|---|---|---|
| LFADS | 0.427±4e-3 | 0.672±6e-3 | 0.0338±4e-4 | 0.731±2e-3 | 0.855±1e-3 | 0.0164±1e-4 |
| STNDT | 0.525±4e-3 | 0.725±2e-3 | 0.0288±2e-4 | 0.726±9e-3 | 0.852±6e-3 | 0.0167±6e-4 |
| LRNN | 0.507±6e-3 | 0.725±1e-3 | 0.0286±4e-4 | 0.696±6e-4 | 0.833±1e-3 | 0.0187±3e-5 |
| RNNf | 0.472±9e-3 | 0.694±6e-3 | 0.0314±5e-4 | 0.733±2e-3 | 0.856±8e-4 | 0.0163±1e-4 |
| TERN | 0.559±3e-3 | 0.752±2e-3 | 0.0265±2e-4 | 0.746±1e-3 | 0.865±4e-4 | 0.0154±8e-5 |
| NDT | 0.575±4e-3 | 0.773±3e-3 | 0.0250±2e-4 | 0.756±3e-3 | 0.873±1e-3 | 0.0149±2e-4 |
| GWNet | 0.588±2e-3 | 0.769±2e-3 | 0.0242±4e-4 | 0.724±1e-3 | 0.851±2e-4 | 0.0168±9e-5 |
| GS4-S | 0.600±7e-3 | 0.782±1e-3 | 0.0236±3e-4 | 0.740±5e-3 | 0.861±3e-3 | 0.0158±3e-4 |
| GS4-G | 0.659±3e-3 | 0.812±3e-3 | 0.0194±1e-4 | 0.753±8e-4 | 0.869±6e-4 | 0.0149±5e-5 |
| DCRNN | 0.635±4e-3 | 0.797±2e-3 | 0.0208±3e-4 | 0.756±2e-3 | 0.870±9e-4 | 0.0148±1e-4 |
| **AMAG** | **0.665±2e-3** | **0.817±1e-3** | **0.0192±3e-4** | **0.763±4e-3** | **0.874±2e-3** | **0.0144±2e-4** |

---

## 8. Analyses Specific to Monkey A & B

### 8.1 Channel Importance via Adjacency Weight Grouping (Monkey A)
1. Train AMAG on Monkey A (239 channels)
2. Group channels into 4 bins (High/Mid-High/Mid-Low/Low) by **total connection strength** in $A_a$ — i.e., the sum of absolute adjacency weights for each channel (row sum or equivalent aggregate of $A_a$)
3. For each group: mask those channels as zero input, run inference, record R²
4. Repeat with NDT for comparison
5. Expected: High-weight group masking causes largest R² drop for both models (Fig. 4A shows R² range ~0.45–0.60)

### 8.2 Spatial Proximity Alignment (Monkey A µECoG array)
1. Select a target channel T
2. Extract row T from learned $A_a$
3. Visualize as heatmap using physical µECoG electrode layout (239-electrode arrangement)
4. Expected: stronger interaction weights for spatially neighboring channels

### 8.3 Neural Trajectory PCA (Monkey A or B)
1. Collect original neural recordings $X$ and AMAG hidden embeddings $Z$ for all test trials
2. Apply PCA to reduce to 2D separately for $X$ and $Z$
3. Color-code by timepoint: Movement On - 300ms, Movement On, Movement On + 450ms
4. Expected: $X$ tangled; $Z$ shows disentangled circular structure

### 8.4 Adjacency Matrix Visualization (Monkey A)
1. Compute 239×239 correlation matrix from training data
2. Train AMAG; save learned $A_a$
3. Visualize both (scale: -1.0 to 1.0)
4. Expected: learned $A_a$ more sparse than correlation matrix

---

## 9. Ablation Study (Monkey B, Multi-step)

| Variant | Description | R² | Corr | MSE |
|---|---|---|---|---|
| –AG | Remove both Add and Modulator | 0.424±1e-3 | 0.650±1e-3 | 0.0456±1e-4 |
| amAG | Remove self-connection | 0.427±6e-3 | 0.652±4e-3 | 0.0453±5e-4 |
| -MAG | Remove Add only | 0.647±2e-3 | 0.805±6e-4 | 0.0274±2e-4 |
| ã MAG | Only Adaptor in Add | 0.648±1e-3 | 0.806±4e-4 | 0.0273±6e-5 |
| A-AG | Remove Modulator only | 0.652±9e-4 | 0.807±3e-4 | 0.0268±1e-4 |
| aMAG | Remove Adaptor from Add | 0.655±2e-3 | 0.810±1e-3 | 0.0269±2e-4 |
| AM-G (Rand Init) | Non-learnable adj, random init | 0.575±2e-2 | 0.767±7e-3 | 0.0329±1e-3 |
| AM-G (Corr Init) | Non-learnable adj, corr init | 0.617±2e-4 | 0.786±7e-4 | 0.0296±2e-5 |

> **Note on AM-G:** The paper text references "AM-G (Rand Init 1)" and "AM-G (Rand Init 2)" as two separate random seeds to demonstrate variance, while Table 4 shows only one representative row. This is to illustrate that random init for a **non-learnable** adjacency is sensitive to seed (high variance ±2e-2), whereas a learnable adjacency (AMAG) with random init converges stably regardless of seed.
| AMAG (Rand Init) | Full model, random init | 0.652±1e-3 | 0.807±8e-4 | 0.0270±1e-4 |
| **AMAG (Corr Init)** | **Full model, corr init** | **0.657±2e-3** | **0.811±2e-3** | **0.0266±2e-4** |

---

## 10. Additional Implementation Details (Appendix B.3 & B.4)

### 10.1 Hyperparameter Search Space (All Models)

| Hyperparameter | Values Searched |
|---|---|
| Learning rate | 1e-3, 5e-4, 1e-4 |
| Hidden size | 64, 128, 512, 1024, 2048 |
| Weight decay | 0, 1e-4, 1e-5, 1e-6, 1e-7 |

Best configuration selected by R² on validation set, evaluated every 10 epochs.

### 10.2 NDT Attention Dimension (A & B datasets)

- **Chosen:** 1024 (both one-step and multi-step)
- **Rationale:** Scaling study (Fig. 3B in paper) shows performance plateaus for dim>1024; weight decay helps for smaller dims but becomes less effective at larger dims. For A/B datasets (large C), 1024 is optimal.

### 10.3 GRU vs Transformer for Multi-step TE/TR (Monkey B)

From Table 5 in paper — both variants tested on Monkey B multi-step:

| Variant | R² | Corr | MSE |
|---|---|---|---|
| AMAG-G (GRU) | 0.653 | 0.807 | **0.0283** |
| AMAG-T (Transformer) | **0.658** | **0.811** | 0.0285 |

**Decision:** Transformer used as default for multi-step TE/TR (slightly better R² and Corr).  
Note: MSE is marginally better for GRU; Transformer is preferred for overall performance.

### 10.4 Correlation Initialization: Stability vs. Final Performance

- **Final R² (Monkey B, multi-step):** Rand Init = 0.659, Corr Init = 0.658 — nearly identical
- **Training stability:** Corr-initialized learning curves are smoother (less fluctuation in both train and test R²)
- **Early training speed:** Corr init achieves higher R² in early epochs → faster convergence
- **Conclusion:** Corr init recommended for stable, reproducible training; random init works but is noisier

### 10.5 Adjacency Matrix Evolution During Training

- **Before training:** $A_a$ = correlation matrix (dense, many values near 1.0)
- **After training:** $A_a$ is more sparse (fewer values close to 1.0, many pruned toward 0)
- **Interpretation:** AMAG adaptively prunes uninformative connections; retains only forecasting-relevant edges
- **Visualization scale:** [-1.0, 1.0] (Fig. 8 in paper, 239×239 for Monkey A)

### 10.6 STNDT Architecture Details (A & B)

- **Layers:** 10 spatio-temporal blocks
- **Each block:** 1 spatial attention Transformer + 1 temporal attention Transformer
- **Spatial Transformer hidden dim:** = temporal context length (number of timesteps in context window, i.e., 5)
- **Temporal Transformer hidden dim:** = number of input neurons (C = 239 for Monkey A, 87 for Monkey B)
- **Training epochs:** 1000 (both one-step and multi-step — unlike most other models which use 500)
- **Weight decay:** 1e-5 (one-step), 1e-4 (multi-step)
- **LR:** 1e-4 decaying ×0.95 every 50 epochs

### 10.7 Why RNN PSID is Excluded from Multi-step

RNN PSID's first training stage requires access to all previous time steps at a given moment to learn the behaviorally-relevant neural subspace. In multi-step forecasting, only the context window is available (not the full future sequence), making this stage infeasible. Therefore RNN PSID is **one-step only**.

### 10.8 Why GS4-G and GS4-S are Excluded from One-step

GraphS4mer is designed to learn dynamic (time-dependent) adjacency matrices across full temporal windows. This design increases computational complexity and is unnecessary for one-step prediction (which only requires predicting $t+1$ from past context). Therefore GS4-G and GS4-S are **multi-step only**.

---

## 11. Code & Data

- **GitHub:** https://github.com/shlizee/AMAG
- **Data (A & B):** To be released via project page; includes neural time series + electrode location metadata
- **Reference for electrode array:** Trumpis et al., *Journal of Neural Engineering*, 18(3), 2021
- **Ethics approval:** University of Washington Institutional Animal Care and Use Committee (UW IACUC)
