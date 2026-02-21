"""Codabench-compatible submission model.

Self-contained AMAG model definition + inference with TTA normalization.
Phase 2: Snapshot ensemble (averages predictions from up to 6 snapshots x multiple seeds).
RevIN (Kim et al., ICLR 2022) on all 9 features for distribution shift handling.
Prediction smoothing post-processing.

Expected files alongside this script:
  - amag_{monkey}_snap{1..6}.pth (default seed)
  - seed_{seed}/amag_{monkey}_snap{1..6}.pth (multi-seed)
  - Falls back to amag_{monkey}_best.pth if snapshots not found
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---- RevIN (Kim et al., ICLR 2022) ----
class RevIN(nn.Module):
    """Reversible Instance Normalization for time-series distribution shift."""

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1, 1, num_channels))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_channels))
        self._mean = None
        self._std = None

    def normalize(self, x):
        self._mean = x.mean(dim=1, keepdim=True)
        self._std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        x_norm = (x - self._mean) / self._std
        return x_norm * self.affine_weight + self.affine_bias

    def denormalize(self, x):
        x = (x - self.affine_bias) / self.affine_weight
        return x * self._std + self._mean


# ---- Positional Encoding ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ---- Transformer Block (multi-head) ----
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff=256, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.head_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(0.0)
        self.ffn_drop = nn.Dropout(0.0)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(d_ff, d_model))

    def forward(self, x):
        B, T, D = x.shape
        H = self.num_heads
        HD = self.head_dim

        q = self.q_proj(x).view(B, T, H, HD).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, HD).transpose(1, 2)

        attn = torch.softmax(q @ k.transpose(-2, -1) / self.scale, dim=-1)
        attn = self.attn_drop(attn)
        attn_out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(attn_out)

        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn_drop(self.ffn(x)))
        return x


# ---- TE ----
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, d_ff=256, num_heads=1, num_layers=1):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, d_ff, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        B, T, C, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * C, T, F)
        h = self.pe(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return h.reshape(B, C, T, -1).permute(0, 2, 1, 3)


# ---- Adaptor MLP ----
class AdaptorMLP(nn.Module):
    """S^(u,v) = sigma(MLP([H^(u), H^(v)])) in [0, 1]."""
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)


# ---- SI ----
class SpatialInteraction(nn.Module):
    def __init__(self, num_channels, hidden_dim=64, total_len=20,
                 use_adaptor=False, adaptor_chunk_size=8):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.use_adaptor = use_adaptor
        self.adaptor_chunk_size = adaptor_chunk_size
        self.A_a = nn.Parameter(torch.zeros(num_channels, num_channels))
        self.A_m = nn.Parameter(torch.zeros(num_channels, num_channels))
        self.fc_add = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mod = nn.Linear(hidden_dim, hidden_dim)
        self.beta1 = nn.Parameter(torch.tensor(1.0))
        self.beta2 = nn.Parameter(torch.tensor(1.0))
        self.beta3 = nn.Parameter(torch.tensor(1.0))
        if use_adaptor:
            adaptor_input_dim = 2 * total_len * hidden_dim
            self.adaptor = AdaptorMLP(adaptor_input_dim)

    def forward(self, h):
        if self.use_adaptor:
            a = self._add_with_adaptor(h)
        else:
            a = torch.einsum("uv,btud->btvd", self.A_a, h)
        a = self.fc_add(a)
        ws = torch.einsum("uv,btud->btvd", self.A_m, h)
        m = self.fc_mod(ws * h)
        return self.beta1 * h + self.beta2 * a + self.beta3 * m

    def _add_with_adaptor(self, h):
        B, T, C, D = h.shape
        cs = self.adaptor_chunk_size
        chunks = []
        for v_start in range(0, C, cs):
            v_end = min(v_start + cs, C)
            chunks.append(self._compute_add_chunk(h, v_start, v_end))
        return torch.cat(chunks, dim=2)

    def _compute_add_chunk(self, h, v_start, v_end):
        B, T, C, D = h.shape
        v_size = v_end - v_start
        TD = T * D
        h_flat = h.permute(0, 2, 1, 3).reshape(B, C, TD)
        h_v = h_flat[:, v_start:v_end]
        pair = torch.cat([
            h_flat.unsqueeze(2).expand(-1, -1, v_size, -1),
            h_v.unsqueeze(1).expand(-1, C, -1, -1),
        ], dim=-1)
        s = self.adaptor(pair.reshape(-1, 2 * TD)).view(B, C, v_size)
        w = s * self.A_a[:, v_start:v_end].unsqueeze(0)
        a_chunk = torch.einsum("buv,btud->btvd", w, h)
        return a_chunk


# ---- TR ----
class TemporalReadout(nn.Module):
    def __init__(self, hidden_dim=64, d_ff=256, num_heads=1, num_layers=1):
        super().__init__()
        self.pe = PositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, d_ff, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.output_fc = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        B, T, C, D = z.shape
        z = z.permute(0, 2, 1, 3).reshape(B * C, T, D)
        r = self.pe(z)
        for layer in self.layers:
            r = layer(r)
        pred = self.output_fc(r).squeeze(-1)
        return pred.reshape(B, C, T).permute(0, 2, 1)


# ---- Full AMAG ----
class AMAG(nn.Module):
    def __init__(self, num_channels, num_features=9, hidden_dim=64, d_ff=256,
                 total_len=20, use_adaptor=False, adaptor_chunk_size=8,
                 use_revin=False, num_heads=1, num_layers=1):
        super().__init__()
        self.num_features = num_features
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_channels * num_features)
        self.te = TemporalEncoder(num_features, hidden_dim, d_ff,
                                   num_heads=num_heads, num_layers=num_layers)
        self.si = SpatialInteraction(num_channels, hidden_dim, total_len,
                                     use_adaptor, adaptor_chunk_size)
        self.tr = TemporalReadout(hidden_dim, d_ff,
                                   num_heads=num_heads, num_layers=num_layers)

    def forward(self, x):
        B, T, C, F = x.shape

        if self.use_revin:
            x_flat = x.reshape(B, T, C * F)
            x_flat = self.revin.normalize(x_flat)
            x = x_flat.reshape(B, T, C, F)

        h = self.te(x)
        z = self.si(h)
        pred = self.tr(z)

        if self.use_revin:
            pred_expanded = torch.zeros(B, T, C * F, device=pred.device, dtype=pred.dtype)
            pred_expanded[:, :, ::F] = pred
            pred_denorm = self.revin.denormalize(pred_expanded)
            pred = pred_denorm[:, :, ::F]

        return pred


# ---- Normalization ----
def normalize_data(data, mean, std):
    n, t, c, f = data.shape
    flat = data.reshape(n * t, c * f)
    lo = mean - 4 * std
    hi = mean + 4 * std
    denom = hi - lo
    denom = np.where(denom == 0, 1.0, denom)
    norm = 2 * (flat - lo) / denom - 1
    return norm.reshape(n, t, c, f).astype(np.float32)


def denormalize_lmp(data, mean, std, num_features=9):
    mean_lmp = mean[0, ::num_features]
    std_lmp = std[0, ::num_features]
    lo = mean_lmp - 4 * std_lmp
    hi = mean_lmp + 4 * std_lmp
    denom = hi - lo
    denom = np.where(denom == 0, 1.0, denom)
    return (data + 1) * denom / 2 + lo


def smooth_predictions(pred, alpha=0.3):
    """Apply exponential moving average smoothing to predictions.

    Args:
        pred: (N, T, C) predictions
        alpha: smoothing factor (0 = no smoothing, 1 = replace with previous)
    Returns:
        Smoothed (N, T, C) predictions
    """
    if alpha <= 0:
        return pred
    smoothed = pred.copy()
    for t in range(1, pred.shape[1]):
        smoothed[:, t] = (1 - alpha) * pred[:, t] + alpha * smoothed[:, t - 1]
    return smoothed


# ---- Submission Model ----
class Model:
    def __init__(self, monkey_name="beignet"):
        self.monkey_name = monkey_name
        if monkey_name == "affi":
            self.num_channels = 239
        elif monkey_name == "beignet":
            self.num_channels = 89
        else:
            raise ValueError(f"Unknown monkey: {monkey_name}")

        self.models = []
        self.use_revin = False  # Detected from checkpoint

    def _detect_arch(self, state_dict):
        """Detect architecture parameters from checkpoint keys."""
        use_revin = any(k.startswith("revin.") for k in state_dict.keys())

        # Detect hidden_dim from embed layer
        hidden_dim = 64
        if "te.embed.weight" in state_dict:
            hidden_dim = state_dict["te.embed.weight"].shape[0]

        # Detect num_heads from head_dim: look for out_proj (only in multi-head)
        num_heads = 1
        if "te.layers.0.out_proj.weight" in state_dict:
            # Multi-head: has out_proj
            # Detect num_heads from q_proj vs head pattern
            q_weight = state_dict["te.layers.0.q_proj.weight"]
            d_model = q_weight.shape[0]
            # We can't directly detect num_heads from weights alone,
            # but we know d_model. Use hidden_dim / 32 as head_dim=32 heuristic.
            if d_model >= 128:
                num_heads = 4
            else:
                num_heads = 1
        elif "te.transformer.out_proj.weight" in state_dict:
            # Old multi-head format (single transformer block)
            q_weight = state_dict["te.transformer.q_proj.weight"]
            d_model = q_weight.shape[0]
            if d_model >= 128:
                num_heads = 4
            else:
                num_heads = 1

        # Detect num_layers from key pattern
        num_layers = 1
        for k in state_dict.keys():
            if k.startswith("te.layers."):
                layer_idx = int(k.split(".")[2])
                num_layers = max(num_layers, layer_idx + 1)

        # Detect d_ff from FFN
        d_ff = 256
        for k in ("te.layers.0.ffn.0.weight", "te.transformer.ffn.0.weight"):
            if k in state_dict:
                d_ff = state_dict[k].shape[0]
                break

        return use_revin, hidden_dim, num_heads, num_layers, d_ff

    def _make_model(self, use_revin=False, hidden_dim=64, num_heads=1,
                    num_layers=1, d_ff=256):
        """Create a fresh AMAG model instance matching trained config."""
        chunk = 4 if self.num_channels > 200 else 8
        return AMAG(
            num_channels=self.num_channels,
            num_features=9,
            hidden_dim=hidden_dim,
            d_ff=d_ff,
            total_len=20,
            use_adaptor=False,
            adaptor_chunk_size=chunk,
            use_revin=use_revin,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def load(self):
        base = os.path.dirname(__file__)

        # Collect checkpoint paths from multiple seeds
        all_ckpt_paths = []

        # Check for seed-specific directories
        seed_dirs = []
        for entry in os.listdir(base):
            full = os.path.join(base, entry)
            if os.path.isdir(full) and entry.startswith("seed_"):
                seed_dirs.append(full)

        if seed_dirs:
            # Multi-seed: load snapshots from each seed directory
            for seed_dir in sorted(seed_dirs):
                for i in range(1, 7):  # up to 6 snapshots
                    p = os.path.join(seed_dir, f"amag_{self.monkey_name}_snap{i}.pth")
                    if os.path.exists(p):
                        all_ckpt_paths.append(p)
                # Also check for best
                best = os.path.join(seed_dir, f"amag_{self.monkey_name}_best.pth")
                if os.path.exists(best) and not any(
                    os.path.join(seed_dir, f"amag_{self.monkey_name}_snap") in p
                    for p in all_ckpt_paths if seed_dir in p
                ):
                    all_ckpt_paths.append(best)

        # Also check base directory for snapshots (default seed)
        for i in range(1, 7):
            p = os.path.join(base, f"amag_{self.monkey_name}_snap{i}.pth")
            if os.path.exists(p):
                all_ckpt_paths.append(p)

        # Fallback to best checkpoint
        if not all_ckpt_paths:
            best = os.path.join(base, f"amag_{self.monkey_name}_best.pth")
            if os.path.exists(best):
                all_ckpt_paths.append(best)

        if not all_ckpt_paths:
            raise FileNotFoundError(
                f"No checkpoints found for {self.monkey_name} in {base}")

        # Detect architecture from first checkpoint
        first_sd = torch.load(all_ckpt_paths[0], map_location=device, weights_only=True)
        use_revin, hidden_dim, num_heads, num_layers, d_ff = self._detect_arch(first_sd)
        self.use_revin = use_revin

        for path in all_ckpt_paths:
            m = self._make_model(use_revin=use_revin, hidden_dim=hidden_dim,
                                 num_heads=num_heads, num_layers=num_layers, d_ff=d_ff)
            state_dict = torch.load(path, map_location=device, weights_only=True)
            m.load_state_dict(state_dict)
            m.to(device)
            m.eval()
            self.models.append(m)

        print(f"Loaded {len(self.models)} models for ensemble "
              f"(revin={use_revin}, hidden={hidden_dim}, heads={num_heads}, "
              f"layers={num_layers}, d_ff={d_ff})")

    def predict(self, x):
        """
        Args:
            x: (N, 20, C, 9) raw test data (steps 10-19 are repeated from step 9)
        Returns:
            predictions: (N, 20, C) raw-scale LMP predictions
        """
        context_len = 10
        n, t, c, f = x.shape

        # TTA: compute normalization stats from context window only
        context = x[:, :context_len]  # (N, 10, C, F)
        n_ctx, t_ctx, c_ctx, f_ctx = context.shape
        flat = context.reshape(n_ctx * t_ctx, c_ctx * f_ctx)
        mean = flat.mean(axis=0, keepdims=True)
        std = flat.std(axis=0, keepdims=True)

        # Normalize full input with TTA stats
        x_norm = normalize_data(x, mean, std)

        # Mask future
        x_norm[:, context_len:] = x_norm[:, context_len - 1: context_len]

        # Run inference: average predictions from all models
        batch_size = 16 if self.num_channels > 200 else 32
        all_preds = []

        for model in self.models:
            predictions = []
            with torch.no_grad():
                for i in range(0, n, batch_size):
                    batch = torch.from_numpy(x_norm[i:i + batch_size]).to(device)
                    pred_norm = model(batch)  # (B, T, C)
                    predictions.append(pred_norm.cpu().numpy())
            all_preds.append(np.concatenate(predictions, axis=0))

        # Average across all models
        pred_norm_all = np.mean(all_preds, axis=0)  # (N, T, C)

        # Denormalize with TTA stats
        pred_raw = denormalize_lmp(pred_norm_all, mean, std, f)

        # Prediction smoothing: gentle EMA on forecast window
        pred_raw = smooth_predictions(pred_raw, alpha=0.15)

        return pred_raw
