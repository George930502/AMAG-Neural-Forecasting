"""Codabench-compatible submission model.

Self-contained AMAG model definition + inference with TTA normalization.
Phase 2: Snapshot ensemble (averages predictions from 3 snapshots).
Optional RevIN (Kim et al., ICLR 2022) for distribution shift handling.

Expected files alongside this script:
  - amag_{monkey}_snap1.pth, amag_{monkey}_snap2.pth, amag_{monkey}_snap3.pth
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


# ---- Transformer Block ----
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(0.0)
        self.ffn_drop = nn.Dropout(0.0)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(d_ff, d_model))

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) / self.scale, dim=-1)
        attn = self.attn_drop(attn)
        x = self.norm1(x + attn @ v)
        x = self.norm2(x + self.ffn_drop(self.ffn(x)))
        return x


# ---- TE ----
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, d_ff=256):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim)
        self.transformer = TransformerBlock(hidden_dim, d_ff)

    def forward(self, x):
        B, T, C, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * C, T, F)
        h = self.transformer(self.pe(self.embed(x)))
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
    def __init__(self, hidden_dim=64, d_ff=256):
        super().__init__()
        self.pe = PositionalEncoding(hidden_dim)
        self.transformer = TransformerBlock(hidden_dim, d_ff)
        self.output_fc = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        B, T, C, D = z.shape
        z = z.permute(0, 2, 1, 3).reshape(B * C, T, D)
        r = self.transformer(self.pe(z))
        pred = self.output_fc(r).squeeze(-1)
        return pred.reshape(B, C, T).permute(0, 2, 1)


# ---- Full AMAG ----
class AMAG(nn.Module):
    def __init__(self, num_channels, num_features=9, hidden_dim=64, d_ff=256,
                 total_len=20, use_adaptor=False, adaptor_chunk_size=8,
                 use_revin=False):
        super().__init__()
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_channels)
        self.te = TemporalEncoder(num_features, hidden_dim, d_ff)
        self.si = SpatialInteraction(num_channels, hidden_dim, total_len,
                                     use_adaptor, adaptor_chunk_size)
        self.tr = TemporalReadout(hidden_dim, d_ff)

    def forward(self, x):
        if self.use_revin:
            lmp = x[:, :, :, 0]
            lmp_norm = self.revin.normalize(lmp)
            x = x.clone()
            x[:, :, :, 0] = lmp_norm

        h = self.te(x)
        z = self.si(h)
        pred = self.tr(z)

        if self.use_revin:
            pred = self.revin.denormalize(pred)

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

    def _make_model(self, use_revin=False):
        """Create a fresh AMAG model instance matching trained config."""
        chunk = 4 if self.num_channels > 200 else 8
        return AMAG(
            num_channels=self.num_channels,
            num_features=9,
            hidden_dim=64,
            d_ff=256,
            total_len=20,
            use_adaptor=False,
            adaptor_chunk_size=chunk,
            use_revin=use_revin,
        )

    def _detect_revin(self, state_dict):
        """Check if checkpoint was trained with RevIN."""
        return any(k.startswith("revin.") for k in state_dict.keys())

    def load(self):
        base = os.path.dirname(__file__)

        # Try loading 3 snapshot checkpoints for ensemble
        snapshot_paths = [
            os.path.join(base, f"amag_{self.monkey_name}_snap{i}.pth")
            for i in range(1, 4)
        ]

        snapshots_found = [p for p in snapshot_paths if os.path.exists(p)]

        if len(snapshots_found) >= 2:
            # Detect RevIN from first snapshot
            first_sd = torch.load(snapshots_found[0], map_location=device, weights_only=True)
            self.use_revin = self._detect_revin(first_sd)

            for path in snapshots_found:
                m = self._make_model(use_revin=self.use_revin)
                state_dict = torch.load(path, map_location=device, weights_only=True)
                m.load_state_dict(state_dict)
                m.to(device)
                m.eval()
                self.models.append(m)
            print(f"Loaded {len(self.models)} snapshot models for ensemble "
                  f"(revin={self.use_revin})")
        else:
            # Fallback: single best checkpoint
            weight_path = os.path.join(base, f"amag_{self.monkey_name}_best.pth")
            state_dict = torch.load(weight_path, map_location=device, weights_only=True)
            self.use_revin = self._detect_revin(state_dict)

            m = self._make_model(use_revin=self.use_revin)
            m.load_state_dict(state_dict)
            m.to(device)
            m.eval()
            self.models.append(m)
            print(f"Loaded single best model (revin={self.use_revin})")

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

        # Run inference: average predictions from all snapshot models
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

        # Average across snapshots
        pred_norm_all = np.mean(all_preds, axis=0)  # (N, T, C)

        # Denormalize with TTA stats
        # Note: if RevIN is enabled, model already denormalizes internally
        # via instance stats. We still apply TTA denorm for the global scale.
        pred_raw = denormalize_lmp(pred_norm_all, mean, std, f)

        return pred_raw
