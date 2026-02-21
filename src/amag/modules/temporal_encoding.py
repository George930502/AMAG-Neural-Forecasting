import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    """Single-head self-attention + FFN with residual connections and LayerNorm."""

    def __init__(self, d_model: int, d_ff: int = 256, dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        # Self-attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        attn_out = torch.matmul(attn_weights, v)

        # Residual + LayerNorm
        x = self.norm1(x + attn_out)

        # FFN + Residual + LayerNorm
        x = self.norm2(x + self.ffn_drop(self.ffn(x)))

        return x


class TemporalEncoder(nn.Module):
    """TE module: FC embedding + positional encoding + Transformer block.

    Shared across all channels.
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, d_ff: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim)
        self.transformer = TransformerBlock(hidden_dim, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, F) input features
        Returns:
            H: (B, T, C, D) temporal encodings
        """
        B, T, C, F = x.shape

        # Reshape to process all channels together: (B*C, T, F)
        x = x.permute(0, 2, 1, 3).reshape(B * C, T, F)

        # Embed + PE + Transformer
        h = self.embed(x)       # (B*C, T, D)
        h = self.pe(h)          # (B*C, T, D)
        h = self.transformer(h)  # (B*C, T, D)

        # Reshape back: (B, C, T, D) -> (B, T, C, D)
        h = h.reshape(B, C, T, -1).permute(0, 2, 1, 3)
        return h
