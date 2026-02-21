import torch
import torch.nn as nn
from .modules.temporal_encoding import TemporalEncoder
from .modules.spatial_interaction import SpatialInteraction
from .modules.temporal_readout import TemporalReadout
from .modules.revin import RevIN


class AMAG(nn.Module):
    """AMAG: Additive, Multiplicative and Adaptive Graph Neural Network.

    Architecture: TE (Transformer) -> SI (Add + Modulator + Adaptor) -> TR (Transformer) -> FC
    Optional: RevIN (Kim et al., ICLR 2022) for distribution shift handling.
    """

    def __init__(self, num_channels: int, num_features: int = 9,
                 hidden_dim: int = 64, d_ff: int = 256,
                 total_len: int = 20,
                 corr_matrix: torch.Tensor | None = None,
                 dropout: float = 0.0,
                 use_adaptor: bool = True,
                 adaptor_chunk_size: int = 8,
                 use_revin: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(num_channels)

        self.te = TemporalEncoder(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.si = SpatialInteraction(
            num_channels=num_channels,
            hidden_dim=hidden_dim,
            total_len=total_len,
            corr_matrix=corr_matrix,
            use_adaptor=use_adaptor,
            adaptor_chunk_size=adaptor_chunk_size,
        )

        self.tr = TemporalReadout(
            hidden_dim=hidden_dim,
            d_ff=d_ff,
            output_dim=1,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, F) input with future steps masked
        Returns:
            pred: (B, T, C) LMP predictions for all timesteps
        """
        if self.use_revin:
            # Normalize LMP channel (feature 0) per-instance
            lmp = x[:, :, :, 0]  # (B, T, C)
            lmp_norm = self.revin.normalize(lmp)
            x = x.clone()
            x[:, :, :, 0] = lmp_norm

        h = self.te(x)    # (B, T, C, D)
        z = self.si(h)    # (B, T, C, D)
        pred = self.tr(z)  # (B, T, C)

        if self.use_revin:
            pred = self.revin.denormalize(pred)

        return pred

    def get_te_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Return TE hidden states for CORAL loss computation."""
        if self.use_revin:
            lmp = x[:, :, :, 0]
            lmp_norm = self.revin.normalize(lmp)
            x = x.clone()
            x[:, :, :, 0] = lmp_norm
        return self.te(x)
