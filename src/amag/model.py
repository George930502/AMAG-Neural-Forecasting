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
                 use_revin: bool = False,
                 num_heads: int = 1,
                 num_layers: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.use_revin = use_revin

        if use_revin:
            # RevIN on all features: normalize (B, T, C*F) jointly
            self.revin = RevIN(num_channels * num_features)

        self.te = TemporalEncoder(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            d_ff=d_ff,
            dropout=dropout,
            num_heads=num_heads,
            num_layers=num_layers,
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
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, F) input with future steps masked
        Returns:
            pred: (B, T, C) LMP predictions for all timesteps
        """
        B, T, C, F = x.shape

        if self.use_revin:
            # RevIN on all features: reshape (B, T, C, F) -> (B, T, C*F)
            x_flat = x.reshape(B, T, C * F)
            x_flat = self.revin.normalize(x_flat)
            x = x_flat.reshape(B, T, C, F)

        h = self.te(x)    # (B, T, C, D)
        z = self.si(h)    # (B, T, C, D)
        pred = self.tr(z)  # (B, T, C)

        if self.use_revin:
            # Denormalize: pred is (B, T, C) but RevIN was on C*F
            # We need to extract the LMP component's denormalization
            # Expand pred to C*F with zeros, denorm, then extract LMP
            pred_expanded = torch.zeros(B, T, C * F, device=pred.device, dtype=pred.dtype)
            pred_expanded[:, :, ::F] = pred  # LMP at every F-th position
            pred_denorm = self.revin.denormalize(pred_expanded)
            pred = pred_denorm[:, :, ::F]  # Extract LMP back

        return pred

    def get_te_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Return TE hidden states for CORAL loss computation."""
        B, T, C, F = x.shape
        if self.use_revin:
            x_flat = x.reshape(B, T, C * F)
            x_flat = self.revin.normalize(x_flat)
            x = x_flat.reshape(B, T, C, F)
        return self.te(x)
