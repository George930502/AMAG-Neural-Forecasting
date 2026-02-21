"""Domain adaptation losses for cross-session generalization.

Reference: Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep
Domain Adaptation", ECCV 2016.
"""

import torch


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute CORAL loss between source and target feature distributions.

    Minimizes the Frobenius norm of the difference between covariance matrices
    of source and target hidden representations.

    Args:
        source: (N_s, D) feature vectors from source domain
        target: (N_t, D) feature vectors from target domain
    Returns:
        Scalar CORAL loss
    """
    d = source.size(1)

    # Center features
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)

    # Covariance matrices: (D, D) — bounded by hidden_dim
    n_s = source.size(0)
    n_t = target.size(0)
    cov_s = (source_centered.T @ source_centered) / max(n_s - 1, 1)
    cov_t = (target_centered.T @ target_centered) / max(n_t - 1, 1)

    # Frobenius norm of difference, normalized by 4*d^2
    loss = ((cov_s - cov_t) ** 2).sum() / (4 * d * d)
    return loss


def compute_coral_from_hidden(h: torch.Tensor, session_ids: torch.Tensor) -> torch.Tensor:
    """Compute per-channel CORAL loss between session 0 and other sessions.

    Instead of mean-pooling all channels into a single (B, D) vector,
    computes CORAL per-channel and averages. This preserves spatial structure
    and aligns each electrode's temporal dynamics separately.

    Args:
        h: (B, T, C, D) hidden representations from TE
        session_ids: (B,) integer session IDs
    Returns:
        Scalar CORAL loss (0 if only one session in batch)
    """
    primary_mask = session_ids == 0
    other_mask = session_ids > 0

    n_primary = primary_mask.sum().item()
    n_other = other_mask.sum().item()

    if n_primary < 2 or n_other < 2:
        return torch.tensor(0.0, device=h.device)

    B, T, C, D = h.shape

    # Per-channel CORAL: for each channel, flatten time dimension
    # h_primary: (n_primary, T, C, D), h_other: (n_other, T, C, D)
    h_primary = h[primary_mask]  # (n_p, T, C, D)
    h_other = h[other_mask]      # (n_o, T, C, D)

    # Flatten time: (n, T, C, D) -> (n*T, C, D) -> per channel: (n*T, D)
    h_p_flat = h_primary.reshape(n_primary * T, C, D)
    h_o_flat = h_other.reshape(n_other * T, C, D)

    # Sample channels to keep computation bounded (max 32 channels)
    max_channels = 32
    if C > max_channels:
        # Deterministic subset: evenly spaced channels
        channel_idx = torch.linspace(0, C - 1, max_channels, device=h.device).long()
        h_p_flat = h_p_flat[:, channel_idx]
        h_o_flat = h_o_flat[:, channel_idx]
        n_channels = max_channels
    else:
        n_channels = C

    total_coral = torch.tensor(0.0, device=h.device)
    for c in range(n_channels):
        total_coral = total_coral + coral_loss(h_p_flat[:, c], h_o_flat[:, c])

    return total_coral / n_channels
