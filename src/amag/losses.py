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

    # Covariance matrices: (D, D) — bounded by hidden_dim (64)
    n_s = source.size(0)
    n_t = target.size(0)
    cov_s = (source_centered.T @ source_centered) / max(n_s - 1, 1)
    cov_t = (target_centered.T @ target_centered) / max(n_t - 1, 1)

    # Frobenius norm of difference, normalized by 4*d^2
    loss = ((cov_s - cov_t) ** 2).sum() / (4 * d * d)
    return loss


def compute_coral_from_hidden(h: torch.Tensor, session_ids: torch.Tensor) -> torch.Tensor:
    """Compute CORAL loss between session 0 (primary) and other sessions.

    Uses mean-pooled hidden states over time and channels to produce a
    (B, D) representation. This keeps the covariance matrix at (D, D)
    where D=hidden_dim (64), avoiding OOM from full T*C*D flattening.

    Args:
        h: (B, T, C, D) hidden representations from TE
        session_ids: (B,) integer session IDs
    Returns:
        Scalar CORAL loss (0 if only one session in batch)
    """
    # Mean-pool over time and channels: (B, T, C, D) -> (B, D)
    h_pooled = h.mean(dim=(1, 2))

    primary_mask = session_ids == 0
    other_mask = session_ids > 0

    n_primary = primary_mask.sum().item()
    n_other = other_mask.sum().item()

    if n_primary < 2 or n_other < 2:
        return torch.tensor(0.0, device=h.device)

    return coral_loss(h_pooled[primary_mask], h_pooled[other_mask])
