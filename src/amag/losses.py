"""Domain adaptation and auxiliary losses for cross-session generalization.

Includes:
- Multi-kernel MMD (Gretton et al., JMLR 2012)
- Spectral loss (FFT magnitude MSE)
"""

import torch
import torch.nn.functional as F


def _pairwise_dist(x, y):
    """Compute pairwise squared distances between x and y."""
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)
    dist = x_norm + y_norm.T - 2 * x @ y.T
    return dist.clamp(min=0)  # Numerical safety


def mmd_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Multi-Kernel MMD loss with median-heuristic bandwidths.

    Reference: Gretton et al., "A Kernel Two-Sample Test", JMLR 2012.
    Uses adaptive bandwidths based on median pairwise distance (standard practice).

    Args:
        source: (N_s, D) feature vectors from source domain
        target: (N_t, D) feature vectors from target domain
    Returns:
        Scalar MMD loss
    """
    all_data = torch.cat([source, target], dim=0)
    dist_all = _pairwise_dist(all_data, all_data)

    # Median heuristic for bandwidth selection
    median_dist = torch.median(dist_all[dist_all > 0]).detach()
    median_dist = median_dist.clamp(min=1e-6)

    # Use bandwidths at multiple scales around median
    bandwidths = [median_dist * f for f in [0.2, 0.5, 1.0, 2.0, 5.0]]

    dist_ss = _pairwise_dist(source, source)
    dist_tt = _pairwise_dist(target, target)
    dist_st = _pairwise_dist(source, target)

    k_ss = torch.zeros_like(dist_ss)
    k_tt = torch.zeros_like(dist_tt)
    k_st = torch.zeros_like(dist_st)

    for bw in bandwidths:
        k_ss = k_ss + torch.exp(-dist_ss / (2 * bw))
        k_tt = k_tt + torch.exp(-dist_tt / (2 * bw))
        k_st = k_st + torch.exp(-dist_st / (2 * bw))

    n_bw = len(bandwidths)
    return k_ss.mean() / n_bw + k_tt.mean() / n_bw - 2 * k_st.mean() / n_bw


def compute_mmd_from_hidden(h: torch.Tensor, session_ids: torch.Tensor) -> torch.Tensor:
    """Compute MMD loss between session 0 (primary) and other sessions.

    Uses mean-pooled hidden states over time and channels to produce a
    (B, D) representation.

    Args:
        h: (B, T, C, D) hidden representations from TE
        session_ids: (B,) integer session IDs
    Returns:
        Scalar MMD loss (0 if only one session in batch)
    """
    # Mean-pool over time and channels: (B, T, C, D) -> (B, D)
    h_pooled = h.mean(dim=(1, 2))

    primary_mask = session_ids == 0
    other_mask = session_ids > 0

    n_primary = primary_mask.sum().item()
    n_other = other_mask.sum().item()

    if n_primary < 2 or n_other < 2:
        return torch.tensor(0.0, device=h.device)

    return mmd_loss(h_pooled[primary_mask], h_pooled[other_mask])


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute spectral (frequency-domain) loss.

    Computes MSE between FFT magnitudes of predicted and target signals.
    Helps model learn correct frequency content.

    Args:
        pred: (B, T, C) predicted signal
        target: (B, T, C) target signal
    Returns:
        Scalar spectral loss
    """
    pred_fft = torch.fft.rfft(pred.float(), dim=1)
    target_fft = torch.fft.rfft(target.float(), dim=1)
    return F.mse_loss(pred_fft.abs(), target_fft.abs())
