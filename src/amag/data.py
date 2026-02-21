import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


def compute_norm_stats(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel-feature mean and std from data.

    Args:
        data: (N, T, C, F) array
    Returns:
        mean, std: each (1, C*F) arrays
    """
    n, t, c, f = data.shape
    flat = data.reshape(n * t, c * f)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    return mean, std


def compute_sample_norm_stats(context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample normalization stats from context window.

    Matches the TTA normalization used in submission/model.py predict().

    Args:
        context: (T_ctx, C, F) context window for a single sample
    Returns:
        mean, std: each (1, C*F) arrays
    """
    t, c, f = context.shape
    flat = context.reshape(t, c * f)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    return mean, std


def normalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize data to [-1, 1] using [mean-4*std, mean+4*std] range.

    Args:
        data: (N, T, C, F) or (T, C, F)
        mean, std: (1, C*F)
    Returns:
        Normalized data, same shape as input
    """
    orig_shape = data.shape
    c_f = mean.shape[1]
    flat = data.reshape(-1, c_f)
    lo = mean - 4 * std
    hi = mean + 4 * std
    denom = hi - lo
    denom = np.where(denom == 0, 1.0, denom)
    norm = 2 * (flat - lo) / denom - 1
    return norm.reshape(orig_shape)


def normalize_sample(data: np.ndarray, context_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize a single sample using its context window stats (TTA-style).

    Args:
        data: (T, C, F) raw data for one sample
        context_len: number of context timesteps
    Returns:
        data_norm: (T, C, F) normalized data
        mean: (1, C*F) normalization mean
        std: (1, C*F) normalization std
    """
    context = data[:context_len]  # (T_ctx, C, F)
    mean, std = compute_sample_norm_stats(context)
    data_norm = normalize(data, mean, std)
    return data_norm, mean, std


def denormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray,
                num_channels: int, num_features: int = 9) -> np.ndarray:
    """Denormalize data from [-1, 1] back to original scale.

    Args:
        data: (N, T, C) — LMP predictions only (feature 0)
        mean, std: (1, C*F) — full feature stats
    Returns:
        Denormalized data (N, T, C)
    """
    mean_lmp = mean[0, ::num_features]  # (C,)
    std_lmp = std[0, ::num_features]    # (C,)

    lo = mean_lmp - 4 * std_lmp
    hi = mean_lmp + 4 * std_lmp
    denom = hi - lo
    denom = np.where(denom == 0, 1.0, denom)

    return (data + 1) * denom / 2 + lo


def denormalize_torch(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                      num_features: int = 9) -> torch.Tensor:
    """Denormalize torch tensor from [-1, 1] back to original scale. LMP only."""
    mean_lmp = mean[0, ::num_features]
    std_lmp = std[0, ::num_features]
    lo = mean_lmp - 4 * std_lmp
    hi = mean_lmp + 4 * std_lmp
    denom = hi - lo
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return (data + 1) * denom / 2 + lo


class NeuralForecastDataset(Dataset):
    def __init__(self, data_raw: np.ndarray, context_len: int = 10,
                 session_ids: np.ndarray | None = None,
                 augment: bool = False,
                 jitter_std: float = 0.02,
                 scale_std: float = 0.1,
                 channel_drop_p: float = 0.1):
        """
        Args:
            data_raw: (N, T, C, F) raw (unnormalized) data
            context_len: number of context timesteps
            session_ids: (N,) integer session index per trial
            augment: whether to apply on-the-fly augmentation
            jitter_std: Gaussian noise std for jittering
            scale_std: per-channel random scale std
            channel_drop_p: probability of zeroing each channel
        """
        self.context_len = context_len
        self.data_raw = data_raw.astype(np.float32)
        self.session_ids = session_ids
        self.augment = augment
        self.jitter_std = jitter_std
        self.scale_std = scale_std
        self.channel_drop_p = channel_drop_p

    def __len__(self):
        return len(self.data_raw)

    def __getitem__(self, idx):
        x_raw = self.data_raw[idx].copy()  # (T, C, F)

        # Per-sample TTA normalization: use context window stats
        x_norm, mean, std = normalize_sample(x_raw, self.context_len)

        if self.augment:
            x_norm = self._apply_augmentation(x_norm)

        # Mask future: replace steps after context with copy of last context step
        x_norm[self.context_len:] = x_norm[self.context_len - 1]

        x_tensor = torch.from_numpy(x_norm)
        target_norm = torch.from_numpy(x_norm[:, :, 0].copy())  # (T, C) normalized LMP
        target_raw = torch.from_numpy(x_raw[:, :, 0])  # (T, C) raw LMP

        # Pack per-sample norm stats for denormalization during training
        mean_t = torch.from_numpy(mean)  # (1, C*F)
        std_t = torch.from_numpy(std)    # (1, C*F)

        if self.session_ids is not None:
            session_id = torch.tensor(self.session_ids[idx], dtype=torch.long)
            return x_tensor, target_norm, target_raw, session_id, mean_t, std_t

        return x_tensor, target_norm, target_raw, mean_t, std_t

    def _apply_augmentation(self, x: np.ndarray) -> np.ndarray:
        """Apply jittering, scaling, and channel dropout to normalized input.

        Args:
            x: (T, C, F) normalized data
        Returns:
            Augmented (T, C, F)
        """
        T, C, F = x.shape

        # Jittering: additive Gaussian noise
        if self.jitter_std > 0:
            x = x + np.random.randn(*x.shape).astype(np.float32) * self.jitter_std

        # Scaling: per-channel multiplicative noise (simulates impedance drift)
        if self.scale_std > 0:
            scale = np.random.normal(1.0, self.scale_std, size=(1, C, 1)).astype(np.float32)
            x = x * scale

        # Channel dropout: zero out random channels entirely
        if self.channel_drop_p > 0:
            mask = (np.random.rand(1, C, 1) > self.channel_drop_p).astype(np.float32)
            x = x * mask

        return x


def load_all_train_data(dataset_dir: str, train_files: list[str]) -> list[np.ndarray]:
    """Load all training data files for a monkey."""
    datasets = []
    base = Path(dataset_dir) / "train"
    for fname in train_files:
        path = base / fname
        data = np.load(str(path))["arr_0"]
        datasets.append(data)
    return datasets


def prepare_datasets(dataset_dir: str, train_files: list[str],
                     context_len: int = 10, val_split: float = 0.1,
                     seed: int = 42,
                     augment: bool = False,
                     jitter_std: float = 0.02,
                     scale_std: float = 0.1,
                     channel_drop_p: float = 0.1):
    """Load and prepare train/val datasets with per-sample TTA normalization.

    Each sample is normalized on-the-fly using its own context window stats,
    matching the inference-time TTA normalization in submission/model.py.
    Cross-date sessions contribute to both train and validation sets.

    Returns:
        train_dataset, val_dataset, per_session_stats
    """
    all_data = load_all_train_data(dataset_dir, train_files)

    # Compute per-session stats (still needed for backward compat / reference)
    per_session_stats = []
    for session_data in all_data:
        mean, std = compute_norm_stats(session_data)
        per_session_stats.append((mean, std))

    rng = np.random.RandomState(seed)

    # Split each session into train/val
    train_raw_parts = []
    val_raw_parts = []
    train_session_parts = []
    val_session_parts = []

    for i, session_data in enumerate(all_data):
        n = len(session_data)
        indices = rng.permutation(n)
        n_val = max(1, int(n * val_split))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_raw_parts.append(session_data[train_idx])
        val_raw_parts.append(session_data[val_idx])
        train_session_parts.append(np.full(len(train_idx), i, dtype=np.int64))
        val_session_parts.append(np.full(len(val_idx), i, dtype=np.int64))

    train_raw = np.concatenate(train_raw_parts, axis=0)
    val_raw = np.concatenate(val_raw_parts, axis=0)
    train_sessions = np.concatenate(train_session_parts, axis=0)
    val_sessions = np.concatenate(val_session_parts, axis=0)

    train_ds = NeuralForecastDataset(
        train_raw, context_len,
        session_ids=train_sessions,
        augment=augment,
        jitter_std=jitter_std,
        scale_std=scale_std,
        channel_drop_p=channel_drop_p,
    )
    val_ds = NeuralForecastDataset(val_raw, context_len,
                                    session_ids=val_sessions)

    return train_ds, val_ds, per_session_stats
