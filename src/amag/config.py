from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MonkeyConfig:
    name: str
    num_channels: int
    train_files: list[str]
    test_files: list[str]


AFFI = MonkeyConfig(
    name="affi",
    num_channels=239,
    train_files=[
        "train_data_affi.npz",
        "train_data_affi_2024-03-20_private.npz",
    ],
    test_files=[
        "test_data_affi_masked.npz",
        "test_data_affi_2024-03-20_private_masked.npz",
    ],
)

BEIGNET = MonkeyConfig(
    name="beignet",
    num_channels=89,
    train_files=[
        "train_data_beignet.npz",
        "train_data_beignet_2022-06-01_private.npz",
        "train_data_beignet_2022-06-02_private.npz",
    ],
    test_files=[
        "test_data_beignet_masked.npz",
        "test_data_beignet_2022-06-01_private_masked.npz",
        "test_data_beignet_2022-06-02_private_masked.npz",
    ],
)

MONKEY_CONFIGS = {"affi": AFFI, "beignet": BEIGNET}


@dataclass
class TrainConfig:
    # Data
    dataset_dir: str = "dataset"
    context_len: int = 10
    total_len: int = 20
    num_features: int = 9

    # Model
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.0
    use_adaptor: bool = False

    # Training
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 1e-4
    lr_decay: float = 0.95
    lr_decay_every: int = 50
    epochs: int = 150
    val_split: float = 0.1
    val_every: int = 5
    seed: int = 42
    cooldown_ms: int = 50
    patience: int = 20
    accumulate_steps: int = 1  # Gradient accumulation: effective_batch = batch_size * this

    # Optimizer / Scheduler selection
    optimizer_type: str = "adamw"   # "adam" or "adamw"
    scheduler_type: str = "cosine"  # "step" or "cosine"

    # EMA (Exponential Moving Average)
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_start_epoch: int = 10

    # Snapshot Ensemble (CosineAnnealingWarmRestarts)
    num_snapshots: int = 3
    snapshot_cycle_len: int = 50

    # Augmentation (training only)
    aug_jitter_std: float = 0.02
    aug_scale_std: float = 0.1
    aug_channel_drop_p: float = 0.1

    # Mixup
    use_mixup: bool = True
    mixup_alpha: float = 0.3

    # OOD: RevIN (Kim et al., ICLR 2022)
    use_revin: bool = False

    # OOD: CORAL loss (Sun & Saenko, ECCV 2016)
    coral_lambda: float = 0.0

    # Output
    checkpoint_dir: str = "checkpoints"


def phase1_config() -> TrainConfig:
    """Paper-faithful AMAG config (Li et al., NeurIPS 2023)."""
    return TrainConfig(
        # Model: paper-faithful (64d, 1 head, 1 layer, adaptor)
        hidden_dim=64,
        num_heads=1,
        num_layers=1,
        use_adaptor=True,
        # Optimizer: Adam (Kingma & Ba, ICLR 2015) — paper specifies Adam, not AdamW
        optimizer_type="adam",
        weight_decay=1e-5,
        # Scheduler: StepLR x0.95 every 50 epochs — paper specification
        scheduler_type="step",
        lr=5e-4,
        lr_decay=0.95,
        lr_decay_every=50,
        # Training: paper uses 1000 epochs, 500 is thermally safer with early stopping
        epochs=500,
        val_every=10,
        patience=50,
        # Disable all non-paper techniques
        use_ema=False,
        use_mixup=False,
        aug_jitter_std=0.0,
        aug_scale_std=0.0,
        aug_channel_drop_p=0.0,
        use_revin=False,
        coral_lambda=0.0,
    )


def phase2_config() -> TrainConfig:
    """Competition config with OOD extensions for cross-date generalization."""
    return TrainConfig(
        # Model: adaptor off (overfits on small cross-date sets)
        use_adaptor=False,
        # Optimizer: AdamW (Loshchilov & Hutter, ICLR 2019)
        optimizer_type="adamw",
        weight_decay=1e-4,
        # Scheduler: CosineAnnealingWarmRestarts (6 cycles of 50 epochs)
        scheduler_type="cosine",
        lr=5e-4,
        epochs=300,
        val_every=5,
        patience=30,
        # EMA (Polyak & Juditsky, 1992)
        use_ema=True,
        ema_decay=0.999,
        ema_start_epoch=10,
        # Snapshot ensemble (Huang et al., ICLR 2017)
        num_snapshots=6,
        snapshot_cycle_len=50,
        # Augmentation
        aug_jitter_std=0.02,
        aug_scale_std=0.1,
        aug_channel_drop_p=0.1,
        # Mixup (Zhang et al., ICLR 2018)
        use_mixup=True,
        mixup_alpha=0.3,
        # OOD: RevIN (Kim et al., ICLR 2022)
        use_revin=True,
        # OOD: CORAL (Sun & Saenko, ECCV 2016)
        coral_lambda=0.1,
    )
