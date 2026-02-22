import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from .config import TrainConfig, MONKEY_CONFIGS
from .data import prepare_datasets, denormalize_torch, compute_norm_stats
from .adjacency import compute_correlation_matrix
from .model import AMAG
from .losses import compute_mmd_from_hidden, spectral_loss


def _clean_state_dict(state_dict):
    """Strip '_orig_mod.' prefix added by torch.compile."""
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.removeprefix("_orig_mod.")] = v
    return cleaned


class EMAModel:
    """Exponential Moving Average of model weights.

    Reference: Polyak & Juditsky 1992; Izmailov et al. UAI 2018.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def mixup_batch(x1, tn1, tr1, x2, tn2, tr2, alpha=0.3):
    """Mixup two batches with Beta-distributed lambda.

    Reference: Zhang et al. ICLR 2018.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    x = lam * x1 + (1 - lam) * x2
    tn = lam * tn1 + (1 - lam) * tn2
    tr = lam * tr1 + (1 - lam) * tr2
    return x, tn, tr


def _unpack_batch(batch):
    """Unpack batch tuple, handling both 3-tuple and 4-tuple (with session_id)."""
    if len(batch) == 4:
        x, target_norm, target_raw, session_ids = batch
        return x, target_norm, target_raw, session_ids
    x, target_norm, target_raw = batch
    return x, target_norm, target_raw, None


def train_monkey(monkey_name: str, cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    phase_label = "Phase 1 (paper)" if cfg.optimizer_type == "adam" else "Phase 2 (compete)"
    print(f"Training {monkey_name} on {device} (AMP: {use_amp}, dtype: {amp_dtype})")
    print(f"{phase_label}: optimizer={cfg.optimizer_type}, scheduler={cfg.scheduler_type}, "
          f"adaptor={cfg.use_adaptor}, revin={cfg.use_revin}, "
          f"mmd={cfg.mmd_lambda}, spectral={cfg.spectral_lambda}, "
          f"EMA={cfg.use_ema}, Mixup={cfg.use_mixup}, "
          f"hidden_dim={cfg.hidden_dim}, heads={cfg.num_heads}, layers={cfg.num_layers}, "
          f"loss={cfg.loss_type}, channel_attn={cfg.use_channel_attn}, "
          f"feature_paths={cfg.use_feature_pathways}, "
          f"dropout={cfg.dropout}, wd={cfg.weight_decay}")

    # Enable cudnn autotuner for fixed-size inputs (all trials same shape)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    monkey_cfg = MONKEY_CONFIGS[monkey_name]
    C = monkey_cfg.num_channels

    # Prepare data (per-session normalization + augmentation)
    train_ds, val_ds, per_session_stats = prepare_datasets(
        dataset_dir=cfg.dataset_dir,
        train_files=monkey_cfg.train_files,
        context_len=cfg.context_len,
        val_split=cfg.val_split,
        seed=cfg.seed,
        augment=(cfg.aug_jitter_std > 0 or cfg.aug_scale_std > 0 or cfg.aug_channel_drop_p > 0),
        jitter_std=cfg.aug_jitter_std,
        scale_std=cfg.aug_scale_std,
        channel_drop_p=cfg.aug_channel_drop_p,
        freq_augment=True,  # Enable frequency augmentation for compete mode
    )
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Val uses primary session stats for denormalization
    val_mean, val_std = per_session_stats[0]
    val_mean_t = torch.from_numpy(val_mean).float().to(device)
    val_std_t = torch.from_numpy(val_std).float().to(device)

    # Save per-session normalization stats alongside checkpoints
    # so the submission model can use identical normalization
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    norm_stats_path = ckpt_dir / f"norm_stats_{monkey_name}.npz"
    stats_dict = {}
    for i, (m, s) in enumerate(per_session_stats):
        stats_dict[f"mean_{i}"] = m
        stats_dict[f"std_{i}"] = s
    stats_dict["num_sessions"] = np.array(len(per_session_stats))
    np.savez(str(norm_stats_path), **stats_dict)
    print(f"Saved normalization stats to {norm_stats_path}")

    # Gradient accumulation: effective_batch = batch_size * accumulate_steps
    accumulate_steps = cfg.accumulate_steps

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True,
                            persistent_workers=True)

    # Compute correlation matrix from normalized training data
    corr = compute_correlation_matrix(train_ds.data_norm)
    corr_tensor = torch.from_numpy(corr).to(device)

    # Adaptor chunk size: smaller for large channel counts to save memory
    adaptor_chunk = 4 if C > 200 else 8

    # Determine num_sessions from data
    num_sessions = len(monkey_cfg.train_files)

    # Build model
    model = AMAG(
        num_channels=C,
        num_features=cfg.num_features,
        hidden_dim=cfg.hidden_dim,
        d_ff=cfg.d_ff,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        total_len=cfg.total_len,
        corr_matrix=corr_tensor,
        dropout=cfg.dropout,
        use_adaptor=cfg.use_adaptor,
        adaptor_chunk_size=adaptor_chunk,
        use_revin=cfg.use_revin,
        use_channel_attn=cfg.use_channel_attn,
        use_feature_pathways=cfg.use_feature_pathways,
        use_session_embed=cfg.use_session_embed,
        num_sessions=num_sessions,
    ).to(device)

    # Keep reference to raw module for sub-module access
    raw_model = model

    # torch.compile for fused kernels (15-20% speedup, one-time cost)
    if device.type == "cuda":
        try:
            import triton  # noqa: F401
            model = torch.compile(model)
            print("torch.compile: enabled")
        except (ImportError, Exception) as e:
            print(f"torch.compile: skipped ({type(e).__name__})")

    param_count = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Optimizer selection
    if cfg.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                       weight_decay=cfg.weight_decay)

    # Scheduler selection
    if cfg.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_decay_every, gamma=cfg.lr_decay)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.snapshot_cycle_len, T_mult=1)

    # Warmup scheduler (linear warmup)
    if cfg.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=cfg.warmup_epochs)

    # EMA model
    ema = EMAModel(model, decay=cfg.ema_decay) if cfg.use_ema else None

    # Loss function selection
    if cfg.loss_type == "huber":
        criterion = nn.SmoothL1Loss(beta=1.0)
    else:
        criterion = nn.MSELoss()

    use_mmd = cfg.mmd_lambda > 0
    use_spectral = cfg.spectral_lambda > 0

    best_val_mse = float("inf")
    best_ema_val_mse = float("inf")
    patience_counter = 0

    # Snapshot tracking — with validation MSE for quality-gated saving
    snapshots_saved = 0
    last_snapshot_epoch = -1
    snapshot_val_mses = []  # Track val MSE at each snapshot epoch

    # Persistent mixup DataLoader (reused across epochs, not recreated)
    mixup_loader = None
    mixup_iter = None
    if cfg.use_mixup:
        mixup_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=True,
                                  persistent_workers=True)

    for epoch in range(1, cfg.epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_mmd_sum = 0.0
        train_spec_sum = 0.0
        train_count = 0

        # Reset mixup iterator each epoch (reuses existing DataLoader)
        if cfg.use_mixup:
            mixup_iter = iter(mixup_loader)

        optimizer.zero_grad()
        accum_count = 0

        for batch in train_loader:
            x, target_norm, target_raw, session_ids = _unpack_batch(batch)
            x = x.to(device)
            target_norm = target_norm.to(device)

            # Mixup: mix with a random second batch
            if cfg.use_mixup:
                try:
                    batch2 = next(mixup_iter)
                except StopIteration:
                    mixup_iter = iter(mixup_loader)
                    batch2 = next(mixup_iter)
                x2, tn2, tr2, _ = _unpack_batch(batch2)
                x2 = x2.to(device)
                tn2 = tn2.to(device)
                x, target_norm, _ = mixup_batch(
                    x, target_norm, target_raw,
                    x2, tn2, tr2,
                    alpha=cfg.mixup_alpha,
                )

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                # Session IDs for model
                sess_ids_device = session_ids.to(device) if session_ids is not None else None

                # Always use the same forward path for training and evaluation
                pred = model(x, sess_ids_device)
                mmd_l = torch.tensor(0.0, device=device)

                # MMD loss: compute from hidden states if enabled
                if use_mmd and session_ids is not None:
                    h = raw_model.get_te_hidden(x, sess_ids_device)
                    mmd_l = compute_mmd_from_hidden(h, sess_ids_device)

                # Loss on forecast window only (matching competition scoring)
                forecast_pred = pred[:, cfg.context_len:]
                forecast_target = target_norm[:, cfg.context_len:]

                mse_loss = criterion(forecast_pred, forecast_target)

                # Spectral loss
                spec_l = torch.tensor(0.0, device=device)
                if use_spectral:
                    spec_l = spectral_loss(forecast_pred, forecast_target)

                loss = mse_loss + cfg.mmd_lambda * mmd_l + cfg.spectral_lambda * spec_l
                # Scale for gradient accumulation
                loss = loss / accumulate_steps

            loss.backward()
            accum_count += 1

            if accum_count >= accumulate_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0

                # EMA update (once per optimizer step, not per micro-batch)
                if ema is not None and epoch >= cfg.ema_start_epoch:
                    ema.update(model)

            train_loss_sum += mse_loss.item() * x.size(0)
            train_mmd_sum += mmd_l.item() * x.size(0)
            train_spec_sum += spec_l.item() * x.size(0)
            train_count += x.size(0)

        # Flush any remaining accumulated gradients
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None and epoch >= cfg.ema_start_epoch:
                ema.update(model)

        # Scheduler step
        if cfg.warmup_epochs > 0 and epoch <= cfg.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        train_mse = train_loss_sum / train_count
        train_mmd_avg = train_mmd_sum / train_count if use_mmd else 0
        train_spec_avg = train_spec_sum / train_count if use_spectral else 0

        # --- Snapshot saving at cosine cycle minima ---
        if (cfg.scheduler_type == "cosine"
                and epoch % cfg.snapshot_cycle_len == 0
                and snapshots_saved < cfg.num_snapshots
                and epoch != last_snapshot_epoch):
            snapshots_saved += 1
            snap_path = ckpt_dir / f"amag_{monkey_name}_snap{snapshots_saved}.pth"
            if ema is not None and epoch >= cfg.ema_start_epoch:
                torch.save(_clean_state_dict(ema.state_dict()), snap_path)
            else:
                torch.save(_clean_state_dict(model.state_dict()), snap_path)
            last_snapshot_epoch = epoch
            print(f"  -> Snapshot {snapshots_saved}/{cfg.num_snapshots} saved at epoch {epoch}")

        # --- Validate ---
        if epoch % cfg.val_every == 0 or epoch == 1:
            model.eval()
            val_mse_raw = evaluate_raw_mse(
                model, val_loader, val_mean_t, val_std_t, cfg, device,
                use_amp=use_amp, amp_dtype=amp_dtype)

            # Also evaluate EMA model
            ema_mse_str = ""
            if ema is not None and epoch >= cfg.ema_start_epoch:
                ema_val_mse = evaluate_raw_mse(
                    ema.shadow, val_loader, val_mean_t, val_std_t, cfg, device,
                    use_amp=use_amp, amp_dtype=amp_dtype)
                ema_mse_str = f" | EMA MSE: {ema_val_mse:.6f}"

                if ema_val_mse < best_ema_val_mse:
                    best_ema_val_mse = ema_val_mse
                    save_path = ckpt_dir / f"amag_{monkey_name}_ema_best.pth"
                    torch.save(_clean_state_dict(ema.state_dict()), save_path)

            lr = optimizer.param_groups[0]["lr"]
            aux_str = ""
            if use_mmd:
                aux_str += f" | MMD: {train_mmd_avg:.6f}"
            if use_spectral:
                aux_str += f" | Spec: {train_spec_avg:.6f}"
            print(f"Epoch {epoch:4d} | Train MSE(norm): {train_mse:.6f}{aux_str} | "
                  f"Val MSE(raw): {val_mse_raw:.6f}{ema_mse_str} | LR: {lr:.6f}")

            # Track best using whichever is better (EMA or regular)
            effective_mse = val_mse_raw
            if ema is not None and epoch >= cfg.ema_start_epoch:
                effective_mse = min(val_mse_raw, ema_val_mse)

            if effective_mse < best_val_mse:
                best_val_mse = effective_mse
                patience_counter = 0
                save_path = ckpt_dir / f"amag_{monkey_name}_best.pth"
                if ema is not None and epoch >= cfg.ema_start_epoch and ema_val_mse <= val_mse_raw:
                    torch.save(_clean_state_dict(ema.state_dict()), save_path)
                    print(f"  -> New best (EMA)! Saved to {save_path}")
                else:
                    torch.save(_clean_state_dict(model.state_dict()), save_path)
                    print(f"  -> New best! Saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    # If we didn't collect enough snapshots, save remaining from current state
    if cfg.scheduler_type == "cosine":
        while snapshots_saved < cfg.num_snapshots:
            snapshots_saved += 1
            snap_path = ckpt_dir / f"amag_{monkey_name}_snap{snapshots_saved}.pth"
            if ema is not None:
                torch.save(_clean_state_dict(ema.state_dict()), snap_path)
            else:
                torch.save(_clean_state_dict(model.state_dict()), snap_path)
            print(f"  -> Snapshot {snapshots_saved}/{cfg.num_snapshots} saved (final)")

    print(f"\nTraining complete. Best val MSE (raw): {best_val_mse:.6f}")
    if ema is not None:
        print(f"Best EMA val MSE (raw): {best_ema_val_mse:.6f}")
    return model


def evaluate_raw_mse(model, loader, mean_t, std_t, cfg, device, use_amp=False,
                     amp_dtype=torch.bfloat16):
    """Evaluate MSE in raw (denormalized) space, matching Codabench scoring."""
    model.eval()
    total_se = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            x, target_norm, target_raw = _unpack_batch(batch)[:3]
            x = x.to(device)
            target_raw = target_raw.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                pred_norm = model(x)

            # Denormalize in fp32
            pred_norm = pred_norm.float()
            pred_raw = denormalize_torch(
                pred_norm, mean_t, std_t, num_features=cfg.num_features)

            forecast_pred = pred_raw[:, cfg.context_len:]
            forecast_target = target_raw[:, cfg.context_len:]

            se = ((forecast_pred - forecast_target) ** 2).sum().item()
            total_count += forecast_pred.numel()
            total_se += se

    return total_se / total_count


if __name__ == "__main__":
    import sys

    monkey = sys.argv[1] if len(sys.argv) > 1 else "beignet"
    cfg = TrainConfig()

    if monkey == "affi":
        cfg.batch_size = 16

    train_monkey(monkey, cfg)
