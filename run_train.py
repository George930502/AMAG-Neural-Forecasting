"""Training script for AMAG model.

Usage:
    python run_train.py beignet --phase paper    # Phase 1: paper-faithful reproduction
    python run_train.py affi --phase compete     # Phase 2: OOD competition config
    python run_train.py beignet                  # Default: Phase 2 (compete)
    python run_train.py beignet --seed 123       # Multi-seed training
"""
import sys
import argparse
from src.amag.train import train_monkey
from src.amag.config import TrainConfig, phase1_config, phase2_config


def main():
    parser = argparse.ArgumentParser(description="Train AMAG model")
    parser.add_argument("monkey", nargs="?", default="beignet",
                        choices=["beignet", "affi"],
                        help="Monkey dataset to train on")
    parser.add_argument("--phase", default="compete",
                        choices=["paper", "compete"],
                        help="paper = Phase 1 (faithful reproduction), "
                             "compete = Phase 2 (OOD extensions)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: from config). "
                             "Use different seeds for multi-seed ensemble.")
    args = parser.parse_args()

    if args.phase == "paper":
        cfg = phase1_config()
    else:
        cfg = phase2_config()

    if args.epochs is not None:
        cfg.epochs = args.epochs

    if args.seed is not None:
        cfg.seed = args.seed
        # Use seed-specific checkpoint directory for multi-seed ensemble
        cfg.checkpoint_dir = f"checkpoints/seed_{args.seed}"

    # Thermal-safe batch sizes (RTX 4090 Laptop overheats under sustained load)
    cfg.cooldown_ms = 50

    if args.monkey == "affi":
        cfg.batch_size = 8   # 239 channels = heavy
    else:
        cfg.batch_size = 16  # 89 channels = moderate

    train_monkey(args.monkey, cfg)


if __name__ == "__main__":
    main()
