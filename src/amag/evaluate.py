"""Local evaluation matching Codabench scoring.py logic.

Uses the submission Model class for inference (TTA normalization + snapshot ensemble),
matching exactly what Codabench will run.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add project root so submission.model can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from submission.model import Model


def score_mse(prediction: np.ndarray, solution: np.ndarray) -> float:
    """Compute MSE matching Codabench scoring: only on steps 10-19.

    Args:
        prediction: (N, 20, C) — model output (raw scale)
        solution: (N, 20, C, F) or (N, 20, C) — ground truth (raw scale)
    Returns:
        MSE float
    """
    sol_lmp = solution[:, :, :, 0] if solution.ndim == 4 else solution

    pred_t = torch.tensor(prediction[:, 10:])
    sol_t = torch.tensor(sol_lmp[:, 10:])

    mse = torch.nn.functional.mse_loss(pred_t, sol_t, reduction="mean")
    return mse.item()


def evaluate_all(checkpoint_dir: str = "checkpoints",
                 dataset_dir: str = "dataset"):
    """Evaluate using the submission Model on all train files."""

    # Copy checkpoints to submission/ so the Model class finds them
    ckpt_path = Path(checkpoint_dir)
    sub_path = Path("submission")

    for monkey_name in ["beignet", "affi"]:
        print(f"\n{'='*50}")
        print(f"  {monkey_name.upper()}")
        print(f"{'='*50}")

        # Symlink/copy checkpoints to submission dir
        _link_checkpoints(ckpt_path, sub_path, monkey_name)

        # Load submission model
        model = Model(monkey_name)
        model.load()

        # Evaluate on each training file
        if monkey_name == "beignet":
            train_files = [
                "train_data_beignet.npz",
                "train_data_beignet_2022-06-01_private.npz",
                "train_data_beignet_2022-06-02_private.npz",
            ]
        else:
            train_files = [
                "train_data_affi.npz",
                "train_data_affi_2024-03-20_private.npz",
            ]

        for fname in train_files:
            fpath = Path(dataset_dir) / "train" / fname
            if not fpath.exists():
                print(f"  {fname}: NOT FOUND")
                continue

            data = np.load(str(fpath))["arr_0"]  # (N, 20, C, 9)
            pred = model.predict(data)  # (N, 20, C)
            mse = score_mse(pred, data)

            label = "same-day" if "private" not in fname else "cross-date"
            print(f"  {fname}")
            print(f"    [{label}] MSE (steps 10-19): {mse:.2f}")


def _link_checkpoints(ckpt_dir: Path, sub_dir: Path, monkey_name: str):
    """Copy checkpoint files to submission directory for Model class."""
    import shutil

    for pattern in [f"amag_{monkey_name}_snap*.pth",
                    f"amag_{monkey_name}_best.pth",
                    f"amag_{monkey_name}_ema_best.pth"]:
        for src in ckpt_dir.glob(pattern):
            dst = sub_dir / src.name
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(str(src), str(dst))


if __name__ == "__main__":
    evaluate_all()
