import os
import torch
from typing import Any


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        keep_best_count: int = 3,
        keep_last_count: int = 3,
        monitor_loss: str = "mel",
    ):
        """
        Manages checkpoint saving, keeping only the top-N best and latest M checkpoints.
        Also maintains a 'last.pth' symbolic-like checkpoint.
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.keep_best_count = keep_best_count
        self.keep_last_count = keep_last_count
        self.monitor_loss = monitor_loss

        # Track best scores: list of (score, path)
        self.best_checkpoints: list[tuple[float, str]] = []
        # Track last step checkpoints: list of paths
        self.last_checkpoints: list[str] = []

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        epoch: int,
        loss_values: Any = None,
        is_best: bool = False,
    ):
        # 1. Prepare State Dict
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
        }

        # 2. Save last.pth (Always update)
        last_path = os.path.join(self.checkpoint_dir, "last.pth")
        torch.save(state, last_path)

        # 3. Handle Periodic Step Checkpoint
        if not is_best:
            ckpt_name = f"ckpt_step_{step}.pth"
            ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
            torch.save(state, ckpt_path)
            self.last_checkpoints.append(ckpt_path)

            # Keep only N last checkpoints
            if len(self.last_checkpoints) > self.keep_last_count:
                old_ckpt = self.last_checkpoints.pop(0)
                if os.path.exists(old_ckpt):
                    try:
                        os.remove(old_ckpt)
                    except OSError:
                        pass
            print(f"💾 Periodic checkpoint saved: {ckpt_path}")

        # 4. Handle Best Epoch Checkpoint
        else:
            if loss_values is None:
                return

            current_score = getattr(loss_values, self.monitor_loss)
            ckpt_name = f"best_{self.monitor_loss}_epoch_{epoch}.pth"
            ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)

            # Add to best list and sort (assuming lower is better)
            self.best_checkpoints.append((current_score, ckpt_path))
            self.best_checkpoints.sort(key=lambda x: x[0])

            # If it's within top N, save it
            if any(
                path == ckpt_path for score, path in self.best_checkpoints[: self.keep_best_count]
            ):
                torch.save(state, ckpt_path)
                print(
                    f"🌟 Best {self.monitor_loss} checkpoint saved ({current_score:.4f}): {ckpt_path}"
                )

            # Remove from disk if no longer in top N
            if len(self.best_checkpoints) > self.keep_best_count:
                _, removed_path = self.best_checkpoints.pop(-1)
                if os.path.exists(removed_path):
                    try:
                        os.remove(removed_path)
                    except OSError:
                        pass
