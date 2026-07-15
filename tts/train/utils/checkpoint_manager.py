import os
from typing import Any

import torch


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        keep_best_epoch_count: int = 3,
        keep_best_step_count: int = 3,
        keep_last_count: int = 3,
        monitor_loss: str = "mel",
    ):
        """
        Manages checkpoint saving.

        Keeps:
        - last.pth
        - latest periodic step checkpoints
        - top-K best epoch checkpoints
        - top-K best step checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.keep_best_epoch_count = keep_best_epoch_count
        self.keep_best_step_count = keep_best_step_count
        self.keep_last_count = keep_last_count
        self.monitor_loss = monitor_loss

        self.best_epoch_checkpoints: list[tuple[float, str]] = []
        self.best_step_checkpoints: list[tuple[float, str]] = []
        self.last_checkpoints: list[str] = []

    def _make_state(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        epoch: int,
    ) -> dict[str, Any]:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "epoch": epoch,
        }

    def _get_score(self, loss_values: Any) -> float:
        if loss_values is None:
            raise ValueError("loss_values must be provided for best checkpoint saving.")

        score = getattr(loss_values, self.monitor_loss)

        if isinstance(score, torch.Tensor):
            score = score.detach().cpu().item()

        return float(score)

    def _save_topk_best(
        self,
        state: dict[str, Any],
        score: float,
        ckpt_path: str,
        storage: list[tuple[float, str]],
        keep_count: int,
        label: str,
    ) -> None:
        """
        Save checkpoint only if it belongs to top-K.
        Lower score is assumed to be better.
        """
        storage.append((score, ckpt_path))
        storage.sort(key=lambda x: x[0])

        topk_paths = {path for _, path in storage[:keep_count]}

        if ckpt_path in topk_paths:
            torch.save(state, ckpt_path)
            print(f"🌟 Best {label} {self.monitor_loss} saved ({score:.4f}): {ckpt_path}")

        while len(storage) > keep_count:
            _, removed_path = storage.pop(-1)

            if os.path.exists(removed_path):
                try:
                    os.remove(removed_path)
                except OSError:
                    pass

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        epoch: int,
        loss_values: Any = None,
        save_periodic: bool = True,
        save_best_epoch: bool = False,
        save_best_step: bool = False,
    ) -> None:
        state = self._make_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=epoch,
        )

        # 1. Always update last.pth
        last_path = os.path.join(self.checkpoint_dir, "last.pth")
        torch.save(state, last_path)

        # 2. Periodic latest step checkpoint
        if save_periodic:
            ckpt_name = f"ckpt_step_{step}.pth"
            ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)

            torch.save(state, ckpt_path)
            self.last_checkpoints.append(ckpt_path)

            while len(self.last_checkpoints) > self.keep_last_count:
                old_ckpt = self.last_checkpoints.pop(0)

                if os.path.exists(old_ckpt):
                    try:
                        os.remove(old_ckpt)
                    except OSError:
                        pass

            print(f"💾 Periodic checkpoint saved: {ckpt_path}")

        # Nothing else to do if no best checkpoint is requested
        if not save_best_epoch and not save_best_step:
            return

        score = self._get_score(loss_values)

        # 3. Best epoch checkpoint
        if save_best_epoch:
            ckpt_name = (
                f"best_epoch_{self.monitor_loss}_{score:.6f}" + f"_epoch_{epoch}_step_{step}.pth"
            )
            ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)

            self._save_topk_best(
                state=state,
                score=score,
                ckpt_path=ckpt_path,
                storage=self.best_epoch_checkpoints,
                keep_count=self.keep_best_epoch_count,
                label="epoch",
            )

        # 4. Best step checkpoint
        if save_best_step:
            ckpt_name = (
                f"best_step_{self.monitor_loss}_{score:.6f}" + f"_step_{step}_epoch_{epoch}.pth"
            )
            ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)

            self._save_topk_best(
                state=state,
                score=score,
                ckpt_path=ckpt_path,
                storage=self.best_step_checkpoints,
                keep_count=self.keep_best_step_count,
                label="step",
            )

    def get_best_checkpoint_path(self) -> str:
        """
        Return the best step-based checkpoint path.

        Uses in-memory bookkeeping first.
        If empty, scans checkpoint_dir for best_step checkpoints.
        Lower score is better.
        """
        valid_checkpoints = [
            (score, path) for score, path in self.best_step_checkpoints if os.path.exists(path)
        ]

        if valid_checkpoints:
            return min(valid_checkpoints, key=lambda x: x[0])[1]

        prefix = f"best_step_{self.monitor_loss}_"
        candidates: list[tuple[float, str]] = []

        for filename in os.listdir(self.checkpoint_dir):
            if not filename.startswith(prefix):
                continue
            if not filename.endswith(".pth"):
                continue

            # filename:
            # best_step_{monitor_loss}_{score}_step_{step}_epoch_{epoch}.pth
            rest = filename[len(prefix) :]
            score_str = rest.split("_step_", maxsplit=1)[0]

            try:
                score = float(score_str)
            except ValueError:
                continue

            path = os.path.join(self.checkpoint_dir, filename)
            candidates.append((score, path))

        if not candidates:
            raise FileNotFoundError(f"No best step checkpoint found in {self.checkpoint_dir!r}.")

        candidates.sort(key=lambda x: x[0])

        # Restore bookkeeping after directory scan.
        self.best_step_checkpoints = candidates[: self.keep_best_step_count]

        return candidates[0][1]
