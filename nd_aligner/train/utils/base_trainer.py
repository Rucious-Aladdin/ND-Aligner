import os
import time
from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from typing import Any, Generic, TypeVar

import torch
from torch.utils.data import DataLoader

from nd_aligner.config.utils.io import save_config
from nd_aligner.logger.tensorboard_logger import TensorboardLogger
from nd_aligner.models.utils.base_model import BaseModel
from nd_aligner.train.utils.checkpoint_manager import CheckpointManager

T_DataConfig = TypeVar("T_DataConfig")
T_ModelConfig = TypeVar("T_ModelConfig")


class BaseTrainer(ABC, Generic[T_DataConfig, T_ModelConfig]):
    def __init__(
        self, data_config: T_DataConfig, model_config: T_ModelConfig, device: torch.device
    ):
        self.data_config: Any = data_config
        self.model_config: Any = model_config
        self.device = device
        self.train_cfg = data_config.train  # type: ignore

        print("\n" + "🛠️  Initializing Trainer ".center(90, "="))

        self.run_dir = self._setup_run_dir()
        self.logger = self._init_logger()
        self.ckpt_manager = self._init_ckpt_manager()

        self.model: BaseModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.scaler: torch.amp.GradScaler | None = None

        self.global_step = 0
        self.start_epoch = 1

        self.train_loader: DataLoader[Any] | None = None
        self.valid_loader: DataLoader[Any] | None = None

    ## Essential Abstract Methods #

    @abstractmethod
    def setup_model(
        self,
    ) -> tuple[
        BaseModel,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler | None,
    ]:
        """Returns (model, optimizer, scheduler)."""
        raise NotImplementedError()

    @abstractmethod
    def setup_dataloader(self) -> tuple[DataLoader[Any], DataLoader[Any]]:
        """Returns (train_loader, valid_loader)."""
        raise NotImplementedError()

    @abstractmethod
    def train_step(
        self,
        batch: Any,
        epoch: int,
        step: int,
    ) -> tuple[torch.Tensor, Any, Any]:
        """
        Processes one training batch and returns:
        (weighted_total_loss, metrics, model_output)
        """
        raise NotImplementedError()

    @abstractmethod
    def validation_step(
        self,
        batch: Any,
        epoch: int,
        step: int,
    ) -> tuple[float, Any, Any]:
        """
        Processes one validation batch and returns:
        (weighted_val_loss, metrics, model_output)
        """
        raise NotImplementedError()

    ## Optional Hooks ##

    def on_fit_start(self):
        """Optional subclass hook called at the very start of run()."""
        pass

    def on_train_epoch_start(
        self,
        epoch: int,
    ):
        """Optional subclass hook called before iterating over the train loader."""
        pass

    def on_train_step_end(
        self,
        batch: Any,
        epoch: int,
        step: int,
        is_step_boundary: bool,
        weighted_loss: float,
        metrics: Any,
        output: Any,
    ):
        """Optional subclass hook for train logging/visualization."""
        pass

    def on_valid_epoch_end(
        self,
        epoch: int,
        step: int,
        avg_val_loss: float,
        avg_metrics: Any,
        last_batch: Any,
        last_output: Any,
    ):
        """Optional subclass hook for validation logging/visualization."""
        pass

    def on_training_end(
        self,
        epoch: int,
        step: int,
    ):
        """Optional subclass hook for end-of-the-all-training."""
        pass

    ## RUN!! ##

    def run(self):
        print(f"🚀  Using Device: [ {self.device} ]")

        # Setup Core Components
        self.model, self.optimizer, self.scheduler = self.setup_model()
        self.train_loader, self.valid_loader = self.setup_dataloader()

        assert isinstance(self.model, BaseModel), "setup_model() must return a BaseModel instance"
        assert self.optimizer is not None, "setup_model() must initialize self.optimizer"
        assert self.train_loader is not None, "setup_dataloader() must initialize self.train_loader"
        assert self.valid_loader is not None, "setup_dataloader() must initialize self.valid_loader"

        # Checkpoint
        self.load_checkpoint()

        # On Fit Start Hook
        self.on_fit_start()

        if self.train_cfg.fp16_run and self.device.type == "cuda":
            print("⚡  AMP (Automatic Mixed Precision) Enabled.")
            self.scaler = torch.amp.GradScaler("cuda")

        if self.train_cfg.val_sanity_check:
            self._run_validation_sanity_check(self.global_step)
            print("✅  Sanity check passed.")

        print(f"🔥  Starting training for {self.train_cfg.max_epochs} epochs...")
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.start_epoch, self.train_cfg.max_epochs + 1):
            start_time = time.time()

            avg_train_loss = self.train_epoch(epoch)

            if epoch % self.train_cfg.val_interval == 0:
                val_loss, val_metrics = self.valid_epoch(epoch, self.global_step)

                epoch_time = time.time() - start_time
                print("-" * 90)
                print(
                    f"🌟  Epoch {epoch} Summary ({epoch_time:.2f}s) | "
                    + f"Avg Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )
                print("-" * 90)

                self.ckpt_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=self.global_step,
                    epoch=epoch,
                    loss_values=val_metrics,
                    save_periodic=False,
                    save_best_step=False,
                    save_best_epoch=True,
                )

        self.on_training_end(
            epoch=epoch,
            step=self.global_step,
        )

        if self.logger:
            self.logger.close()

    def load_checkpoint(self):
        assert self.optimizer is not None
        assert self.model is not None

        if self.train_cfg.continue_path and os.path.exists(self.train_cfg.continue_path):
            print(f"📥  Loading checkpoint: {self.train_cfg.continue_path}")
            self.model.load_checkpoint(self.train_cfg.continue_path, device=self.device)

            ckpt = torch.load(self.train_cfg.continue_path, map_location=self.device)
            if "optimizer" in ckpt and not self.train_cfg.reset_optimizer:
                print("   - Optimizer state restored.")
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if (
                "scheduler" in ckpt
                and self.scheduler is not None
                and not self.train_cfg.reset_scheduler
            ):
                print("   - Scheduler state restored.")
                self.scheduler.load_state_dict(ckpt["scheduler"])
            if "step" in ckpt:
                self.global_step = ckpt["step"]
            if "epoch" in ckpt:
                self.start_epoch = ckpt["epoch"] + 1
            print(f"✅  Resuming from Step: {self.global_step} | Epoch: {self.start_epoch}")

    ## General Training Methods ##

    def train_epoch(self, epoch: int) -> float:
        assert self.optimizer is not None
        assert self.model is not None
        assert self.train_loader is not None

        self.on_train_epoch_start(epoch)

        self.model.train()
        num_batches = len(self.train_loader)
        epoch_train_loss = 0.0

        mini_batch_step = 0

        for step_in_epoch, batch in enumerate(self.train_loader, 1):
            mini_batch_step += 1
            batch = self.move_batch_to_device(batch)

            is_step_boundary = (mini_batch_step % self.train_cfg.grad_accumulation_steps == 0) or (
                step_in_epoch == num_batches
            )

            if is_step_boundary:
                self.global_step += 1

            loss, metrics, output = self._train_one_step(
                batch,
                epoch,
                self.global_step,
                is_step_boundary,
            )
            epoch_train_loss += loss

            self.on_train_step_end(
                batch=batch,
                epoch=epoch,
                step=self.global_step,
                is_step_boundary=is_step_boundary,
                weighted_loss=loss,
                metrics=metrics,
                output=output,
            )

            del output
            del batch

            if is_step_boundary:
                if self.global_step % self.train_cfg.save_interval == 0:
                    self.ckpt_manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=self.global_step,
                        epoch=epoch,
                        save_periodic=True,
                        save_best_step=False,
                        save_best_epoch=False,
                    )

                if (
                    self.train_cfg.val_interval_step > 0
                    and self.global_step % self.train_cfg.val_interval_step == 0
                ):
                    print(f"🔍  [Step {self.global_step}] Running step-based validation...")

                    val_loss, val_metrics = self.valid_epoch(
                        epoch,
                        self.global_step,
                        skip_epoch_end_hook=self.train_cfg.val_interval_step_skip_hook,
                    )

                    print(f"🌟  Step {self.global_step} Validation | Val Loss: {val_loss:.4f}")

                    self.ckpt_manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=self.global_step,
                        epoch=epoch,
                        loss_values=val_metrics,
                        save_periodic=False,
                        save_best_step=True,
                        save_best_epoch=False,
                    )

                    self.model.train()

        return epoch_train_loss / num_batches

    @torch.no_grad()
    def valid_epoch(
        self,
        epoch: int,
        step: int,
        skip_epoch_end_hook: bool = False,
    ) -> tuple[float, Any]:
        assert self.model is not None
        assert self.valid_loader is not None

        self.model.eval()

        total_val_loss = 0.0
        acc_metrics: dict[str, float] | None = None
        metrics_template: Any = None
        num_processed = 0

        last_batch = None
        last_output = None

        for batch in self.valid_loader:
            batch = self.move_batch_to_device(batch)
            with torch.amp.autocast("cuda", enabled=self.scaler is not None):
                weighted_val_loss, metrics, output = self.validation_step(
                    batch,
                    epoch=epoch,
                    step=step,
                )

            metric_values = self._metric_to_dict(metrics)
            if acc_metrics is None:
                acc_metrics = {k: 0.0 for k in metric_values}
                metrics_template = metrics

            total_val_loss += weighted_val_loss
            for k, v in metric_values.items():
                acc_metrics[k] += v

            num_processed += 1
            last_batch, last_output = batch, output

        if num_processed == 0 or acc_metrics is None or metrics_template is None:
            raise ValueError("Validation loader is empty.")

        avg_val_loss = total_val_loss / num_processed
        avg_metrics_dict = {k: v / num_processed for k, v in acc_metrics.items()}
        avg_metrics = self._dict_to_metric(metrics_template, avg_metrics_dict)

        if not skip_epoch_end_hook:
            self.on_valid_epoch_end(
                epoch=epoch,
                step=step,
                avg_val_loss=avg_val_loss,
                avg_metrics=avg_metrics,
                last_batch=last_batch,
                last_output=last_output,
            )
        return avg_val_loss, avg_metrics

    ## Initialization Helpers ##

    def _setup_run_dir(self) -> str:
        if self.train_cfg.continue_dir:
            run_dir = self.train_cfg.continue_dir
            print(f"🔄  Continuing in existing directory: [ {run_dir} ]")

            os.makedirs(run_dir, exist_ok=True)

        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            base_run_name = f"{self.train_cfg.run_name}_{timestamp}"

            suffix = 0

            while True:
                if suffix == 0:
                    run_name = base_run_name
                else:
                    run_name = f"{base_run_name}_{suffix}"

                run_dir = os.path.join(
                    self.train_cfg.log_dir,
                    run_name,
                )

                try:
                    os.makedirs(run_dir, exist_ok=False)
                    break
                except FileExistsError:  # if already_exist!
                    suffix += 1

            print(f"📂  Creating new run directory: [ {run_dir} ]")

        save_config(
            self.data_config,
            os.path.join(run_dir, "data_config.json"),
        )
        save_config(
            self.model_config,
            os.path.join(run_dir, "model_config.json"),
        )

        print("📄  Configurations backed up for reproducibility.")

        return run_dir

    def _init_logger(self) -> TensorboardLogger | None:
        if self.train_cfg.use_tensorboard:
            run_name = os.path.basename(self.run_dir.rstrip("/"))
            logger = TensorboardLogger(self.train_cfg.log_dir, run_name)
            return logger
        return None

    def _init_ckpt_manager(self) -> CheckpointManager:
        return CheckpointManager(
            checkpoint_dir=os.path.join(self.run_dir, "checkpoints"),
            keep_best_epoch_count=self.train_cfg.keep_best_count,
            keep_best_step_count=self.train_cfg.keep_best_count,
            keep_last_count=self.train_cfg.keep_last_count,
            monitor_loss=self.train_cfg.monitor_loss,
        )

    ## Training Helpers ##

    def _train_one_step(
        self,
        batch: Any,
        epoch: int,
        step: int,
        is_step_boundary: bool,
    ) -> tuple[float, Any, Any]:
        assert self.optimizer is not None
        assert self.model is not None

        self.model.train()

        with torch.amp.autocast("cuda", enabled=self.scaler is not None):
            weighted_loss, metrics, output = self.train_step(batch, epoch, step)
            scaled_loss = weighted_loss / self.train_cfg.grad_accumulation_steps

        if torch.isnan(scaled_loss):
            raise ValueError("NaN loss detected.")

        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
            if is_step_boundary:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.train_cfg.grad_clip_thresh,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            scaled_loss.backward()
            if is_step_boundary:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.train_cfg.grad_clip_thresh,
                )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

        return weighted_loss.item(), metrics, output

    ## Input Accumulation Methods ##

    def _metric_to_dict(self, metrics: Any) -> dict[str, float]:
        if hasattr(metrics, "_asdict"):
            return {k: float(v) for k, v in metrics._asdict().items()}
        if isinstance(metrics, dict):
            return {str(k): float(v) for k, v in metrics.items()}
        raise TypeError("metrics must be a NamedTuple-like object or dict[str, float].")

    def _dict_to_metric(self, template: Any, values: dict[str, float]) -> Any:
        if hasattr(template, "_asdict"):
            return type(template)(**values)
        if isinstance(template, dict):
            return values
        return values

    ## Validataion Helpers ##

    @torch.no_grad()
    def _run_validation_sanity_check(self, step: int):
        assert self.model is not None
        assert self.valid_loader is not None

        if self.train_cfg.val_sanity_check_full_epoch:
            print("🔍  Running FULL epoch validation sanity check...")
            self.valid_epoch(epoch=0, step=step)
        else:
            print("🔍  Running single-batch validation sanity check...")
            self.model.eval()

            try:
                batch = next(iter(self.valid_loader))
            except StopIteration as e:
                raise ValueError("Validation loader is empty.") from e

            batch = self.move_batch_to_device(batch)

            with torch.amp.autocast("cuda", enabled=self.scaler is not None):
                val_loss, val_metrics, output = self.validation_step(batch, epoch=0, step=step)

            self.on_valid_epoch_end(
                epoch=0,
                step=step,
                avg_val_loss=val_loss,
                avg_metrics=val_metrics,
                last_batch=batch,
                last_output=output,
            )

    ## Device IO ##

    def move_batch_to_device(self, batch: Any) -> Any:
        return self._move_to_device(batch, self.device)

    def _move_to_device(self, obj: Any, device: torch.device) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device)

        if isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._move_to_device(v, device) for v in obj]

        if isinstance(obj, tuple):
            if hasattr(obj, "_fields"):
                return type(obj)(*(self._move_to_device(v, device) for v in obj))
            return tuple(self._move_to_device(v, device) for v in obj)

        if is_dataclass(obj) and not isinstance(obj, type):
            return type(obj)(
                **{f.name: self._move_to_device(getattr(obj, f.name), device) for f in fields(obj)}
            )

        return obj
