from abc import ABC
from typing import Any, override

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    @override
    def forward(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def inference(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def load_checkpoint(self, ckpt_path: str, device: torch.device | str = "cpu") -> None:
        """
        Loads the model weights from a specified checkpoint path.
        Handles both nested state_dicts (e.g., {"model": weights, "optimizer": ...})
        and plain state_dicts. It also verifies loaded modules at depth=1.

        Args:
            ckpt_path (str): The file path to the saved checkpoint (.pt or .pth).
            device (torch.device | str): The device to load the weights onto.
        """
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Check if the checkpoint is a nested dictionary containing the 'model' key
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            print(f"Loading nested state_dict from key 'model' in {ckpt_path}")
        else:
            state_dict = checkpoint
            print(f"Loading flat state_dict directly from {ckpt_path}")

        # Load the weights into the current model instance
        # strict=False allows loading even if some modules are missing during inference
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        # Verify the loaded weights at depth=1 (top-level modules)
        if missing_keys or unexpected_keys:
            # Extract top-level module names (e.g., 'text_encoder' from 'text_encoder.layer.weight')
            missing_modules = sorted(list(set(k.split(".")[0] for k in missing_keys)))
            unexpected_modules = sorted(list(set(k.split(".")[0] for k in unexpected_keys)))

            print("⚠️ Checkpoint load summary (strict=False):")
            if missing_modules:
                print(f"  - Missing top-level modules : {missing_modules}")
            if unexpected_modules:
                print(f"  - Unexpected top-level modules: {unexpected_modules}")
        else:
            print("✅ All weights matched perfectly.")

        print("Checkpoint loading process finished.")

    def print_parameter_summary(self) -> None:
        """
        Prints a detailed summary of parameters and memory size for each top-level module.
        """
        print("\n" + "🚀 Model Parameter Summary".center(90))
        print("=" * 90)
        print(
            f"{'Top-level Module':<30} | {'Trainable (M)':>15} | {'Total (M)':>12} | {'Size (MB)':>10}"
        )
        print("-" * 90)

        total_trainable = 0
        total_all = 0
        total_size_bytes = 0

        # Iterate through depth=1 modules (text_encoder, mel_decoder, etc.)
        for name, child in self.named_children():
            if child is None:  # pyright: ignore
                continue

            trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
            total = sum(p.numel() for p in child.parameters())

            # Element size * Number of elements for memory calculation
            size_bytes = sum(p.numel() * p.element_size() for p in child.parameters())
            size_mb = size_bytes / (1024**2)

            # Format parameters as Millions (M) with 3 decimal places
            print(f"{name:<30} | {trainable/1e6:>12.3f} M | {total/1e6:>9.3f} M | {size_mb:>10.2f}")

            total_trainable += trainable
            total_all += total
            total_size_bytes += size_bytes

        print("-" * 90)
        print(
            f"{'TOTAL SYSTEM':<30} | {total_trainable/1e6:>12.3f} M | {total_all/1e6:>9.3f} M | {total_size_bytes/(1024**2):>10.2f}"
        )
        print("=" * 90 + "\n")
