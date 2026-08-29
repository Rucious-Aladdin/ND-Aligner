# run_exp.py
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import traceback
from pathlib import Path


def natural_key(path: Path) -> list[int | str]:
    """
    Natural sort:
        R=1, R=5, R=20
    instead of:
        R=1, R=20, R=5
    """
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def discover_experiments(config_root: Path) -> list[Path]:
    """
    Find experiment directories containing both:
        data_config.json
        model_config.json
    """
    if not config_root.exists():
        raise FileNotFoundError(f"config_root does not exist: {config_root}")

    exp_dirs: list[Path] = []

    for child in config_root.iterdir():
        if not child.is_dir():
            continue

        data_config = child / "data_config.json"
        model_config = child / "model_config.json"

        if data_config.exists() and model_config.exists():
            exp_dirs.append(child)

    return sorted(exp_dirs, key=natural_key)


def run_one_experiment(exp_dir: Path) -> None:
    """
    Call nd_aligner.train.train.main() with temporary argv.
    """
    from nd_aligner.train.train import main as train_main

    data_config = exp_dir / "data_config.json"
    model_config = exp_dir / "model_config.json"

    if not data_config.exists():
        raise FileNotFoundError(f"Missing data_config.json: {data_config}")
    if not model_config.exists():
        raise FileNotFoundError(f"Missing model_config.json: {model_config}")

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "train.py",
            "-c",
            str(data_config),
            "-m",
            str(model_config),
        ]

        print("=" * 100)
        print(f"[RUN] {exp_dir.name}")
        print(f"[DATA]  {data_config}")
        print(f"[MODEL] {model_config}")
        print("=" * 100)

        train_main()

    finally:
        sys.argv = old_argv


def cleanup_cuda() -> None:
    """
    Best-effort cleanup between sequential experiments.
    This is useful when running multiple trainings in one Python process.
    """
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple ND-Aligner experiments from config directories."
    )

    parser.add_argument(
        "--config_root",
        type=str,
        default="",
        help="Root directory containing experiment config folders.",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help=(
            "CUDA device id to expose, e.g. --gpu 6. "
            "If omitted, uses existing CUDA_VISIBLE_DEVICES."
        ),
    )

    parser.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Run only selected experiment directory names. "
            "Example: --only R=1 R=5 global_softmax"
        ),
    )

    parser.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=None,
        help=("Skip selected experiment directory names. " "Example: --skip spec_dec_RF=3"),
    )

    parser.add_argument(
        "--start_from",
        type=str,
        default=None,
        help=(
            "Skip experiments until this directory name is reached. "
            "Useful for resuming a batch run."
        ),
    )

    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Maximum number of experiments to run.",
    )

    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Run experiments in reverse order.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print selected experiments without running training.",
    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue with the next experiment if one experiment fails.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Important:
    # Set CUDA_VISIBLE_DEVICES before importing train / torch.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config_root = Path(args.config_root).expanduser().resolve()
    exp_dirs = discover_experiments(config_root)

    if args.only is not None:
        only = set(args.only)
        exp_dirs = [p for p in exp_dirs if p.name in only]

    if args.skip is not None:
        skip = set(args.skip)
        exp_dirs = [p for p in exp_dirs if p.name not in skip]

    if args.start_from is not None:
        names = [p.name for p in exp_dirs]
        if args.start_from not in names:
            raise ValueError(
                f"--start_from={args.start_from!r} not found. " + f"Available experiments: {names}"
            )

        start_idx = names.index(args.start_from)
        exp_dirs = exp_dirs[start_idx:]

    if args.reverse:
        exp_dirs = list(reversed(exp_dirs))

    if args.max_runs is not None:
        if args.max_runs < 1:
            raise ValueError("--max_runs must be >= 1")
        exp_dirs = exp_dirs[: args.max_runs]

    if not exp_dirs:
        raise RuntimeError(f"No runnable experiment directories found under: {config_root}")

    print(f"[CONFIG_ROOT] {config_root}")
    print(f"[CUDA_VISIBLE_DEVICES] {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print("[EXPERIMENTS]")
    for idx, exp_dir in enumerate(exp_dirs, start=1):
        print(f"  {idx:02d}. {exp_dir.name}")

    if args.dry_run:
        print("[DRY RUN] No training launched.")
        return

    failed: list[tuple[str, BaseException]] = []

    for exp_dir in exp_dirs:
        try:
            run_one_experiment(exp_dir)
        except BaseException as exc:
            failed.append((exp_dir.name, exc))
            print("\n" + "!" * 100)
            print(f"[FAILED] {exp_dir.name}")
            traceback.print_exc()
            print("!" * 100 + "\n")

            if not args.continue_on_error:
                raise
        finally:
            cleanup_cuda()

    if failed:
        print("[SUMMARY] Some experiments failed:")
        for name, exc in failed:
            print(f"  - {name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print("[SUMMARY] All experiments finished successfully.")


if __name__ == "__main__":
    main()
