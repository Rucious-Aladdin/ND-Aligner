from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict

from tts.benchmark.timit.benchmarker import TIMITBenchMarker
from tts.models.ndaligner import NDAligner


class EvalCSVRowInstance(TypedDict):
    # Experiment identity
    variant: str
    epoch: int
    step: int

    # Evaluation setting
    max_test_samples: int | None

    # TIMIT word-boundary metrics
    word_boundary_error: float
    p_word_10ms: float
    p_word_25ms: float
    p_word_50ms: float
    p_word_100ms: float

    # Additional aggregate metrics
    posterior_entropy: float


class NDAlignerTimitLogger:
    def __init__(
        self,
        base_dir: str,
        experiment_name: str,
        exp_variant: str,
        timit_benchmarker: TIMITBenchMarker,
        max_test_samples: int | None = None,
        save_align_figure: bool = False,
        align_figure_dir: str | Path | None = None,
    ) -> None:
        self.dir = Path(base_dir) / experiment_name
        self.dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.dir / "eval_metrics.csv"
        self._init_csv()

        self.exp_variant = exp_variant
        self.timit_benchmarker = timit_benchmarker

        self.max_test_samples = max_test_samples
        self.save_align_figure = save_align_figure

        if align_figure_dir is None:
            self.align_figure_dir = self.dir / "timit_align_figures"
        else:
            self.align_figure_dir = Path(align_figure_dir)

    def __call__(
        self,
        *,
        epoch: int,
        step: int,
        aligner: NDAligner,
        is_test: bool = False,
    ) -> EvalCSVRowInstance:
        metrics = self.timit_benchmarker.__call__(
            aligner=aligner,
            max_test_samples=self.max_test_samples,
            save_align_figure=self.save_align_figure,
            align_figure_dir=self.align_figure_dir,
            is_test=is_test,
        )

        row: EvalCSVRowInstance = {
            "variant": self.exp_variant,
            "epoch": int(epoch),
            "step": int(step),
            "max_test_samples": self.max_test_samples,
            "word_boundary_error": float(metrics.word_boundary_error),
            "p_word_10ms": float(metrics.p_word_10ms),
            "p_word_25ms": float(metrics.p_word_25ms),
            "p_word_50ms": float(metrics.p_word_50ms),
            "p_word_100ms": float(metrics.p_word_100ms),
            "posterior_entropy": float(metrics.posterior_entropy),
        }

        self._append_csv_row(row)
        return row

    def _init_csv(self) -> None:
        if self.csv_path.exists():
            return

        fieldnames = list(EvalCSVRowInstance.__annotations__.keys())

        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def _append_csv_row(self, row: EvalCSVRowInstance) -> None:
        fieldnames = list(EvalCSVRowInstance.__annotations__.keys())

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
