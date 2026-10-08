"""Command-line entry point for the configurable TeS Yin-Yang experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from Code.yinyang_config import parse_yinyang_args
from Code.yinyang_experiment import run_yinyang_training
from Code.yinyang_reporting import write_training_artifacts


def main(argv: Sequence[str] | None = None) -> Path:
    request = parse_yinyang_args(argv)
    artifacts = run_yinyang_training(
        request.config,
        request.run_dir,
        resume_checkpoint=request.resume_checkpoint,
        evaluate_test=request.evaluate_test,
        learn_omega=request.learn_omega,
        omega_learning_rate_scale=request.omega_learning_rate_scale,
        in_epoch_bump_interval_batches=(
            request.in_epoch_bump_interval_batches
        ),
        in_epoch_bump_silence_fraction=(
            request.in_epoch_bump_silence_fraction
        ),
        in_epoch_bump_amount=request.in_epoch_bump_amount,
    )
    write_training_artifacts(request.run_dir, artifacts)
    report = request.run_dir / "metrics.json"
    print(report)
    return report


if __name__ == "__main__":
    main()
