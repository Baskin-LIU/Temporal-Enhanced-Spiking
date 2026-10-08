from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/Users/bingkung/Projects/TeS")
BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "yinyang_best.json"
RESULT_ROOT = PROJECT_ROOT / "results" / "yinyang_h5_e20_tuning_2026-10-08"
REUSED_BASELINE = (
    PROJECT_ROOT
    / "results"
    / "yinyang_learn_omega_h5_e40_seed2_2026-10-07"
    / "metrics.json"
)
SCREEN_REPLICATE = 2
FOLLOWUP_REPLICATES = (1, 3)
SCREEN_THRESHOLD = 0.90
MAX_WORKERS = 4


BASELINE: dict[str, Any] = {
    "omega_lr_scale": 0.01,
    "omega_min": 0.10,
    "omega_max": 0.35,
    "margin_ms": 2.0,
    "readout_mean": 1.50,
    "bump_amount": 0.05,
}


CANDIDATES: tuple[dict[str, Any], ...] = (
    {"name": "baseline", "family": "baseline", **BASELINE},
    {
        "name": "omega_lr_0p005",
        "family": "omega_lr_scale",
        **BASELINE,
        "omega_lr_scale": 0.005,
    },
    {
        "name": "omega_lr_0p02",
        "family": "omega_lr_scale",
        **BASELINE,
        "omega_lr_scale": 0.02,
    },
    {
        "name": "omega_range_0p08_0p40",
        "family": "omega_range",
        **BASELINE,
        "omega_min": 0.08,
        "omega_max": 0.40,
    },
    {
        "name": "omega_range_0p12_0p32",
        "family": "omega_range",
        **BASELINE,
        "omega_min": 0.12,
        "omega_max": 0.32,
    },
    {
        "name": "margin_1p0",
        "family": "margin_ms",
        **BASELINE,
        "margin_ms": 1.0,
    },
    {
        "name": "margin_3p0",
        "family": "margin_ms",
        **BASELINE,
        "margin_ms": 3.0,
    },
    {
        "name": "readout_mean_1p25",
        "family": "readout_mean",
        **BASELINE,
        "readout_mean": 1.25,
    },
    {
        "name": "readout_mean_1p75",
        "family": "readout_mean",
        **BASELINE,
        "readout_mean": 1.75,
    },
    {
        "name": "bump_amount_0p025",
        "family": "bump_amount",
        **BASELINE,
        "bump_amount": 0.025,
    },
    {
        "name": "bump_amount_0p075",
        "family": "bump_amount",
        **BASELINE,
        "bump_amount": 0.075,
    },
)


FAMILY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "omega_lr_scale": ("omega_lr_0p005", "baseline", "omega_lr_0p02"),
    "omega_range": (
        "omega_range_0p08_0p40",
        "baseline",
        "omega_range_0p12_0p32",
    ),
    "margin_ms": ("margin_1p0", "baseline", "margin_3p0"),
    "readout_mean": (
        "readout_mean_1p25",
        "baseline",
        "readout_mean_1p75",
    ),
    "bump_amount": (
        "bump_amount_0p025",
        "baseline",
        "bump_amount_0p075",
    ),
}


def _score_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    first_twenty = [
        epoch for epoch in payload["epochs"] if int(epoch["epoch"]) <= 20
    ]
    if len(first_twenty) != 20:
        raise ValueError(f"expected 20 epochs in {path}, found {len(first_twenty)}")
    best = max(
        first_twenty,
        key=lambda epoch: (
            float(epoch["validation"]["accuracy"]),
            -float(epoch["validation"]["mean_loss"]),
        ),
    )
    return {
        "best_validation_accuracy": float(best["validation"]["accuracy"]),
        "best_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["validation"]["mean_loss"]),
        "runtime_seconds": float(payload["runtime_seconds"]),
        "metrics_path": str(path),
    }


def _run_one(candidate: dict[str, Any], replicate: int) -> dict[str, Any]:
    os.chdir(PROJECT_ROOT)
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    from Code.yinyang_config import YinYangConfig
    from Code.yinyang_experiment import run_yinyang_training
    from Code.yinyang_reporting import write_training_artifacts

    run_dir = RESULT_ROOT / "runs" / candidate["name"] / f"seed{replicate}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        result = _score_metrics(metrics_path)
        result.update(candidate=candidate["name"], replicate=replicate, reused=True)
        return result

    values = json.loads(BASE_CONFIG_PATH.read_text())
    seed_offset = replicate - 1
    values["model"]["n_hidden"] = 5
    values["initialization"]["hidden_seed"] += seed_offset
    values["initialization"]["readout_seed"] += seed_offset
    values["initialization"]["heterogeneity_seed"] += seed_offset
    values["initialization"]["readout_mean"] = candidate["readout_mean"]
    values["loss"]["margin_ms"] = candidate["margin_ms"]
    values["optimization"]["epochs"] = 20
    values["optimization"]["cosine_t_max"] = 40
    values["optimization"]["adam"]["learning_rate"] = 0.01
    values["optimization"]["shuffle_seed"] += seed_offset
    values["restart"]["silence_fraction"] = 0.6
    values["restart"]["seed"] += seed_offset
    values["artifacts"]["save_rasters"] = False
    values["artifacts"]["checkpoint_interval"] = 0
    values["artifacts"]["plot_interval"] = 0
    config = YinYangConfig.from_dict(values)

    original = YinYangConfig.to_training_config

    def with_candidate_omega_range(self, **kwargs):
        training = original(self, **kwargs)
        return replace(
            training,
            omega_min=float(candidate["omega_min"]),
            omega_max=float(candidate["omega_max"]),
        )

    YinYangConfig.to_training_config = with_candidate_omega_range
    try:
        artifacts = run_yinyang_training(
            config,
            run_dir,
            evaluate_test=False,
            learn_omega=True,
            omega_learning_rate_scale=float(candidate["omega_lr_scale"]),
            in_epoch_bump_interval_batches=10,
            in_epoch_bump_silence_fraction=0.5,
            in_epoch_bump_amount=float(candidate["bump_amount"]),
        )
        write_training_artifacts(run_dir, artifacts)
    finally:
        YinYangConfig.to_training_config = original

    result = _score_metrics(metrics_path)
    result.update(candidate=candidate["name"], replicate=replicate, reused=False)
    return result


def _aggregate(
    results: dict[str, list[dict[str, Any]]],
    screened_out: set[str],
) -> dict[str, Any]:
    by_name = {candidate["name"]: candidate for candidate in CANDIDATES}
    summaries: dict[str, dict[str, Any]] = {}
    for name, candidate in by_name.items():
        runs = sorted(results.get(name, []), key=lambda item: item["replicate"])
        scores = [run["best_validation_accuracy"] for run in runs]
        complete = len(runs) == 3
        summaries[name] = {
            "family": candidate["family"],
            "parameters": {
                key: candidate[key]
                for key in (
                    "omega_lr_scale",
                    "omega_min",
                    "omega_max",
                    "margin_ms",
                    "readout_mean",
                    "bump_amount",
                )
            },
            "status": "screened_out" if name in screened_out else (
                "complete" if complete else "incomplete"
            ),
            "runs": runs,
            "screen_peak": next(
                (
                    run["best_validation_accuracy"]
                    for run in runs
                    if run["replicate"] == SCREEN_REPLICATE
                ),
                None,
            ),
            "mean_best_validation_accuracy": (
                sum(scores) / len(scores) if complete else None
            ),
            "sample_standard_deviation": (
                math.sqrt(
                    sum((score - sum(scores) / len(scores)) ** 2 for score in scores)
                    / (len(scores) - 1)
                )
                if complete and len(scores) > 1
                else None
            ),
        }

    best_by_parameter: dict[str, dict[str, Any]] = {}
    for family, names in FAMILY_CANDIDATES.items():
        completed = [
            summaries[name]
            for name in names
            if summaries[name]["status"] == "complete"
        ]
        winner = max(
            completed,
            key=lambda item: item["mean_best_validation_accuracy"],
        )
        winner_name = next(name for name in names if summaries[name] is winner)
        best_by_parameter[family] = {
            "candidate": winner_name,
            "mean_best_validation_accuracy": winner[
                "mean_best_validation_accuracy"
            ],
            "sample_standard_deviation": winner["sample_standard_deviation"],
            "parameters": winner["parameters"],
        }
    return {
        "selection_metric": (
            "mean across three seeds of each run's maximum validation accuracy "
            "within epochs 1-20"
        ),
        "screening_rule": (
            "run seed 2 first; do not run seeds 1 and 3 when seed-2 maximum "
            "validation accuracy is below 0.90"
        ),
        "fixed": {
            "hidden_neurons": 5,
            "epochs": 20,
            "weight_learning_rate": 0.01,
            "cosine_t_max": 40,
            "data_split_seeds": [1001, 1002, 1003],
            "test_evaluations": 0,
        },
        "replicate_seed_offsets": {
            "seed1": 0,
            "seed2": 1,
            "seed3": 2,
        },
        "candidates": summaries,
        "best_by_parameter": best_by_parameter,
    }


def _write_csv(summary: dict[str, Any]) -> None:
    destination = RESULT_ROOT / "summary.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "candidate",
                "family",
                "status",
                "omega_lr_scale",
                "omega_min",
                "omega_max",
                "margin_ms",
                "readout_mean",
                "bump_amount",
                "seed1_best",
                "seed2_best",
                "seed3_best",
                "mean_best_validation_accuracy",
                "sample_standard_deviation",
            )
        )
        for name, candidate in summary["candidates"].items():
            by_seed = {
                run["replicate"]: run["best_validation_accuracy"]
                for run in candidate["runs"]
            }
            parameters = candidate["parameters"]
            writer.writerow(
                (
                    name,
                    candidate["family"],
                    candidate["status"],
                    parameters["omega_lr_scale"],
                    parameters["omega_min"],
                    parameters["omega_max"],
                    parameters["margin_ms"],
                    parameters["readout_mean"],
                    parameters["bump_amount"],
                    by_seed.get(1),
                    by_seed.get(2),
                    by_seed.get(3),
                    candidate["mean_best_validation_accuracy"],
                    candidate["sample_standard_deviation"],
                )
            )


def _write_plot(summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "omega_lr_scale": ("Omega LR scale", ("0.005", "0.01", "0.02")),
        "omega_range": (
            "Omega clamp range",
            ("[.08,.40]", "[.10,.35]", "[.12,.32]"),
        ),
        "margin_ms": ("Loss margin (ms)", ("1", "2", "3")),
        "readout_mean": ("Readout init mean", ("1.25", "1.50", "1.75")),
        "bump_amount": ("Rescue bump", (".025", ".050", ".075")),
    }
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, (family, names) in zip(axes.flat, FAMILY_CANDIDATES.items()):
        title, tick_labels = labels[family]
        for index, name in enumerate(names):
            candidate = summary["candidates"][name]
            if candidate["status"] == "complete":
                axis.errorbar(
                    index,
                    candidate["mean_best_validation_accuracy"],
                    yerr=candidate["sample_standard_deviation"],
                    fmt="o",
                    color="#1f77b4",
                    capsize=5,
                    markersize=8,
                )
                axis.text(
                    index,
                    candidate["mean_best_validation_accuracy"] + 0.008,
                    f"{100 * candidate['mean_best_validation_accuracy']:.2f}%",
                    ha="center",
                    fontsize=9,
                )
            else:
                axis.plot(
                    index,
                    candidate["screen_peak"],
                    "x",
                    color="#d62728",
                    markersize=10,
                    markeredgewidth=2,
                )
                axis.text(
                    index,
                    candidate["screen_peak"] + 0.008,
                    f"screen {100 * candidate['screen_peak']:.1f}%",
                    ha="center",
                    fontsize=8,
                    color="#d62728",
                )
        axis.axhline(SCREEN_THRESHOLD, color="0.5", linestyle="--", linewidth=1)
        axis.set_xticks(range(3), tick_labels)
        axis.set_ylim(0.80, 1.00)
        axis.set_title(title)
        axis.set_ylabel("mean of 3 peak validation accuracies")
        axis.grid(axis="y", alpha=0.25)

    note = axes.flat[-1]
    note.axis("off")
    note.text(
        0.02,
        0.95,
        "Selection rule\n\n"
        "• Peak validation accuracy from epochs 1–20\n"
        "• Mean ± sample SD across seeds 1, 2, 3\n"
        "• Red ×: seed-2 screen < 90%; no repeats\n"
        "• Test split never evaluated\n\n"
        "All panels are one-factor-at-a-time; the center\n"
        "point is the same shared baseline.",
        va="top",
        fontsize=12,
    )
    figure.suptitle("5-hidden-neuron, 20-epoch parameter sweep", fontsize=16)
    figure.savefig(RESULT_ROOT / "tuning_summary.png", dpi=180)
    plt.close(figure)


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict[str, Any]]] = {
        candidate["name"]: [] for candidate in CANDIDATES
    }

    reused = _score_metrics(REUSED_BASELINE)
    reused.update(candidate="baseline", replicate=2, reused=True)
    results["baseline"].append(reused)

    screening = [candidate for candidate in CANDIDATES if candidate["name"] != "baseline"]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_run_one, candidate, SCREEN_REPLICATE): candidate["name"]
            for candidate in screening
        }
        for future in as_completed(futures):
            results[futures[future]].append(future.result())

    screened_out = {
        name
        for name, runs in results.items()
        if next(
            run["best_validation_accuracy"]
            for run in runs
            if run["replicate"] == SCREEN_REPLICATE
        )
        < SCREEN_THRESHOLD
    }

    followups = [
        (candidate, replicate)
        for candidate in CANDIDATES
        if candidate["name"] not in screened_out
        for replicate in FOLLOWUP_REPLICATES
    ]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_run_one, candidate, replicate): (
                candidate["name"],
                replicate,
            )
            for candidate, replicate in followups
        }
        for future in as_completed(futures):
            name, _ = futures[future]
            results[name].append(future.result())

    summary = _aggregate(results, screened_out)
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(summary)
    _write_plot(summary)
    print(RESULT_ROOT / "summary.json")


if __name__ == "__main__":
    main()
