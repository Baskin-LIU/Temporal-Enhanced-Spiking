from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from Code.yinyang_config import YinYangConfig
from Code.yinyang_experiment import run_yinyang_training
from Code.yinyang_reporting import write_training_artifacts


PROJECT_ROOT = Path("/Users/bingkung/Projects/TeS")
RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "yinyang_h5_e30_omega0p02_max0p32_margin3_seed2_2026-10-08"
)


values = json.loads((PROJECT_ROOT / "configs" / "yinyang_best.json").read_text())
values["model"]["n_hidden"] = 5
values["initialization"]["hidden_seed"] = 414
values["initialization"]["readout_seed"] = 24
values["initialization"]["heterogeneity_seed"] = 91
values["initialization"]["readout_mean"] = 1.5
values["loss"]["margin_ms"] = 3.0
values["optimization"]["epochs"] = 30
values["optimization"]["cosine_t_max"] = 40
values["optimization"]["adam"]["learning_rate"] = 0.01
values["optimization"]["shuffle_seed"] = 2002
values["restart"]["silence_fraction"] = 0.6
values["restart"]["seed"] = 3002
config = YinYangConfig.from_dict(values)

original = YinYangConfig.to_training_config


def with_selected_omega_range(self, **kwargs):
    return replace(
        original(self, **kwargs),
        omega_min=0.10,
        omega_max=0.32,
    )


YinYangConfig.to_training_config = with_selected_omega_range
try:
    artifacts = run_yinyang_training(
        config,
        RUN_DIR,
        evaluate_test=False,
        learn_omega=True,
        omega_learning_rate_scale=0.02,
        in_epoch_bump_interval_batches=10,
        in_epoch_bump_silence_fraction=0.5,
        in_epoch_bump_amount=0.05,
    )
    write_training_artifacts(RUN_DIR, artifacts)
finally:
    YinYangConfig.to_training_config = original

print(RUN_DIR / "metrics.json")
