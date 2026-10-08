import json
from pathlib import Path

from Code.tes_training import load_checkpoint
from Code.yinyang_config import (
    DEFAULT_CONFIG_PATH,
    load_yinyang_config,
)
from Code.yinyang_experiment import initialize_yinyang_model


ROOT = Path(__file__).resolve().parents[2]
BEST = ROOT / "results" / "yinyang_best"


def test_retained_best_run_has_resolvable_compact_artifacts():
    canonical = load_yinyang_config(DEFAULT_CONFIG_PATH)
    retained = load_yinyang_config(BEST / "config.json")
    metrics = json.loads((BEST / "metrics.json").read_text())

    assert retained == canonical
    assert metrics["configuration"] == json.loads(
        json.dumps(canonical.to_dict())
    )
    assert metrics["test"]["accuracy"] == 0.935
    assert metrics["selected_best_checkpoint_epoch"] == 14
    assert len(metrics["epochs"]) == 40
    for relative in metrics["artifacts"].values():
        assert not Path(relative).is_absolute()
        assert (BEST / relative).is_file()


def test_retained_checkpoints_load_with_the_current_scientific_config():
    config = load_yinyang_config(BEST / "config.json")
    expected = config.to_training_config()
    for filename, epoch in (("best_validation.pt", 14), ("final_tested.pt", 40)):
        model = initialize_yinyang_model(config)
        state = load_checkpoint(
            BEST / "checkpoints" / filename,
            model,
            expected_config=expected,
            expected_run_config=config.to_dict(),
        )
        assert state.epoch == epoch


def test_experiment_summary_preserves_all_five_completed_runs():
    summary = json.loads(
        (ROOT / "results" / "yinyang_experiment_summary.json").read_text()
    )

    assert len(summary["experiments"]) == 5
    assert summary["canonical_experiment"] == (
        "squared_hinge_margin2_restart_offset0.1"
    )
    canonical = next(
        experiment
        for experiment in summary["experiments"]
        if experiment["name"] == summary["canonical_experiment"]
    )
    assert canonical["test_accuracy"] == 0.935
