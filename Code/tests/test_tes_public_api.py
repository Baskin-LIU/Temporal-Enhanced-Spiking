import torch
from pathlib import Path


def test_package_api_runs_discrete_and_continuous_forward_loss_backward():
    from Code import InputSpike, TeSConfig, TeSNeurons

    model = TeSNeurons(
        1,
        1,
        2,
        torch.tensor([[35.0]], dtype=torch.float64),
        torch.tensor([[16.0], [12.0]], dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )
    inputs = [InputSpike(0.0, 0, 0)]
    for mode in ("discrete", "continuous"):
        forward = getattr(model, f"forward_{mode}")(
            inputs, 50.0, early_stop=True
        )
        loss = model.squared_hinge_loss(forward, label=0, margin=2.0)
        assert loss.output_time_grad is not None
        backward = getattr(model, f"backward_{mode}")(
            forward, loss.output_time_grad
        )
        assert backward.grad_W_in.shape == (1, 1)
        assert backward.grad_W_out.shape == (2, 1)
        assert torch.isfinite(backward.grad_W_in).all()
        assert torch.isfinite(backward.grad_W_out).all()


def test_training_contract_and_method_disclose_approximation_and_compact_artifacts():
    root = Path(__file__).resolve().parents[2]
    contract = (root / "docs/tes_code_contracts.md").read_text()
    method = (root / "Latex/tes_method.tex").read_text()

    for text in (contract, method):
        assert "10" in text
        assert "early_stop=False" in text
        assert "cap" in text.lower()
        assert "approx" in text.lower()
    assert "threshold_sensitivity_cap_activations" in contract
    assert "minimum_causal_slope" in contract
    assert "more than 500" in contract
    assert "physical" in contract.lower() and "prediction" in contract.lower()
    assert "best-validation" in contract
    assert "Intermediate" in contract and "default to zero" in contract
    assert "omit" in contract and "batch" in contract
