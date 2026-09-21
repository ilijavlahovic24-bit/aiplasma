#Test 8

"""
test_bayesian_nn.py
BayesianNN and UncertaintyMetrics tests.

Run:
    python tests/test_bayesian_nn.py
"""

import sys
import os
import math
import time
import torch

from models.bayesian.uncertenty import UncertaintyMetrics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing.feature_pipeline import FeatureBatch
from core.base_model import ModelOutput
from models.bayesian.bayesian_nn import BayesianNN


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n[{title}]")


def make_batch(n: int = 100) -> FeatureBatch:
    x      = torch.rand(n)
    t      = torch.rand(n)
    coords = torch.stack([x, t], dim=1)
    u      = torch.sin(math.pi * x) * torch.exp(torch.tensor(-0.01 * math.pi ** 2) * t)
    fields = u.unsqueeze(1)
    bm     = (x < 0.05) | (x > 0.95)
    return FeatureBatch(
        coords=coords, fields=fields,
        boundary_mask=bm, collocation_mask=~bm,
        physics_params={"alpha": 0.01},
        fidelity_level=0, fidelity_weight=1.0,
    )


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

def test_bayesian_nn_construction():
    section("BayesianNN construction")

    model = BayesianNN(input_dim=2, output_dim=1)
    check("default construction", model is not None)
    check("n_samples default", model.n_samples == 50)
    check("dropout_rate default", model.dropout_rate == 0.1)

    # Invalid dropout_rate
    try:
        BayesianNN(input_dim=2, output_dim=1, dropout_rate=0.0)
        check("dropout_rate=0.0 raises ValueError", False)
    except ValueError:
        check("dropout_rate=0.0 raises ValueError", True)

    try:
        BayesianNN(input_dim=2, output_dim=1, dropout_rate=1.0)
        check("dropout_rate=1.0 raises ValueError", False)
    except ValueError:
        check("dropout_rate=1.0 raises ValueError", True)

    # Invalid activation
    try:
        BayesianNN(input_dim=2, output_dim=1, activation="sigmoid")
        check("invalid activation raises ValueError", False)
    except ValueError:
        check("invalid activation raises ValueError", True)

    # summary
    summary = model.summary()
    check("summary mentions BayesianNN", "BayesianNN" in summary)
    check("summary mentions dropout", "dropout" in summary)
    check("summary mentions n_samples", "n_samples" in summary)


def test_forward_training_mode():
    section("BayesianNN forward — training mode")

    batch = make_batch()
    model = BayesianNN(input_dim=2, output_dim=1, hidden_size=16, n_samples=5)
    model.train()

    output = model.forward(batch)
    check("returns ModelOutput", isinstance(output, ModelOutput))
    check("pred shape (N, 1)", output.pred.shape == (100, 1))
    check("uncertainty is None in training", output.uncertainty is None)

    # Multiple training forward passes give different results (dropout active)
    out1 = model.forward(batch).pred
    out2 = model.forward(batch).pred
    check("training forward passes differ (dropout)",
          not torch.allclose(out1, out2, atol=1e-6))


def test_forward_eval_mode():
    section("BayesianNN forward — eval mode")

    batch = make_batch()
    model = BayesianNN(input_dim=2, output_dim=1, hidden_size=16, n_samples=10)
    model.eval()

    output = model.forward(batch)
    check("returns ModelOutput", isinstance(output, ModelOutput))
    check("pred shape (N, 1)", output.pred.shape == (100, 1))
    check("uncertainty not None in eval", output.uncertainty is not None)
    check("uncertainty shape (N, 1)", output.uncertainty.shape == (100, 1))
    check("uncertainty >= 0", (output.uncertainty >= 0).all())


def test_sample_predictions():
    section("BayesianNN.sample_predictions()")

    batch = make_batch()
    model = BayesianNN(input_dim=2, output_dim=1, hidden_size=16, n_samples=20)

    mean, std = model.sample_predictions(batch)
    check("mean shape (N, 1)", mean.shape == (100, 1))
    check("std shape (N, 1)", std.shape == (100, 1))
    check("std >= 0", (std >= 0).all())
    check("mean is finite", torch.isfinite(mean).all())
    check("std is finite", torch.isfinite(std).all())

    # More samples → smoother uncertainty estimate
    _, std5  = BayesianNN(input_dim=2, output_dim=1, n_samples=5).sample_predictions(batch)
    _, std50 = BayesianNN(input_dim=2, output_dim=1, n_samples=50).sample_predictions(batch)
    check("std is non-zero for n_samples=50", std50.mean().item() > 0)

    # uncertainty() convenience method
    model2 = BayesianNN(input_dim=2, output_dim=1, hidden_size=16, n_samples=10)
    unc    = model2.uncertainty(batch)
    check("uncertainty() shape (N, 1)", unc.shape == (100, 1))
    check("uncertainty() >= 0", (unc >= 0).all())


def test_dropout_placement():
    section("BayesianNN dropout placement (Option B)")

    import torch.nn as nn
    model = BayesianNN(input_dim=2, output_dim=1, hidden_size=32, hidden_layers=3)

    # Count dropout layers
    dropout_layers = [m for m in model.net.modules() if isinstance(m, nn.Dropout)]
    linear_layers  = [m for m in model.net.modules() if isinstance(m, nn.Linear)]

    check("has dropout layers", len(dropout_layers) > 0)
    check("dropout count = hidden_layers", len(dropout_layers) == model.hidden_layers)
    # input + hidden_layers + output = hidden_layers + 2 linear layers
    check("linear count = hidden_layers + 2",
          len(linear_layers) == model.hidden_layers + 2)


def test_uncertainty_metrics():
    section("UncertaintyMetrics")

    # Create synthetic MC samples
    n_samples  = 20
    n_points   = 50
    output_dim = 1

    samples = torch.randn(n_samples, n_points, output_dim)
    targets = torch.zeros(n_points, output_dim)

    # epistemic
    epistemic = UncertaintyMetrics.epistemic(samples)
    check("epistemic shape", epistemic.shape == (n_points, output_dim))
    check("epistemic >= 0", (epistemic >= 0).all())

    # aleatoric
    aleatoric = UncertaintyMetrics.aleatoric(samples, targets)
    check("aleatoric shape", aleatoric.shape == (n_points, output_dim))
    check("aleatoric >= 0", (aleatoric >= 0).all())

    # calibration_error
    pred_mean = samples.mean(dim=0)
    pred_std  = samples.std(dim=0).clamp(min=1e-8)
    ece       = UncertaintyMetrics.calibration_error(pred_mean, pred_std, targets)
    check("calibration_error returns float", isinstance(ece, float))
    check("calibration_error in [0, 1]", 0.0 <= ece <= 1.0)

    # Perfect predictions → low uncertainty → high ECE (miscalibrated)
    perfect_mean = targets.clone()
    perfect_std  = torch.ones_like(targets) * 0.001
    ece_perfect  = UncertaintyMetrics.calibration_error(perfect_mean, perfect_std, targets)
    check("perfect predictions computable", isinstance(ece_perfect, float))


def test_backprop_through_bayesian():
    section("BayesianNN backpropagation")

    batch = make_batch()
    model = BayesianNN(input_dim=2, output_dim=1, hidden_size=16, n_samples=5)
    model.train()

    output = model.forward(batch)
    loss   = ((output.pred - batch.fields) ** 2).mean()

    try:
        loss.backward()
        has_grads = all(
            p.grad is not None
            for p in model.parameters()
            if p.requires_grad
        )
        check("backprop works", True)
        check("all params have gradients", has_grads)
    except Exception as e:
        check("backprop works", False, str(e))
        check("all params have gradients", False)


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("AIPlasma — BayesianNN Test")
    print("=" * 60)

    start = time.perf_counter()

    test_bayesian_nn_construction()
    test_forward_training_mode()
    test_forward_eval_mode()
    test_sample_predictions()
    test_dropout_placement()
    test_uncertainty_metrics()
    test_backprop_through_bayesian()

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    print(f"REZULTATI: {PASSED} passed, {FAILED} failed | {elapsed:.2f}s")
    if FAILED == 0:
        print("TEST PROSAO ✓")
    else:
        print(f"TEST NIJE PROSAO ✗ — {FAILED} test(ova) palo")
    print("=" * 60)


if __name__ == "__main__":
    run()
