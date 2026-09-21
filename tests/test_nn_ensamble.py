#Test 9



"""
test_ensemble_model.py
EnsembleModel tests.

Run:
    python tests/test_ensemble_model.py
"""

import sys
import os
import math
import time
import torch

from models.pinn.base_pinn import BasePINN
from models.pinn.residual_pinn import ResidualPINN

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing.feature_pipeline import FeatureBatch
from core.base_model import ModelOutput
from models.bayesian.bayesian_nn import BayesianNN
from models.ensemble.ensemble_model import EnsembleModel

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


def make_pinn(hidden_size: int = 16) -> BasePINN:
    return BasePINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=hidden_size)


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

def test_construction():
    section("EnsembleModel construction")

    p1 = make_pinn()
    p2 = make_pinn()

    # Default mean aggregation
    ensemble = EnsembleModel(models=[p1, p2])
    check("default aggregation is mean", ensemble.aggregation == "mean")
    check("models registered", len(ensemble.models) == 2)

    # weighted_mean
    ensemble_w = EnsembleModel(models=[p1, p2], aggregation="weighted_mean",
                               weights=[0.3, 0.7])
    check("weighted_mean aggregation", ensemble_w.aggregation == "weighted_mean")
    check("weights sum to 1.0",
          abs(ensemble_w.weights.sum().item() - 1.0) < 1e-6)

    # Empty models raises
    try:
        EnsembleModel(models=[])
        check("empty models raises ValueError", False)
    except ValueError:
        check("empty models raises ValueError", True)

    # Wrong aggregation raises
    try:
        EnsembleModel(models=[p1], aggregation="vote")
        check("unsupported aggregation raises ValueError", False)
    except ValueError:
        check("unsupported aggregation raises ValueError", True)

    # Mismatched weights raises
    try:
        EnsembleModel(models=[p1, p2], weights=[0.5, 0.3, 0.2])
        check("mismatched weights raises ValueError", False)
    except ValueError:
        check("mismatched weights raises ValueError", True)

    # Single model
    ensemble_1 = EnsembleModel(models=[p1])
    check("single model ensemble works", len(ensemble_1.models) == 1)

    # summary
    summary = ensemble_1.summary()
    check("summary mentions EnsembleModel", "EnsembleModel" in summary)
    check("summary mentions aggregation", "aggregation" in summary)


def test_forward_mean():
    section("EnsembleModel forward — mean aggregation")

    batch    = make_batch()
    models   = [make_pinn() for _ in range(3)]
    ensemble = EnsembleModel(models=models, aggregation="mean")

    output = ensemble.forward(batch)
    check("returns ModelOutput", isinstance(output, ModelOutput))
    check("pred shape (N, 1)", output.pred.shape == (100, 1))
    check("uncertainty not None", output.uncertainty is not None)
    check("uncertainty shape (N, 1)", output.uncertainty.shape == (100, 1))
    check("uncertainty >= 0", (output.uncertainty >= 0).all())
    check("pred is finite", torch.isfinite(output.pred).all())

    # Mean prediction is between min and max of individual preds
    individual = torch.stack([m.forward(batch).pred for m in models], dim=0)
    ind_min    = individual.min(dim=0).values
    ind_max    = individual.max(dim=0).values
    check("mean pred within individual range",
          (output.pred >= ind_min - 1e-5).all() and
          (output.pred <= ind_max + 1e-5).all())


def test_forward_weighted_mean():
    section("EnsembleModel forward — weighted_mean aggregation")

    batch  = make_batch()
    p1     = make_pinn()
    p2     = make_pinn()

    # Equal weights → same as mean
    ensemble_eq = EnsembleModel(models=[p1, p2], aggregation="weighted_mean",
                                weights=[0.5, 0.5])
    ensemble_mn = EnsembleModel(models=[p1, p2], aggregation="mean")

    out_eq = ensemble_eq.forward(batch).pred
    out_mn = ensemble_mn.forward(batch).pred
    check("equal weights == mean", torch.allclose(out_eq, out_mn, atol=1e-6))

    # Different weights → different result
    ensemble_w = EnsembleModel(models=[p1, p2], aggregation="weighted_mean",
                               weights=[0.9, 0.1])
    out_w = ensemble_w.forward(batch).pred
    check("unequal weights differ from mean",
          not torch.allclose(out_w, out_mn, atol=1e-6))

    # Extreme weights → close to dominant model
    out_p1 = p1.forward(batch).pred
    ensemble_dom = EnsembleModel(models=[p1, p2], aggregation="weighted_mean",
                                 weights=[0.9999, 0.0001])
    out_dom = ensemble_dom.forward(batch).pred
    check("near-zero weight ≈ dominant model",
          torch.allclose(out_dom, out_p1, atol=1e-3))


def test_add_model():
    section("EnsembleModel.add_model()")

    batch    = make_batch()
    p1       = make_pinn()
    p2       = make_pinn()
    ensemble = EnsembleModel(models=[p1, p2])

    check("initial size 2", len(ensemble.models) == 2)

    p3 = make_pinn()
    ensemble.add_model(p3, weight=1.0)
    check("size after add_model", len(ensemble.models) == 3)
    check("weights still sum to 1.0",
          abs(ensemble.weights.sum().item() - 1.0) < 1e-6)

    # Still produces valid output
    output = ensemble.forward(batch)
    check("forward after add_model works", output.pred.shape == (100, 1))


def test_mixed_model_types():
    section("EnsembleModel with mixed model types")

    batch   = make_batch()
    pinn    = make_pinn()
    rpinn   = ResidualPINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16)
    bayes   = BayesianNN(input_dim=2, output_dim=1, hidden_size=16, n_samples=5)

    ensemble = EnsembleModel(models=[pinn, rpinn, bayes], aggregation="mean")
    output   = ensemble.forward(batch)

    check("mixed types forward works", isinstance(output, ModelOutput))
    check("pred shape correct", output.pred.shape == (100, 1))
    check("uncertainty not None", output.uncertainty is not None)


def test_uncertainty_increases_with_disagreement():
    section("EnsembleModel uncertainty reflects model disagreement")

    batch = make_batch()

    # Identical models → low uncertainty (all predict same thing)
    p_base   = make_pinn()
    ensemble_same = EnsembleModel(models=[p_base, p_base, p_base])
    out_same = ensemble_same.forward(batch)
    low_unc  = out_same.uncertainty.mean().item()

    # Different random models → higher uncertainty
    models_diff   = [make_pinn() for _ in range(3)]
    ensemble_diff = EnsembleModel(models=models_diff)
    out_diff      = ensemble_diff.forward(batch)
    high_unc      = out_diff.uncertainty.mean().item()

    check("identical models → zero uncertainty", low_unc < 1e-6)
    check("different models → positive uncertainty", high_unc > 0)


def test_parameter_tracking():
    section("EnsembleModel parameter tracking (ModuleList)")

    p1       = make_pinn(hidden_size=16)
    p2       = make_pinn(hidden_size=32)
    ensemble = EnsembleModel(models=[p1, p2])

    total_params = sum(p.numel() for p in ensemble.parameters())
    p1_params    = sum(p.numel() for p in p1.parameters())
    p2_params    = sum(p.numel() for p in p2.parameters())

    check("ensemble tracks all parameters",
          total_params == p1_params + p2_params)


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("AIPlasma — EnsembleModel Test")
    print("=" * 60)

    start = time.perf_counter()

    test_construction()
    test_forward_mean()
    test_forward_weighted_mean()
    test_add_model()
    test_mixed_model_types()
    test_uncertainty_increases_with_disagreement()
    test_parameter_tracking()

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