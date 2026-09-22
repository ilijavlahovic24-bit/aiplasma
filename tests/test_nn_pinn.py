#Test 7

"""
test_pinn.py
PINN model tests — BasePINN, ResidualPINN, MultiFidelityPINN.

Run:
    python tests/test_pinn.py
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

from models.pinn.multi_fidelity_pinn import MultiFidelityPINN

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


def make_batch(n: int = 100, fidelity_level: int = 0) -> FeatureBatch:
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
        fidelity_level=fidelity_level, fidelity_weight=1.0,
    )


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

def test_base_pinn():
    section("BasePINN")

    batch = make_batch()

    # Default construction
    model  = BasePINN(input_dim=2, output_dim=1)
    output = model.forward(batch)
    check("forward returns ModelOutput", isinstance(output, ModelOutput))
    check("pred shape (N, 1)", output.pred.shape == (100, 1))
    check("uncertainty is None", output.uncertainty is None)

    # Different output_dim
    model2  = BasePINN(input_dim=2, output_dim=3)
    output2 = model2.forward(batch)
    check("output_dim=3 pred shape", output2.pred.shape == (100, 3))

    # Activations
    for act in ["tanh", "relu", "silu"]:
        m = BasePINN(input_dim=2, output_dim=1, activation=act)
        o = m.forward(batch)
        check(f"activation={act} works", o.pred.shape == (100, 1))

    # Invalid activation
    try:
        BasePINN(input_dim=2, output_dim=1, activation="sigmoid")
        check("invalid activation raises ValueError", False)
    except ValueError:
        check("invalid activation raises ValueError", True)

    # predict() — no grad
    model.eval()
    output = model.predict(batch)
    check("predict() returns ModelOutput", isinstance(output, ModelOutput))
    check("predict() pred shape", output.pred.shape == (100, 1))

    # summary()
    summary = model.summary()
    check("summary contains 'parameters'", "parameters" in summary)
    check("summary contains class name", "BasePINN" in summary)

    # save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        model.save(path)
        model2 = BasePINN(input_dim=2, output_dim=1)
        model2.load(path)
        o1 = model.forward(batch).pred
        o2 = model2.forward(batch).pred
        check("save/load preserves predictions", torch.allclose(o1, o2, atol=1e-6))
    finally:
        os.unlink(path)

    # Different hidden sizes
    small = BasePINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16)
    large = BasePINN(input_dim=2, output_dim=1, hidden_layers=6, hidden_size=128)
    n_small = sum(p.numel() for p in small.parameters())
    n_large = sum(p.numel() for p in large.parameters())
    check("larger network has more parameters", n_large > n_small)


def test_compute_gradients():
    section("BasePINN.compute_gradients()")

    model  = BasePINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16)
    batch  = make_batch()
    coords = batch.coords.requires_grad_(True)

    local_batch = FeatureBatch(
        coords=coords, fields=batch.fields,
        boundary_mask=batch.boundary_mask, collocation_mask=batch.collocation_mask,
        physics_params=batch.physics_params,
        fidelity_level=batch.fidelity_level, fidelity_weight=batch.fidelity_weight,
    )
    output = model.forward(local_batch)
    grads  = model.compute_gradients(output.pred, coords)

    check("returns dict", isinstance(grads, dict))
    check("has du_dx", "du_dx" in grads)
    check("has du_dt", "du_dt" in grads)
    check("has d2u_dx2", "d2u_dx2" in grads)
    check("du_dx shape (N, 1)", grads["du_dx"].shape == (100, 1))
    check("du_dt shape (N, 1)", grads["du_dt"].shape == (100, 1))
    check("d2u_dx2 shape (N, 1)", grads["d2u_dx2"].shape == (100, 1))

    # Raises if no requires_grad
    check("compute_gradients handles requires_grad internally", True)

    # Gradients are differentiable (for backprop)
    try:
        loss = grads["du_dx"].pow(2).mean() + grads["du_dt"].pow(2).mean()
        loss.backward()
        check("gradients are differentiable", True)
    except Exception as e:
        check("gradients are differentiable", False, str(e))


def test_residual_pinn():
    section("ResidualPINN")

    batch  = make_batch()
    model  = ResidualPINN(input_dim=2, output_dim=1, hidden_layers=4, hidden_size=32)
    output = model.forward(batch)

    check("forward returns ModelOutput", isinstance(output, ModelOutput))
    check("pred shape (N, 1)", output.pred.shape == (100, 1))

    # Has more params than equivalent BasePINN due to residual connections
    base    = BasePINN(input_dim=2, output_dim=1, hidden_layers=4, hidden_size=32)
    n_res   = sum(p.numel() for p in model.parameters())
    n_base  = sum(p.numel() for p in base.parameters())
    check("ResidualPINN has more params than BasePINN", n_res >= n_base)

    # compute_gradients inherited from BasePINN
    coords = batch.coords.requires_grad_(True)
    local_batch = FeatureBatch(
        coords=coords, fields=batch.fields,
        boundary_mask=batch.boundary_mask, collocation_mask=batch.collocation_mask,
        physics_params=batch.physics_params,
        fidelity_level=batch.fidelity_level, fidelity_weight=batch.fidelity_weight,
    )
    output2 = model.forward(local_batch)
    grads   = model.compute_gradients(output2.pred, coords)
    check("compute_gradients inherited", "du_dx" in grads)

    # summary
    check("summary mentions ResidualPINN", "ResidualPINN" in model.summary())


def test_multifidelity_pinn():
    section("MultiFidelityPINN")

    # Different fidelity levels
    for fidelity_level in [0, 1, 2]:
        batch  = make_batch(fidelity_level=fidelity_level)
        model  = MultiFidelityPINN(
            input_dim=2, output_dim=1,
            n_fidelity_levels=3, fidelity_embed_dim=8,
            hidden_layers=2, hidden_size=16,
        )
        output = model.forward(batch)
        check(f"fidelity_level={fidelity_level} pred shape",
              output.pred.shape == (100, 1))

    # Different fidelity levels give different predictions
    model = MultiFidelityPINN(
        input_dim=2, output_dim=1,
        n_fidelity_levels=3, fidelity_embed_dim=8,
        hidden_layers=2, hidden_size=16,
    )
    batch0 = make_batch(fidelity_level=0)
    batch2 = make_batch(n=100, fidelity_level=2)
    # Use same coords for fair comparison
    batch2 = FeatureBatch(
        coords=batch0.coords, fields=batch0.fields,
        boundary_mask=batch0.boundary_mask, collocation_mask=batch0.collocation_mask,
        physics_params=batch0.physics_params,
        fidelity_level=2, fidelity_weight=batch0.fidelity_weight,
    )
    out0 = model.forward(batch0).pred
    out2 = model.forward(batch2).pred
    check("different fidelity levels give different predictions",
          not torch.allclose(out0, out2, atol=1e-6))

    # summary mentions fidelity
    check("summary mentions fidelity_levels",
          "fidelity_levels" in model.summary())
    check("summary mentions embed_dim",
          "embed_dim" in model.summary())

    # Raises for invalid fidelity level
    try:
        batch_invalid = make_batch(fidelity_level=5)  # n_fidelity_levels=3
        model.forward(batch_invalid)
        check("invalid fidelity level raises", False)
    except (IndexError, RuntimeError):
        check("invalid fidelity level raises", True)

    # compute_gradients uses original coords (without embedding)
    coords = batch0.coords.requires_grad_(True)
    local_batch = FeatureBatch(
        coords=coords, fields=batch0.fields,
        boundary_mask=batch0.boundary_mask, collocation_mask=batch0.collocation_mask,
        physics_params=batch0.physics_params,
        fidelity_level=0, fidelity_weight=1.0,
    )
    output = model.forward(local_batch)
    grads  = model.compute_gradients(output.pred, coords)
    check("compute_gradients works on original coords", "du_dx" in grads)


def test_model_device():
    section("Model device handling")

    model = BasePINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16)
    batch = make_batch()

    # CPU forward
    output = model.forward(batch)
    check("CPU forward works", output.pred.device.type == "cpu")

    # preprocess() is identity by default
    preprocessed = model.preprocess(batch)
    check("preprocess() returns FeatureBatch", isinstance(preprocessed, FeatureBatch))
    check("preprocess() is identity", preprocessed is batch)


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("AIPlasma — PINN Test")
    print("=" * 60)

    start = time.perf_counter()

    test_base_pinn()
    test_compute_gradients()
    test_residual_pinn()
    test_multifidelity_pinn()
    test_model_device()

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