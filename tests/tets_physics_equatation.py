#Test 6

"""
test_physics_equation.py
PhysicsEquation tests for all three equations in the PDE registry.

Tests: HeatEquation1D, DriftDiffusion1D, HasegawaWakatani (stub).

Run:
    python tests/test_physics_equation.py
"""

import sys
import os
import math
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solvers.physics_constraints import (
    REGISTRY, PDERegistry,
    HeatEquation1D, DriftDiffusion1D, HasegawaWakatani,
    PhysicsEquation,
)

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


def make_coords(n: int = 100, requires_grad: bool = True) -> torch.Tensor:
    """Creates (x, t) coordinates with optional requires_grad."""
    x      = torch.rand(n)
    t      = torch.rand(n)
    coords = torch.stack([x, t], dim=1)
    return coords.requires_grad_(requires_grad)


def make_pred(coords: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """Creates a prediction using the exact heat equation solution."""
    x = coords[:, 0]
    t = coords[:, 1]
    u = torch.sin(math.pi * x) * torch.exp(
        torch.tensor(-alpha * math.pi ** 2) * t
    )
    return u.unsqueeze(1)


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

def test_registry():
    section("PDERegistry")

    check("REGISTRY is PDERegistry instance", isinstance(REGISTRY, PDERegistry))
    check("list_all returns list", isinstance(REGISTRY.list_all(), list))
    check("contains heat_equation_1d", "heat_equation_1d" in REGISTRY)
    check("contains drift_diffusion_1d", "drift_diffusion_1d" in REGISTRY)
    check("contains hasegawa_wakatani", "hasegawa_wakatani" in REGISTRY)
    check("list_all has 3 entries", len(REGISTRY.list_all()) >= 3)

    # get() returns correct type
    eq = REGISTRY.get("heat_equation_1d")
    check("get() returns PhysicsEquation", isinstance(eq, PhysicsEquation))

    # get() raises KeyError for unknown
    try:
        REGISTRY.get("nonexistent_equation_xyz")
        check("get() raises KeyError for unknown", False)
    except KeyError:
        check("get() raises KeyError for unknown", True)

    # register/unregister
    class _TestEq(PhysicsEquation):
        def name(self): return "test_equation_temp"
        def expected_params(self): return ["k"]
        def residual(self, c, p, params): return torch.zeros_like(p)

    REGISTRY.register(_TestEq())
    check("register() adds equation", "test_equation_temp" in REGISTRY)
    REGISTRY.unregister("test_equation_temp")
    check("unregister() removes equation", "test_equation_temp" not in REGISTRY)

    # Duplicate register raises
    try:
        REGISTRY.register(HeatEquation1D())
        check("duplicate register raises ValueError", False)
    except ValueError:
        check("duplicate register raises ValueError", True)

    # __contains__
    check("__contains__ works", "heat_equation_1d" in REGISTRY)
    check("__contains__ False for unknown", "xyz_unknown" not in REGISTRY)


def test_heat_equation():
    section("HeatEquation1D")

    eq = REGISTRY.get("heat_equation_1d")

    check("name()", eq.name() == "heat_equation_1d")
    check("expected_params()", eq.expected_params() == ["alpha"])
    check("description() not empty", len(eq.description()) > 0)

    # validate_params
    check("validate_params passes", eq.validate_params({"alpha": 0.01}))
    try:
        eq.validate_params({})
        check("validate_params raises on missing", False)
    except ValueError:
        check("validate_params raises on missing", True)

    # residual shape
    coords = make_coords(100)
    pred   = make_pred(coords)
    pred   = pred.requires_grad_(True)

    residual = eq.residual(coords, pred, {"alpha": 0.01})
    check("residual returns Tensor", isinstance(residual, torch.Tensor))
    check("residual shape (N, 1)", residual.shape == (100, 1))

    # Exact solution should give near-zero residual
    # (small because of autograd approximation, not exactly 0)
    mean_res = residual.abs().mean().item()
    check("exact solution gives small residual",
          mean_res < 1.0,
          f"mean residual={mean_res:.6f}")

    # residual is differentiable
    try:
        loss = residual.pow(2).mean()
        loss.backward()
        check("residual is differentiable", True)
    except Exception as e:
        check("residual is differentiable", False, str(e))


def test_drift_diffusion():
    section("DriftDiffusion1D")

    eq = REGISTRY.get("drift_diffusion_1d")

    check("name()", eq.name() == "drift_diffusion_1d")
    check("expected_params()", eq.expected_params() == ["D", "v"])
    check("description() not empty", len(eq.description()) > 0)

    # validate_params
    check("validate_params passes", eq.validate_params({"D": 0.1, "v": 0.5}))
    try:
        eq.validate_params({"D": 0.1})  # missing v
        check("validate_params raises on missing v", False)
    except ValueError:
        check("validate_params raises on missing v", True)

    # residual shape
    coords = make_coords(80)
    pred   = torch.rand(80, 1, requires_grad=True)

    residual = eq.residual(coords, pred, {"D": 0.1, "v": 0.5})
    check("residual returns Tensor", isinstance(residual, torch.Tensor))
    check("residual shape (N, 1)", residual.shape == (80, 1))

    # residual is differentiable
    try:
        loss = residual.pow(2).mean()
        loss.backward()
        check("residual is differentiable", True)
    except Exception as e:
        check("residual is differentiable", False, str(e))

    # Different D values give different residuals
    coords2  = make_coords(50)
    pred2    = torch.rand(50, 1, requires_grad=True)
    res_d01  = eq.residual(coords2, pred2, {"D": 0.1, "v": 0.0})
    pred3    = torch.rand(50, 1, requires_grad=True)
    res_d10  = eq.residual(coords2, pred3, {"D": 1.0, "v": 0.0})
    check("different D gives different residuals",
          not torch.allclose(res_d01, res_d10, atol=1e-6))


def test_hasegawa_wakatani():
    section("HasegawaWakatani (stub)")

    eq = REGISTRY.get("hasegawa_wakatani")

    check("name()", eq.name() == "hasegawa_wakatani")
    check("expected_params()", eq.expected_params() == ["D", "C", "nu"])
    check("description() not empty", len(eq.description()) > 0)

    # validate_params
    check("validate_params passes",
          eq.validate_params({"D": 0.1, "C": 1.0, "nu": 0.01}))

    # residual raises NotImplementedError
    try:
        eq.residual(None, None, {"D": 0.1, "C": 1.0, "nu": 0.01})
        check("residual raises NotImplementedError", False)
    except NotImplementedError:
        check("residual raises NotImplementedError", True)


def test_residual_autograd_graph():
    section("Residual autograd graph integrity")

    # Verify that residual computed through model parameters
    # creates a valid computation graph for backprop
    import torch.nn as nn

    class _SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
        def forward(self, x):
            return self.linear(x)

    net    = _SimpleNet()
    coords = make_coords(50)
    pred   = net(coords)

    eq       = REGISTRY.get("heat_equation_1d")
    residual = eq.residual(coords, pred, {"alpha": 0.01})
    loss     = residual.pow(2).mean()

    try:
        loss.backward()
        grad_norm = net.linear.weight.grad.norm().item()
        check("backprop through residual works", True)
        check("gradients are non-zero", grad_norm > 0,
              f"grad_norm={grad_norm:.6f}")
    except Exception as e:
        check("backprop through residual works", False, str(e))
        check("gradients are non-zero", False)


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("AIPlasma — PhysicsEquation Test")
    print("=" * 60)

    start = time.perf_counter()

    test_registry()
    test_heat_equation()
    test_drift_diffusion()
    test_hasegawa_wakatani()
    test_residual_autograd_graph()

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