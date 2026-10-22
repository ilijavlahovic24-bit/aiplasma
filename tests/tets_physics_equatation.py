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
from data.preprocessing.feature_pipeline import FeatureBatch
from models.pinn.base_pinn import BasePINN
from solvers.pde_solver import AutogradPDESolver
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
    coords = make_coords(80)
    net = torch.nn.Linear(2, 1)
    pred = net(coords)

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
    coords2 = make_coords(50)
    net2 = torch.nn.Linear(2, 1)
    pred2 = net2(coords2)
    res_d01 = eq.residual(coords2, pred2, {"D": 0.1, "v": 0.0})

    coords3 = make_coords(50)
    net3 = torch.nn.Linear(2, 1)
    pred3 = net3(coords3)
    res_d10 = eq.residual(coords3, pred3, {"D": 1.0, "v": 0.0})

    check("different D gives different residuals",
          not torch.allclose(res_d01, res_d10, atol=1e-6))

def make_coords_3d(n: int = 60) -> torch.Tensor:
    """Creates (x, y, t) coordinates with requires_grad."""
    return torch.rand(n, 3, dtype=torch.float64).requires_grad_(True)


def hw_fields(coords: torch.Tensor) -> torch.Tensor:
    """Smooth test fields (n, phi, zeta) as one (N, 3) tensor."""
    x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
    e    = torch.exp(-t)
    phi  = torch.sin(x) * torch.cos(y) * e
    n    = torch.cos(x) * torch.sin(y) * e
    zeta = torch.sin(2 * x) * torch.cos(y) * e
    return torch.cat([n, phi, zeta], dim=1)


def hw_expected(coords, D, C, nu, kappa, S) -> torch.Tensor:
    """Residual from hand-derived analytic derivatives of hw_fields()."""
    x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
    e = torch.exp(-t)

    phi = torch.sin(x) * torch.cos(y) * e
    phi_x, phi_y, lap_phi = torch.cos(x) * torch.cos(y) * e, -torch.sin(x) * torch.sin(y) * e, -2 * phi

    n = torch.cos(x) * torch.sin(y) * e
    n_x, n_y, n_t, lap_n = -torch.sin(x) * torch.sin(y) * e, torch.cos(x) * torch.cos(y) * e, -n, -2 * n

    zeta = torch.sin(2 * x) * torch.cos(y) * e
    zeta_x = 2 * torch.cos(2 * x) * torch.cos(y) * e
    zeta_y = -torch.sin(2 * x) * torch.sin(y) * e
    zeta_t, lap_zeta = -zeta, -5 * zeta

    res_n    = n_t + (phi_x * n_y - phi_y * n_x) - C * (phi - n) + kappa * phi_y - D * lap_n - S
    res_zeta = zeta_t + (phi_x * zeta_y - phi_y * zeta_x) - C * (phi - n) - nu * lap_zeta
    res_con  = zeta - lap_phi
    return torch.cat([res_n, res_zeta, res_con], dim=1)


def test_hasegawa_wakatani():
    section("HasegawaWakatani")

    eq = REGISTRY.get("hasegawa_wakatani")

    check("name()", eq.name() == "hasegawa_wakatani")
    check("expected_params()", eq.expected_params() == ["D", "C", "nu"])
    check("description() not empty", len(eq.description()) > 0)
    check("validate_params passes",
          eq.validate_params({"D": 0.1, "C": 1.0, "nu": 0.01}))

    # Residual against analytic derivatives (also verifies sign conventions)
    params = {"D": 0.1, "C": 0.7, "nu": 0.02, "kappa": 0.8, "S": 0.05}
    coords = make_coords_3d(60)
    pred   = hw_fields(coords)
    res    = eq.residual(coords, pred, params)

    check("residual shape (N, 3)", res.shape == (60, 3))
    diff = (res - hw_expected(coords.detach(), **params)).abs().max().item()
    check("residual matches analytic derivatives", diff < 1e-9, f"max diff={diff:.2e}")

    # Optional parameters
    base = {"D": 0.1, "C": 0.7, "nu": 0.02}
    check("kappa defaults to 1.0 and S to 0.0",
          torch.allclose(
              eq.residual(coords, pred, base),
              eq.residual(coords, pred, {**base, "kappa": 1.0, "S": 0.0}),
          ))

    # Zero fields: exercises the "second derivative is zero" path
    z    = coords[:, 0:1] * 0.0
    zero = eq.residual(coords, torch.cat([z, z, z], dim=1), base)
    check("zero fields give zero residual", torch.allclose(zero, torch.zeros_like(zero)))

    # Layout errors
    try:
        eq.residual(torch.rand(10, 2, dtype=torch.float64, requires_grad=True),
                    torch.rand(10, 3, dtype=torch.float64), base)
        check("2-column coords raise ValueError", False)
    except ValueError:
        check("2-column coords raise ValueError", True)

    try:
        eq.residual(coords, pred[:, :2], base)
        check("2-channel pred raises ValueError", False)
    except ValueError:
        check("2-channel pred raises ValueError", True)

    try:
        eq.residual(torch.rand(10, 3), torch.rand(10, 3), base)
        check("coords without requires_grad raise RuntimeError", False)
    except RuntimeError:
        check("coords without requires_grad raise RuntimeError", True)

    # Backprop through the residual (third-order graph) into network weights
    net    = torch.nn.Sequential(
        torch.nn.Linear(3, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3)
    ).double()
    coords = make_coords_3d(40)
    loss   = eq.residual(coords, net(coords), base).pow(2).mean()
    try:
        loss.backward()
        check("backprop through residual works", True)
        check("all weights have gradients",
              all(p.grad is not None for p in net.parameters()))
    except Exception as e:
        check("backprop through residual works", False, str(e))
        check("all weights have gradients", False)

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

def test_solve_with_grad_hasegawa_wakatani():
    section("AutogradPDESolver.solve_with_grad() — Hasegawa-Wakatani")

    n = 80
    x, y, t = torch.rand(n), torch.rand(n), torch.rand(n)
    bm      = (x < 0.05) | (x > 0.95)
    bm[0]   = True                                    # guarantee at least one boundary point
    batch   = FeatureBatch(
        coords=torch.stack([x, y, t], dim=1),
        fields=torch.rand(n, 3),                      # (n, phi, zeta) targets at boundary
        boundary_mask=bm,
        collocation_mask=~bm,
        physics_params={"D": 0.1, "C": 0.7, "nu": 0.02},
        fidelity_level=0, fidelity_weight=1.0,
    )
    model  = BasePINN(input_dim=3, output_dim=3, hidden_layers=2,
                      hidden_size=16, activation="tanh")
    solver = AutogradPDESolver(equation=REGISTRY.get("hasegawa_wakatani"))

    solver_out = solver.solve_with_grad(model, batch)
    pde_res    = solver_out.residuals["pde"]
    bc_res     = solver_out.residuals["bc"]

    check("pde residual shape (N, 3)", pde_res.shape == (n, 3))
    check("pde residual is finite", torch.isfinite(pde_res).all())
    check("bc residual shape (N_bc, 3)", bc_res.shape == (int(bm.sum()), 3))

    loss = pde_res.pow(2).mean() + bc_res.pow(2).mean()
    loss.backward()
    check("backprop through the full solver works",
          all(p.grad is not None for p in model.parameters()))
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
    test_solve_with_grad_hasegawa_wakatani()
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