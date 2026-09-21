#Test 11

"""
test_pde_solver.py
PDE Solver tests — AutogradPDESolver.

Run:
    python tests/test_pde_solver.py
"""

import sys
import os
import math
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing.feature_pipeline import FeatureBatch
from core.base_model import ModelOutput
from core.base_solver import SolverOutput
from models.pinn.base_pinn import BasePINN
from solvers.pde_solver import AutogradPDESolver
from solvers.physics_constraints import REGISTRY

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


def make_batch(n: int = 80, alpha: float = 0.01) -> FeatureBatch:
    x      = torch.rand(n)
    t      = torch.rand(n)
    coords = torch.stack([x, t], dim=1)
    u      = torch.sin(math.pi * x) * torch.exp(
        torch.tensor(-alpha * math.pi ** 2) * t
    )
    fields = u.unsqueeze(1)
    bm     = (x < 0.05) | (x > 0.95)
    return FeatureBatch(
        coords=coords, fields=fields,
        boundary_mask=bm, collocation_mask=~bm,
        physics_params={"alpha": alpha, "D": 0.1, "v": 0.5},
        fidelity_level=0, fidelity_weight=1.0,
    )


def make_model() -> BasePINN:
    return BasePINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16)


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

def test_solve_mode():
    section("AutogradPDESolver.solve() — no autograd")

    batch  = make_batch()
    model  = make_model()
    solver = AutogradPDESolver()

    model.eval()
    with torch.no_grad():
        output = model.forward(batch)

    solver_out = solver.solve(output, batch)

    check("returns SolverOutput", isinstance(solver_out, SolverOutput))
    check("has 'u' in quantities", "u" in solver_out.quantities)
    check("has 'u_boundary' in quantities", "u_boundary" in solver_out.quantities)
    check("has 'u_interior' in quantities", "u_interior" in solver_out.quantities)
    check("residuals is empty", solver_out.residuals == {})
    check("solver_type correct",
          solver_out.solver_info.solver_type == "AutogradPDE")
    check("u shape (N, 1)",
          solver_out.quantities["u"].shape == (80, 1))

    # u_boundary and u_interior are subsets
    n_bc  = batch.boundary_mask.sum().item()
    n_col = batch.collocation_mask.sum().item()
    check("u_boundary size matches boundary_mask",
          solver_out.quantities["u_boundary"].shape[0] == n_bc)
    check("u_interior size matches collocation_mask",
          solver_out.quantities["u_interior"].shape[0] == n_col)


def test_solve_with_grad_no_equation():
    section("AutogradPDESolver.solve_with_grad() — no equation")

    batch  = make_batch()
    model  = make_model()
    solver = AutogradPDESolver(equation=None)

    solver_out = solver.solve_with_grad(model, batch)

    check("returns SolverOutput", isinstance(solver_out, SolverOutput))
    check("has 'u' in quantities", "u" in solver_out.quantities)
    check("has 'du_dx'", "du_dx" in solver_out.quantities)
    check("has 'du_dt'", "du_dt" in solver_out.quantities)
    check("has 'd2u_dx2'", "d2u_dx2" in solver_out.quantities)
    check("has 'pde' in residuals", "pde" in solver_out.residuals)
    check("has 'bc' in residuals", "bc" in solver_out.residuals)

    # PDE residual is zeros when no equation
    pde_res = solver_out.residuals["pde"]
    check("pde residual is zeros without equation",
          torch.allclose(pde_res, torch.zeros_like(pde_res)))


def test_solve_with_grad_heat_equation():
    section("AutogradPDESolver.solve_with_grad() — heat equation")

    batch  = make_batch(alpha=0.01)
    model  = make_model()
    eq     = REGISTRY.get("heat_equation_1d")
    solver = AutogradPDESolver(equation=eq)

    solver_out = solver.solve_with_grad(model, batch)

    check("returns SolverOutput", isinstance(solver_out, SolverOutput))
    check("has du_dx", "du_dx" in solver_out.quantities)
    check("has du_dt", "du_dt" in solver_out.quantities)
    check("has d2u_dx2", "d2u_dx2" in solver_out.quantities)
    check("du_dx shape (N, 1)",
          solver_out.quantities["du_dx"].shape == (80, 1))
    check("du_dt shape (N, 1)",
          solver_out.quantities["du_dt"].shape == (80, 1))
    check("d2u_dx2 shape (N, 1)",
          solver_out.quantities["d2u_dx2"].shape == (80, 1))

    # PDE residual is non-trivial (untrained model)
    pde_res = solver_out.residuals["pde"]
    check("pde residual is non-zero for untrained model",
          pde_res.abs().mean().item() > 0)
    check("pde residual shape (N, 1)", pde_res.shape == (80, 1))

    # BC residual
    bc_res = solver_out.residuals["bc"]
    n_bc   = batch.boundary_mask.sum().item()
    check("bc residual shape (N_bc, 1)", bc_res.shape == (n_bc, 1))

    # solver_type
    check("solver_type correct",
          solver_out.solver_info.solver_type == "AutogradPDE")
    check("converged True", solver_out.solver_info.converged)


def test_solve_with_grad_drift_diffusion():
    section("AutogradPDESolver.solve_with_grad() — drift diffusion")

    batch  = make_batch()
    model  = make_model()
    eq     = REGISTRY.get("drift_diffusion_1d")
    solver = AutogradPDESolver(equation=eq)

    solver_out = solver.solve_with_grad(model, batch)

    check("returns SolverOutput", isinstance(solver_out, SolverOutput))
    check("pde residual computed", "pde" in solver_out.residuals)
    pde_res = solver_out.residuals["pde"]
    check("pde residual shape (N, 1)", pde_res.shape == (80, 1))
    check("pde residual is finite", torch.isfinite(pde_res).all())


def test_compute_derivatives():
    section("AutogradPDESolver.compute_derivatives()")

    batch  = make_batch()
    model  = make_model()
    solver = AutogradPDESolver()
    coords = batch.coords.requires_grad_(True)

    local_batch = FeatureBatch(
        coords=coords, fields=batch.fields,
        boundary_mask=batch.boundary_mask, collocation_mask=batch.collocation_mask,
        physics_params=batch.physics_params,
        fidelity_level=batch.fidelity_level, fidelity_weight=batch.fidelity_weight,
    )
    output = model.forward(local_batch)
    derivs = solver.compute_derivatives(output.pred, coords)

    check("returns dict", isinstance(derivs, dict))
    check("has du_dx", "du_dx" in derivs)
    check("has du_dt", "du_dt" in derivs)
    check("has d2u_dx2", "d2u_dx2" in derivs)
    check("all shapes (N, 1)",
          all(v.shape == (80, 1) for v in derivs.values()))
    check("all values finite",
          all(torch.isfinite(v).all() for v in derivs.values()))

    # Raises without requires_grad
    try:
        solver.compute_derivatives(output.pred, batch.coords)
        check("raises RuntimeError without requires_grad", False)
    except RuntimeError:
        check("raises RuntimeError without requires_grad", True)

    # Derivatives are differentiable
    try:
        loss = sum(v.pow(2).mean() for v in derivs.values())
        loss.backward()
        check("derivatives are differentiable for backprop", True)
    except Exception as e:
        check("derivatives are differentiable for backprop", False, str(e))


def test_bc_residual():
    section("AutogradPDESolver BC residual")

    batch  = make_batch()
    model  = make_model()
    solver = AutogradPDESolver()

    solver_out = solver.solve_with_grad(model, batch)
    bc_res     = solver_out.residuals["bc"]
    n_bc       = batch.boundary_mask.sum().item()

    check("BC residual shape matches boundary points",
          bc_res.shape == (n_bc, 1))
    check("BC residual is finite", torch.isfinite(bc_res).all())

    # Batch with no boundary points
    x      = torch.rand(50) * 0.5 + 0.25  # all interior: [0.25, 0.75]
    t      = torch.rand(50)
    coords = torch.stack([x, t], dim=1)
    no_bc_batch = FeatureBatch(
        coords=coords,
        fields=torch.rand(50, 1),
        boundary_mask=torch.zeros(50, dtype=torch.bool),
        collocation_mask=torch.ones(50, dtype=torch.bool),
        physics_params={"alpha": 0.01},
        fidelity_level=0, fidelity_weight=1.0,
    )
    solver_out2 = solver.solve_with_grad(model, no_bc_batch)
    check("no BC points → zero BC residual",
          torch.allclose(
              solver_out2.residuals["bc"],
              torch.zeros_like(solver_out2.residuals["bc"])
          ))


def test_pde_loss_decreases():
    section("PDE loss decreases during training")

    batch  = make_batch()
    model  = make_model()
    eq     = REGISTRY.get("heat_equation_1d")
    solver = AutogradPDESolver(equation=eq)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(10):
        opt.zero_grad()
        solver_out = solver.solve_with_grad(model, batch)
        pde_loss   = solver_out.residuals["pde"].pow(2).mean()
        bc_loss    = solver_out.residuals["bc"].pow(2).mean()
        loss       = pde_loss + bc_loss
        loss.backward()
        opt.step()
        losses.append(loss.item())

    check("loss is finite throughout training",
          all(math.isfinite(l) for l in losses))
    check("loss decreases or stays stable over 10 steps",
          losses[-1] <= losses[0] * 2.0,
          f"first={losses[0]:.6f}, last={losses[-1]:.6f}")


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("AIPlasma — PDE Solver Test")
    print("=" * 60)

    start = time.perf_counter()

    test_solve_mode()
    test_solve_with_grad_no_equation()
    test_solve_with_grad_heat_equation()
    test_solve_with_grad_drift_diffusion()
    test_compute_derivatives()
    test_bc_residual()
    test_pde_loss_decreases()

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