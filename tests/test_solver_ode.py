#test 10

"""
test_ode_solver.py
ODE Solver tests — EulerODESolver, RK4ODESolver, AdaptiveODESolver.

Run:
    python tests/test_ode_solver.py
"""

import sys
import os
import math
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing.feature_pipeline import FeatureBatch
from core.base_model import ModelOutput
from core.base_solver import SolverOutput, SolverInfo
from models.pinn.base_pinn import BasePINN
from solvers.ode_solver import EulerODESolver, RK4ODESolver, AdaptiveODESolver

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


def make_batch(n: int = 50) -> FeatureBatch:
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


def make_model() -> BasePINN:
    return BasePINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16)


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

def test_euler_solver():
    section("EulerODESolver")

    batch  = make_batch()
    model  = make_model()
    solver = EulerODESolver()

    # solve() mode — no autograd
    model.eval()
    with torch.no_grad():
        output = model.forward(batch)
    solver_out = solver.solve(output, batch)

    check("solve() returns SolverOutput", isinstance(solver_out, SolverOutput))
    check("solve() has 'u' in quantities", "u" in solver_out.quantities)
    check("solve() solver_type", solver_out.solver_info.solver_type == "EulerODE")

    # solve_with_grad() mode
    solver_out2 = solver.solve_with_grad(model, batch)
    check("solve_with_grad() returns SolverOutput", isinstance(solver_out2, SolverOutput))
    check("solve_with_grad() has u_next", "u_next" in solver_out2.quantities)
    check("u_next shape (N, 1)", solver_out2.quantities["u_next"].shape == (50, 1))

    # step() directly
    step_out = solver.step(model, batch, dt=0.01)
    check("step() returns SolverOutput", isinstance(step_out, SolverOutput))
    check("step() has u_next", "u_next" in step_out.quantities)
    check("step() has dudt", "dudt" in step_out.quantities)
    check("step() converged", step_out.solver_info.converged)
    check("step() iterations = 1", step_out.solver_info.iterations == 1)
    check("step() u_next is finite",
          torch.isfinite(step_out.quantities["u_next"]).all())

    # Different dt gives different result
    out_small = solver.step(model, batch, dt=0.001)
    out_large = solver.step(model, batch, dt=0.1)
    check("different dt gives different u_next",
          not torch.allclose(
              out_small.quantities["u_next"],
              out_large.quantities["u_next"],
              atol=1e-6
          ))


def test_rk4_solver():
    section("RK4ODESolver")

    batch  = make_batch()
    model  = make_model()
    solver = RK4ODESolver()

    # solve() mode
    model.eval()
    with torch.no_grad():
        output = model.forward(batch)
    solver_out = solver.solve(output, batch)

    check("solve() returns SolverOutput", isinstance(solver_out, SolverOutput))
    check("solve() has 'u'", "u" in solver_out.quantities)
    check("solve() solver_type", solver_out.solver_info.solver_type == "RK4ODE")

    # solve_with_grad() mode
    solver_out2 = solver.solve_with_grad(model, batch)
    check("solve_with_grad() returns SolverOutput", isinstance(solver_out2, SolverOutput))
    check("solve_with_grad() has u_next", "u_next" in solver_out2.quantities)

    # step() — 4 internal evaluations
    step_out = solver.step(model, batch, dt=0.01)
    check("step() returns SolverOutput", isinstance(step_out, SolverOutput))
    check("step() iterations = 4", step_out.solver_info.iterations == 4)
    check("step() has k1", "k1" in step_out.quantities)
    check("step() has k4", "k4" in step_out.quantities)
    check("step() u_next is finite",
          torch.isfinite(step_out.quantities["u_next"]).all())

    # RK4 more accurate than Euler for same dt
    euler  = EulerODESolver()
    dt     = 0.1
    eu_out = euler.step(model, batch, dt=dt)
    rk_out = solver.step(model, batch, dt=dt)

    # Both should produce valid outputs
    check("RK4 u_next shape correct",
          rk_out.quantities["u_next"].shape == (50, 1))
    check("Euler and RK4 give different results for same dt",
          not torch.allclose(
              eu_out.quantities["u_next"],
              rk_out.quantities["u_next"],
              atol=1e-4
          ))


def test_adaptive_solver():
    section("AdaptiveODESolver (RK45)")

    batch  = make_batch()
    model  = make_model()
    solver = AdaptiveODESolver(dt_min=1e-6, dt_max=1.0)

    # step() returns tuple (SolverOutput, new_dt)
    result = solver.step(model, batch, dt=0.1, tol=1e-3)
    check("step() returns tuple", isinstance(result, tuple))
    check("tuple length 2", len(result) == 2)

    solver_out, new_dt = result
    check("first element is SolverOutput", isinstance(solver_out, SolverOutput))
    check("second element is float", isinstance(new_dt, float))
    check("new_dt > 0", new_dt > 0)
    check("new_dt <= dt_max", new_dt <= solver.dt_max)
    check("u_next in quantities", "u_next" in solver_out.quantities)
    check("error_norm in quantities", "error_norm" in solver_out.quantities)
    check("solver_type is AdaptiveRK45",
          solver_out.solver_info.solver_type == "AdaptiveRK45")
    check("converged", solver_out.solver_info.converged)
    check("u_next is finite",
          torch.isfinite(solver_out.quantities["u_next"]).all())

    # Tighter tolerance → smaller new_dt
    _, new_dt_tight = solver.step(model, batch, dt=0.1, tol=1e-6)
    _, new_dt_loose = solver.step(model, batch, dt=0.1, tol=1e-1)
    check("tighter tol → smaller or equal new_dt",
          new_dt_tight <= new_dt_loose + 1e-6)

    # solve_with_grad() discards new_dt and returns SolverOutput only
    solver_out2 = solver.solve_with_grad(model, batch)
    check("solve_with_grad() returns SolverOutput",
          isinstance(solver_out2, SolverOutput))

    # solve() — no autograd
    model.eval()
    with torch.no_grad():
        output = model.forward(batch)
    solver_out3 = solver.solve(output, batch)
    check("solve() returns SolverOutput", isinstance(solver_out3, SolverOutput))
    check("solve() has 'u'", "u" in solver_out3.quantities)


def test_solver_info():
    section("SolverInfo")

    info = SolverInfo(
        solver_type="test",
        converged=True,
        iterations=4,
        wall_time=0.123,
        extra={"custom": "value"},
    )
    check("solver_type", info.solver_type == "test")
    check("converged", info.converged)
    check("iterations", info.iterations == 4)
    check("wall_time", info.wall_time == 0.123)
    check("extra dict", info.extra["custom"] == "value")

    # Defaults
    info2 = SolverInfo(solver_type="minimal")
    check("converged default None", info2.converged is None)
    check("iterations default None", info2.iterations is None)
    check("wall_time default 0.0", info2.wall_time == 0.0)
    check("extra default empty", info2.extra == {})


def test_timed_solve():
    section("Timed solve methods")

    batch  = make_batch()
    model  = make_model()
    solver = EulerODESolver()

    model.eval()
    with torch.no_grad():
        output = model.forward(batch)

    timed_out = solver.timed_solve(output, batch)
    check("timed_solve returns SolverOutput", isinstance(timed_out, SolverOutput))
    check("timed_solve sets wall_time", timed_out.solver_info.wall_time >= 0.0)

    timed_grad = solver.timed_solve_with_grad(model, batch)
    check("timed_solve_with_grad returns SolverOutput",
          isinstance(timed_grad, SolverOutput))
    check("timed_solve_with_grad sets wall_time",
          timed_grad.solver_info.wall_time >= 0.0)


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("AIPlasma — ODE Solver Test")
    print("=" * 60)

    start = time.perf_counter()

    test_euler_solver()
    test_rk4_solver()
    test_adaptive_solver()
    test_solver_info()
    test_timed_solve()

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