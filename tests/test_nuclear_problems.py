# Test 13
#Test 12

"""
test_nuclear_problems.py
Nuclear-material and radiation problems: equations, helpers, boundary
conditions, normalization and integration with the solver.

Every equation is checked against a closed-form solution.

Run:
    python tests/test_nuclear_problems.py
"""

import sys
import os
import math
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solvers.physics_constraints import REGISTRY
from solvers.pde_solver import AutogradPDESolver
from core.boundary_conditions import dirichlet, neumann, robin
from core.material_problem import NuclearMaterialProblem, arrhenius, dpa_from_flux
from core.radiation_problem import RadiationProblem, half_life_to_lambda, mean_free_path
from data.preprocessing.feature_pipeline import FeatureBatch
from data.sources.physical_tensor import PhysicalTensor, UnitSystem, Domain, CoordinateSystem
from models.pinn.base_pinn import BasePINN

# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

PASSED = 0
FAILED = 0

DT = torch.float64


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


def raises(exc, fn) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def max_abs(t: torch.Tensor) -> float:
    return t.abs().max().item()


def coords_2d(n: int = 60, lo: float = 0.0) -> torch.Tensor:
    """(x, t) points in [lo, 1]^2 with requires_grad, float64."""
    c = lo + (1.0 - lo) * torch.rand(n, 2, dtype=DT)
    return c.requires_grad_(True)


# ════════════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════════════

NEW_EQUATIONS = {
    "heat_conduction_source_1d":   ["k", "rho_cp", "q"],
    "fission_gas_booth":           ["D", "lam", "beta"],
    "point_defect_rate_theory_1d": ["D_v", "D_i", "K0", "K_iv", "K_v", "K_i"],
    "neutron_diffusion_1d":        ["v", "D", "Sigma_a", "nu_Sigma_f"],
    "bateman_chain":               ["lambdas"],
    "radiative_loss_1d":           ["n_e", "kappa"],
}


def test_registry():
    section("Registry")

    for name, params in NEW_EQUATIONS.items():
        check(f"{name} is registered", name in REGISTRY)
        eq = REGISTRY.get(name)
        check(f"{name} expected_params", eq.expected_params() == params)
        check(f"{name} description not empty", len(eq.description()) > 0)

    check("registry has at least 9 equations", len(REGISTRY.list_all()) >= 9)
    check("missing parameter raises ValueError",
          raises(ValueError, lambda: REGISTRY.get("heat_conduction_source_1d").validate_params({"k": 1.0})))


# ════════════════════════════════════════════════════════════════════════════
# Equations against closed-form solutions
# ════════════════════════════════════════════════════════════════════════════

def test_heat_conduction():
    section("heat_conduction_source_1d")

    eq     = REGISTRY.get("heat_conduction_source_1d")
    params = {"k": 0.8, "rho_cp": 2.0, "q": 1.5}
    alpha  = params["k"] / params["rho_cp"]

    coords = coords_2d()
    x, t   = coords[:, 0:1], coords[:, 1:2]
    # Steady parabola with fixed ends plus one decaying sine mode
    T = (params["q"] * x * (1 - x) / (2 * params["k"])
         + 0.7 * torch.sin(math.pi * x) * torch.exp(-alpha * math.pi ** 2 * t))

    res = eq.residual(coords, T, params)
    check("residual shape (N, 1)", res.shape == (60, 1))
    check("exact solution gives zero residual", max_abs(res) < 1e-9, f"max={max_abs(res):.2e}")

    wrong = eq.residual(coords, 1.0 + x, params)
    check("wrong solution gives non-zero residual", max_abs(wrong) > 1e-3)

    check("3-column coords raise ValueError",
          raises(ValueError, lambda: eq.residual(torch.rand(5, 3, dtype=DT, requires_grad=True),
                                                 torch.rand(5, 1, dtype=DT), params)))
    check("coords without requires_grad raise RuntimeError",
          raises(RuntimeError, lambda: eq.residual(torch.rand(5, 2, dtype=DT),
                                                   torch.rand(5, 1, dtype=DT), params)))


def test_fission_gas_booth():
    section("fission_gas_booth")

    eq = REGISTRY.get("fission_gas_booth")
    D, lam, beta, a = 0.05, 0.3, 1.2, 1.0
    params = {"D": D, "lam": lam, "beta": beta}

    coords = coords_2d(lo=0.1)                      # r >= 0.1: c has a 1/r mode
    r, t   = coords[:, 0:1], coords[:, 1:2]
    kap    = math.sqrt(lam / D)
    # Steady profile with a perfect sink at r = a, plus a decaying transient mode
    c = (beta / lam * (1 - a * torch.sinh(kap * r) / (r * math.sinh(kap * a)))
         + 0.4 * torch.sin(math.pi * r / a) / r
         * torch.exp(-(D * math.pi ** 2 / a ** 2 + lam) * t))

    res = eq.residual(coords, c, params)
    check("residual shape (N, 1)", res.shape == (60, 1))
    check("exact solution gives zero residual", max_abs(res) < 1e-8, f"max={max_abs(res):.2e}")
    check("wrong solution gives non-zero residual",
          max_abs(eq.residual(coords, 1.0 + r, params)) > 1e-3)


def test_rate_theory():
    section("point_defect_rate_theory_1d")

    eq = REGISTRY.get("point_defect_rate_theory_1d")
    p  = {"D_v": 0.1, "D_i": 1.0, "K0": 0.02, "K_iv": 5.0, "K_v": 0.3, "K_i": 0.8}

    # Uniform stationary limit
    a   = p["K_iv"] * p["K_v"] / p["K_i"]
    C_v = (-p["K_v"] + math.sqrt(p["K_v"] ** 2 + 4 * a * p["K0"])) / (2 * a)
    C_i = p["K_v"] / p["K_i"] * C_v

    coords = coords_2d()
    x      = coords[:, 0:1]
    pred   = torch.cat([C_v + 0.0 * x, C_i + 0.0 * x], dim=1)   # kept connected to coords
    res    = eq.residual(coords, pred, p)
    check("residual shape (N, 2)", res.shape == (60, 2))
    check("stationary limit gives zero residual", max_abs(res) < 1e-9, f"max={max_abs(res):.2e}")

    # Non-uniform fields against hand-derived derivatives
    coords = coords_2d()
    x, t   = coords[:, 0:1], coords[:, 1:2]
    e      = torch.exp(-t)
    Cv     = 0.05 + 0.02 * torch.sin(x) * e
    Ci     = 0.02 + 0.01 * torch.cos(x) * e
    Cv_t, Cv_xx = -0.02 * torch.sin(x) * e, -0.02 * torch.sin(x) * e
    Ci_t, Ci_xx = -0.01 * torch.cos(x) * e, -0.01 * torch.cos(x) * e
    rec    = p["K_iv"] * Ci * Cv
    expected = torch.cat([
        Cv_t - p["D_v"] * Cv_xx - p["K0"] + rec + p["K_v"] * Cv,
        Ci_t - p["D_i"] * Ci_xx - p["K0"] + rec + p["K_i"] * Ci,
    ], dim=1)
    res = eq.residual(coords, torch.cat([Cv, Ci], dim=1), p)
    diff = max_abs(res - expected.detach())
    check("matches hand-derived residual", diff < 1e-9, f"max diff={diff:.2e}")

    check("1-channel pred raises ValueError",
          raises(ValueError, lambda: eq.residual(coords, Cv, p)))


def test_neutron_diffusion():
    section("neutron_diffusion_1d")

    eq = REGISTRY.get("neutron_diffusion_1d")
    p  = {"v": 2.2, "D": 0.9, "Sigma_a": 0.07, "nu_Sigma_f": 0.09}

    coords = coords_2d()
    x, t   = coords[:, 0:1], coords[:, 1:2]
    B      = math.pi
    omega  = p["v"] * (p["nu_Sigma_f"] - p["Sigma_a"] - p["D"] * B ** 2)
    phi    = torch.cos(B * x) * torch.exp(omega * t)             # slab eigenmode

    res = eq.residual(coords, phi, p)
    check("residual shape (N, 1)", res.shape == (60, 1))
    check("eigenmode gives zero residual", max_abs(res) < 1e-9, f"max={max_abs(res):.2e}")

    res_S = eq.residual(coords, phi, {**p, "S": 0.5})
    check("source S enters with a minus sign",
          torch.allclose((res - res_S).detach(), torch.full_like(res, 0.5), atol=1e-9))


def test_bateman():
    section("bateman_chain")

    eq = REGISTRY.get("bateman_chain")
    l0, l1, l2 = 0.5, 0.2, 0.05
    p = {"lambdas": [l0, l1, l2]}

    t  = torch.rand(50, 1, dtype=DT).requires_grad_(True)
    N0 = torch.exp(-l0 * t)
    N1 = l0 / (l1 - l0) * (torch.exp(-l0 * t) - torch.exp(-l1 * t))
    N2 = l0 * l1 * (torch.exp(-l0 * t) / ((l1 - l0) * (l2 - l0))
                    + torch.exp(-l1 * t) / ((l0 - l1) * (l2 - l1))
                    + torch.exp(-l2 * t) / ((l0 - l2) * (l1 - l2)))

    res = eq.residual(t, torch.cat([N0, N1, N2], dim=1), p)
    check("residual shape (N, 3)", res.shape == (50, 3))
    check("closed-form chain gives zero residual", max_abs(res) < 1e-9, f"max={max_abs(res):.2e}")

    res_t = eq.residual(t, torch.cat([N0, N1, N2], dim=1),
                        {"lambdas": torch.tensor([l0, l1, l2], dtype=DT)})
    check("lambdas as tensor give the same residual", torch.allclose(res, res_t))

    check("wrong number of lambdas raises ValueError",
          raises(ValueError, lambda: eq.residual(t, torch.cat([N0, N1, N2], dim=1),
                                                 {"lambdas": [l0, l1]})))


def test_radiative_loss():
    section("radiative_loss_1d")

    eq = REGISTRY.get("radiative_loss_1d")
    p  = {"n_e": 2.0, "kappa": 0.7, "Z_eff": 1.5, "c_br": 0.3, "T_unit_J": 1.0}
    # Uniform cooling: sqrt(T) = sqrt(T0) - A t / 2, A = c Z n / (1.5 T_unit) = 0.6, T0 = 4
    coords = coords_2d()
    t      = coords[:, 1:2]
    T      = (2.0 - 0.3 * t) ** 2

    res = eq.residual(coords, T, p)
    check("residual shape (N, 1)", res.shape == (60, 1))
    check("uniform bremsstrahlung cooling gives zero residual",
          max_abs(res) < 1e-9, f"max={max_abs(res):.2e}")

    L_z = lambda temp: 0.01 * temp
    res_line = eq.residual(coords, T, {**p, "n_imp": 0.5, "L_z": L_z})
    check("line radiation adds n_e n_imp L_z(T)",
          torch.allclose((res_line - res).detach(), (2.0 * 0.5 * 0.01 * T).detach(), atol=1e-9))

    res_heat = eq.residual(coords, T, {**p, "P_heat": 3.0})
    check("P_heat enters with a minus sign",
          torch.allclose((res - res_heat).detach(), torch.full_like(res, 3.0), atol=1e-9))


# ════════════════════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════════════════════

def test_helper_functions():
    section("Helper functions")

    check("arrhenius with E = 0 returns the prefactor", arrhenius(2.0, 0.0, 300.0) == 2.0)
    check("arrhenius increases with temperature",
          arrhenius(1.0, 1.0, 900.0) > arrhenius(1.0, 1.0, 600.0))
    check("arrhenius accepts a tensor temperature",
          isinstance(arrhenius(1.0, 1.0, torch.tensor([600.0, 900.0])), torch.Tensor))

    check("dpa_from_flux", math.isclose(dpa_from_flux(1e18, 1e-25, 1e7), 1.0, rel_tol=1e-9))
    check("dpa_from_flux is linear in time",
          math.isclose(dpa_from_flux(1e18, 1e-25, 2.0), 2 * dpa_from_flux(1e18, 1e-25, 1.0)))

    check("half_life_to_lambda(1) = ln 2", math.isclose(half_life_to_lambda(1.0), math.log(2.0)))
    check("lambda * half_life = ln 2", math.isclose(half_life_to_lambda(30.0) * 30.0, math.log(2.0)))
    check("mean_free_path", math.isclose(mean_free_path(4.0), 0.25))


# ════════════════════════════════════════════════════════════════════════════
# Boundary-condition helpers
# ════════════════════════════════════════════════════════════════════════════

def test_boundary_helpers():
    section("Boundary-condition helpers")

    x      = torch.linspace(0.0, 1.0, 21, dtype=DT)
    coords = torch.stack([x, torch.rand(21, dtype=DT)], dim=1).requires_grad_(True)
    pred   = coords[:, 0:1] ** 2 + coords[:, 1:2]          # du/dx = 2x
    mask   = coords[:, 0] > 0.9                             # x = 0.95 and x = 1.0
    xs     = coords[:, 0:1].detach()

    res = dirichlet(pred, pred.detach(), mask)
    check("dirichlet shape (N_b, F)", res.shape == (2, 1))
    check("dirichlet zero when pred matches the target", max_abs(res) < 1e-12)
    check("dirichlet with a float target",
          torch.allclose(dirichlet(pred, 0.0, mask), pred[mask]))

    res = neumann(pred, coords, mask, axis=0, target=2 * xs)
    check("neumann matches du/dx = 2x", max_abs(res) < 1e-9, f"max={max_abs(res):.2e}")
    res = neumann(pred, coords, mask, axis=0, target=-2 * xs, sign=-1.0)
    check("neumann with a reversed normal", max_abs(res) < 1e-9)

    target = 3.0 * pred.detach() + 0.5 * 2 * xs
    res = robin(pred, coords, mask, axis=0, alpha=3.0, beta=0.5, target=target)
    check("robin matches alpha u + beta du/dx", max_abs(res) < 1e-9, f"max={max_abs(res):.2e}")

    empty = torch.zeros(21, dtype=torch.bool)
    check("empty mask gives a zero residual of shape (1, F)",
          dirichlet(pred, 0.0, empty).shape == (1, 1)
          and neumann(pred, coords, empty, axis=0).shape == (1, 1)
          and robin(pred, coords, empty, axis=0, alpha=1.0, beta=1.0).shape == (1, 1))

    check("neumann without requires_grad raises RuntimeError",
          raises(RuntimeError, lambda: neumann(pred, coords.detach(), mask, axis=0)))


# ════════════════════════════════════════════════════════════════════════════
# Problem classes
# ════════════════════════════════════════════════════════════════════════════

class HeatRod(NuclearMaterialProblem):
    equation = REGISTRY.get("heat_conduction_source_1d")

    def physics_params(self) -> dict:
        return {"k": 0.8, "rho_cp": 2.0, "q": 1.5}


class DecayChain(RadiationProblem):
    equation = REGISTRY.get("bateman_chain")

    def physics_params(self) -> dict:
        return {"lambdas": [0.5, 0.2, 0.05]}


def make_batch(coords, fields, params=None, n_boundary: int = 5) -> FeatureBatch:
    n  = coords.shape[0]
    bm = torch.zeros(n, dtype=torch.bool)
    bm[:n_boundary] = True
    return FeatureBatch(
        coords=coords, fields=fields,
        boundary_mask=bm, collocation_mask=~bm,
        physics_params=params or {},
        fidelity_level=0, fidelity_weight=1.0,
    )


def heat_solution(coords):
    k, rho_cp, q = 0.8, 2.0, 1.5
    x, t = coords[:, 0:1], coords[:, 1:2]
    return (q * x * (1 - x) / (2 * k)
            + 0.7 * torch.sin(math.pi * x) * torch.exp(-(k / rho_cp) * math.pi ** 2 * t))


def test_problem_classes():
    section("NuclearMaterialProblem / RadiationProblem")

    coords = coords_2d(40)
    pred   = heat_solution(coords)
    batch  = make_batch(coords, pred.detach())
    problem = HeatRod()

    res = problem.pde_residual(batch, pred)
    check("pde_residual shape (N, 1)", res.shape == (40, 1))
    check("pde_residual is zero for the exact solution", max_abs(res) < 1e-9)

    bc = problem.boundary_conditions(batch, pred)
    check("default boundary_conditions is Dirichlet from data",
          bc.shape == (5, 1) and max_abs(bc) < 1e-12)

    # Parameters in batch.physics_params are used when the problem defines none
    class NoParams(NuclearMaterialProblem):
        equation = REGISTRY.get("heat_conduction_source_1d")

    batch_p = make_batch(coords, pred.detach(), {"k": 0.8, "rho_cp": 2.0, "q": 1.5})
    check("batch.physics_params are used as a fallback",
          max_abs(NoParams().pde_residual(batch_p, pred)) < 1e-9)

    check("no equation raises ValueError",
          raises(ValueError, lambda: NuclearMaterialProblem().pde_residual(batch, pred)))
    check("validate_material passes", problem.validate_material(batch) is None)
    check("validate_material without equation raises ValueError",
          raises(ValueError, lambda: NuclearMaterialProblem().validate_material(batch)))

    check("fidelity_weights default", problem.fidelity_weights() == [1.0])

    class ThreeLevels(HeatRod):
        n_fidelity_levels = 3
    w = ThreeLevels().fidelity_weights()
    check("fidelity_weights for 3 levels sum to 1", len(w) == 3 and abs(sum(w) - 1.0) < 1e-12)

    # Radiation: a decay chain
    t = torch.rand(30, 1, dtype=DT).requires_grad_(True)
    l0, l1, l2 = 0.5, 0.2, 0.05
    N0 = torch.exp(-l0 * t)
    N1 = l0 / (l1 - l0) * (torch.exp(-l0 * t) - torch.exp(-l1 * t))
    N2 = l0 * l1 * (torch.exp(-l0 * t) / ((l1 - l0) * (l2 - l0))
                    + torch.exp(-l1 * t) / ((l0 - l1) * (l2 - l1))
                    + torch.exp(-l2 * t) / ((l0 - l2) * (l1 - l2)))
    chain_pred  = torch.cat([N0, N1, N2], dim=1)
    chain_batch = make_batch(t, chain_pred.detach())
    res = DecayChain().pde_residual(chain_batch, chain_pred)
    check("radiation pde_residual shape (N, 3)", res.shape == (30, 3))
    check("radiation pde_residual is zero for the closed form", max_abs(res) < 1e-9)
    check("validate_radiation passes", DecayChain().validate_radiation(chain_batch) is None)


def test_normalization():
    section("Unit normalization")

    class ScaledRod(HeatRod):
        def scales(self) -> dict:
            return {"length": 2.0, "time": 10.0, "value": 300.0}

    def make_tensor(n_fields: int = 1) -> PhysicalTensor:
        return PhysicalTensor(
            values=torch.rand(20, n_fields) * 300.0,
            coordinates=torch.rand(20, 2) * torch.tensor([2.0, 10.0]),
            units=UnitSystem(spatial="m", temporal="s", field="K"),
            coord_system=CoordinateSystem.CARTESIAN,
            domain=Domain(x_range=(0.0, 2.0), t_range=(0.0, 10.0)),
            metadata={},
        )

    problem = ScaledRod()
    tensor  = make_tensor()
    norm    = problem.to_normalized_units(tensor)

    check("coordinates are scaled to [0, 1]",
          norm.coordinates.min().item() >= 0.0 and norm.coordinates.max().item() <= 1.0)
    check("values are divided by the value scale",
          torch.allclose(norm.values, tensor.values / 300.0))
    check("metadata marks the tensor as normalized", norm.metadata.get("normalized") is True)
    check("input tensor is not modified", tensor.metadata.get("normalized") is None)

    back = problem.from_normalized_units(norm)
    check("round trip restores coordinates", torch.allclose(back.coordinates, tensor.coordinates))
    check("round trip restores values", torch.allclose(back.values, tensor.values))
    check("round trip drops the normalization metadata", "normalized" not in back.metadata)

    check("from_normalized_units on a plain tensor raises ValueError",
          raises(ValueError, lambda: problem.from_normalized_units(tensor)))
    check("normalizing twice raises ValueError",
          raises(ValueError, lambda: problem.to_normalized_units(norm)))

    class PerChannel(HeatRod):
        def scales(self) -> dict:
            return {"length": 1.0, "time": 1.0, "value": [300.0, 1.0]}

    t2    = make_tensor(n_fields=2)
    norm2 = PerChannel().to_normalized_units(t2)
    check("per-channel value scales",
          torch.allclose(norm2.values, t2.values / torch.tensor([300.0, 1.0])))


# ════════════════════════════════════════════════════════════════════════════
# Integration with the solver and the default loss
# ════════════════════════════════════════════════════════════════════════════

def test_solver_integration():
    section("Integration with AutogradPDESolver and PhysicsProblem.loss()")

    n = 80

    # One-field problem
    batch = make_batch(torch.rand(n, 2), torch.rand(n, 1),
                       {"k": 0.8, "rho_cp": 2.0, "q": 1.5})
    model  = BasePINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16, activation="tanh")
    solver = AutogradPDESolver(equation=REGISTRY.get("heat_conduction_source_1d"))

    out = solver.solve_with_grad(model, batch)
    check("heat: pde residual shape (N, 1)", out.residuals["pde"].shape == (n, 1))
    check("heat: pde residual is finite", torch.isfinite(out.residuals["pde"]).all().item())

    loss = out.residuals["pde"].pow(2).mean() + out.residuals["bc"].pow(2).mean()
    loss.backward()
    check("heat: backprop through the solver works",
          all(p.grad is not None for p in model.parameters()))

    # Two-field problem
    params2 = {"D_v": 0.1, "D_i": 1.0, "K0": 0.02, "K_iv": 5.0, "K_v": 0.3, "K_i": 0.8}
    batch2  = make_batch(torch.rand(n, 2), torch.rand(n, 2), params2)
    model2  = BasePINN(input_dim=2, output_dim=2, hidden_layers=2, hidden_size=16, activation="tanh")
    solver2 = AutogradPDESolver(equation=REGISTRY.get("point_defect_rate_theory_1d"))

    out2 = solver2.solve_with_grad(model2, batch2)
    check("rate theory: pde residual shape (N, 2)", out2.residuals["pde"].shape == (n, 2))
    check("rate theory: bc residual has 2 channels", out2.residuals["bc"].shape[1] == 2)

    # Default PhysicsProblem.loss() through the problem class
    coords = torch.rand(n, 2).requires_grad_(True)
    batch3 = make_batch(coords, torch.rand(n, 1))
    model3 = BasePINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16, activation="tanh")
    pred   = model3.forward(batch3).pred
    total  = HeatRod().loss(batch3, pred)
    check("PhysicsProblem.loss() is a finite scalar",
          total.dim() == 0 and torch.isfinite(total).item())
    total.backward()
    check("loss backpropagates into the model",
          all(p.grad is not None for p in model3.parameters()))


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("AIPlasma — Nuclear Material and Radiation Problems Test")
    print("=" * 60)

    start = time.perf_counter()

    test_registry()
    test_heat_conduction()
    test_fission_gas_booth()
    test_rate_theory()
    test_neutron_diffusion()
    test_bateman()
    test_radiative_loss()
    test_helper_functions()
    test_boundary_helpers()
    test_problem_classes()
    test_normalization()
    test_solver_integration()

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed | {elapsed:.2f}s")
    if FAILED == 0:
        print("TEST PASSED")
    else:
        print(f"TEST FAILED ✗ — {FAILED} test(s) failed")
    print("=" * 60)


if __name__ == "__main__":
    run()