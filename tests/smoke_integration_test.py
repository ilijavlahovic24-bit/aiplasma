#Test 2


#Generating syntetic data
#THis test is funtionaly same as smoke test, but it uses framework classes and interfaces
# instead of locally made ones


"""
test_integration_smoke.py
Integration smoke test — verifies that all core interfaces
work together using the actual project classes.

Unlike smoke_test.py which uses isolated inline code,
this test imports and uses the real framework classes.
If a file was changed and broke an interface, this test catches it.

Run:
    python tests/test_integration_smoke.py
"""

import sys
import os
import math
import time
import torch

# sys.path.insert MORA biti prvi
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Sada svi importi koriste paketne putanje
from data.preprocessing.feature_pipeline import FeatureBatch
from data_parser import DataSourceConfig, SyntheticDataSource
from data.data_pipeline import DataPipeline
from core.material_problem import NuclearMaterialProblem
from core.radiation_problem import RadiationProblem
from models.pinn.base_pinn import BasePINN
from models.pinn.residual_pinn import ResidualPINN
from trainers.callbacks.mf_trainer import MultiFidelityConfig, MultiFidelityTrainer


from data.sources.physical_tensor import PhysicalTensor, UnitSystem, Domain, CoordinateSystem

from data.preprocessing.data_transformer import DataTransformer, TransformerConfig, FidelityAssigner,FeatureBatchDataset
from core.base_model import ModelOutput
from core.base_problem import PhysicsProblem, CompiledProblem
from core.base_solver import SolverOutput, SolverInfo
from core.base_trainer import TrainerConfig, TrainingHistory

from core.plasma_problem import PlasmaPhysicsProblem

from models.pinn.multi_fidelity_pinn import MultiFidelityPINN
from models.bayesian.bayesian_nn import BayesianNN

from models.ensemble.ensemble_model import EnsembleModel
from solvers.physics_constraints import REGISTRY, HeatEquation1D, DriftDiffusion1D, HasegawaWakatani
from solvers.pde_solver import AutogradPDESolver
from solvers.ode_solver import EulerODESolver, RK4ODESolver, AdaptiveODESolver
from trainers.pinn_trainer import PINNTrainer
from trainers.bayesian_trainer import BayesianTrainer

from trainers.callbacks.base_callback import Callback
from trainers.callbacks.check_pointing import CheckPointing
from trainers.callbacks.early_stopping import EarlyStopping


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
        msg = f"  ✗ {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def section(title: str) -> None:
    print(f"\n[{title}]")


def make_batch(n: int = 100, fidelity_level: int = 0) -> FeatureBatch:
    """Creates a minimal FeatureBatch for testing."""
    x      = torch.rand(n)
    t      = torch.rand(n)
    coords = torch.stack([x, t], dim=1)
    u      = torch.sin(math.pi * x) * torch.exp(torch.tensor(-0.01 * math.pi ** 2) * t)
    fields = u.unsqueeze(1)
    bm     = (x < 0.05) | (x > 0.95)
    cm     = ~bm
    return FeatureBatch(
        coords=coords, fields=fields,
        boundary_mask=bm, collocation_mask=cm,
        physics_params={"alpha": 0.01, "D": 0.1, "v": 0.5},
        fidelity_level=fidelity_level, fidelity_weight=1.0,
    )


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

def test_data_layer():
    section("Data Layer")

    # FeatureBatch
    batch = make_batch()
    check("FeatureBatch created", batch.coords.shape == (100, 2))
    check("FeatureBatch masks disjoint", not (batch.boundary_mask & batch.collocation_mask).any())

    # FeatureBatchDataset
    batches = [make_batch(50, i) for i in range(3)]
    dataset = FeatureBatchDataset(batches)
    check("FeatureBatchDataset __len__", len(dataset) == 3)
    check("FeatureBatchDataset __getitem__", isinstance(dataset[0], FeatureBatch))

    train_ds, val_ds = dataset.split(val_ratio=0.33)
    check("FeatureBatchDataset split returns two datasets",
          isinstance(train_ds, FeatureBatchDataset) and isinstance(val_ds, FeatureBatchDataset))

    # PhysicalTensor
    coords = torch.rand(50, 2)
    values = torch.rand(50, 1)
    domain = Domain(x_range=(0.0, 1.0), t_range=(0.0, 1.0))
    units  = UnitSystem(spatial="m", temporal="s", field="normalized")
    pt     = PhysicalTensor(
        values=values, coordinates=coords,
        units=units, coord_system=CoordinateSystem.CARTESIAN,
        domain=domain, metadata={"source": "test"},
    )
    check("PhysicalTensor created", pt.n_points() == 50)
    check("PhysicalTensor n_fields", pt.n_fields() == 1)
    check("PhysicalTensor spatial_resolution > 0", pt.spatial_resolution() > 0)

    # FidelityAssigner
    pt2 = PhysicalTensor(
        values=torch.rand(200, 1), coordinates=torch.rand(200, 2),
        units=units, coord_system=CoordinateSystem.CARTESIAN,
        domain=domain,
    )
    assigner = FidelityAssigner()
    levels   = assigner.assign([pt, pt2])
    check("FidelityAssigner returns correct length", len(levels) == 2)
    check("FidelityAssigner deterministic", levels == assigner.assign([pt, pt2]))
    check("FidelityAssigner assigns different levels", levels[0] != levels[1])

    # SyntheticDataSource
    config = DataSourceConfig(
        coord_system=CoordinateSystem.CARTESIAN,
        units=units,
        domain=domain,
    )
    source = SyntheticDataSource(
        solution_fn=lambda x, t: torch.sin(math.pi * x),
        config=config,
        n_points=100,
    )
    check("SyntheticDataSource validates", source.validate())
    tensor = source.load()
    check("SyntheticDataSource load returns PhysicalTensor",
          isinstance(tensor, PhysicalTensor))
    check("SyntheticDataSource correct shape", tensor.n_points() == 100)

    # DataTransformer
    transformer = DataTransformer()
    fb_list     = transformer.transform([tensor])
    print("fb type:", type(fb_list[0]))
    print("fb module:", type(fb_list[0]).__module__)
    print("FeatureBatch module:", FeatureBatch.__module__)
    check("DataTransformer returns list of FeatureBatch", len(fb_list) == 1)
    check("DataTransformer output is FeatureBatch", isinstance(fb_list[0], FeatureBatch))


def test_pde_registry():
    section("PDE Registry")

    check("REGISTRY contains heat_equation_1d", "heat_equation_1d" in REGISTRY)
    check("REGISTRY contains drift_diffusion_1d", "drift_diffusion_1d" in REGISTRY)
    check("REGISTRY contains hasegawa_wakatani", "hasegawa_wakatani" in REGISTRY)
    check("REGISTRY list_all returns 3", len(REGISTRY.list_all()) >= 3)

    eq = REGISTRY.get("heat_equation_1d")
    check("HeatEquation1D name", eq.name() == "heat_equation_1d")
    check("HeatEquation1D expected_params", eq.expected_params() == ["alpha"])
    check("HeatEquation1D validate_params", eq.validate_params({"alpha": 0.01}))

    try:
        eq.validate_params({})
        check("HeatEquation1D missing param raises", False)
    except ValueError:
        check("HeatEquation1D missing param raises", True)

    try:
        REGISTRY.get("nonexistent_equation")
        check("REGISTRY.get raises KeyError for unknown", False)
    except KeyError:
        check("REGISTRY.get raises KeyError for unknown", True)

    hw = REGISTRY.get("hasegawa_wakatani")
    check("HasegawaWakatani is stub", hw.expected_params() == ["D", "C", "nu"])
    try:
        hw.residual(None, None, {})
        check("HasegawaWakatani.residual raises NotImplementedError", False)
    except NotImplementedError:
        check("HasegawaWakatani.residual raises NotImplementedError", True)


def test_models():
    section("Models")

    batch = make_batch()

    # BasePINN
    pinn   = BasePINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16)
    output = pinn.forward(batch)
    check("BasePINN forward returns ModelOutput", isinstance(output, ModelOutput))
    check("BasePINN pred shape", output.pred.shape == (100, 1))
    check("BasePINN uncertainty is None", output.uncertainty is None)
    check("BasePINN summary contains parameters", "parameters" in pinn.summary())

    # ResidualPINN
    rpinn  = ResidualPINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16)
    output = rpinn.forward(batch)
    check("ResidualPINN forward returns ModelOutput", isinstance(output, ModelOutput))
    check("ResidualPINN pred shape", output.pred.shape == (100, 1))

    # MultiFidelityPINN
    mfpinn = MultiFidelityPINN(input_dim=2, output_dim=1, n_fidelity_levels=3, hidden_layers=2)
    output = mfpinn.forward(batch)
    check("MultiFidelityPINN forward returns ModelOutput", isinstance(output, ModelOutput))
    check("MultiFidelityPINN pred shape", output.pred.shape == (100, 1))
    check("MultiFidelityPINN summary mentions fidelity", "fidelity" in mfpinn.summary().lower())

    # BayesianNN
    bayes  = BayesianNN(input_dim=2, output_dim=1, hidden_size=16, n_samples=5)
    bayes.eval()
    output = bayes.forward(batch)
    check("BayesianNN forward returns ModelOutput", isinstance(output, ModelOutput))
    check("BayesianNN pred shape", output.pred.shape == (100, 1))
    check("BayesianNN uncertainty not None in eval", output.uncertainty is not None)
    check("BayesianNN uncertainty >= 0", (output.uncertainty >= 0).all())

    mean, std = bayes.sample_predictions(batch)
    check("BayesianNN sample_predictions mean shape", mean.shape == (100, 1))
    check("BayesianNN sample_predictions std shape", std.shape == (100, 1))

    # EnsembleModel
    p1       = BasePINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16)
    p2       = ResidualPINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16)
    ensemble = EnsembleModel(models=[p1, p2], aggregation="mean")
    output   = ensemble.forward(batch)
    check("EnsembleModel forward returns ModelOutput", isinstance(output, ModelOutput))
    check("EnsembleModel pred shape", output.pred.shape == (100, 1))
    check("EnsembleModel uncertainty not None", output.uncertainty is not None)

    ensemble.add_model(BasePINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16))
    check("EnsembleModel add_model", len(ensemble.models) == 3)

    try:
        EnsembleModel(models=[])
        check("EnsembleModel empty raises ValueError", False)
    except ValueError:
        check("EnsembleModel empty raises ValueError", True)


def test_domain_hierarchy():
    section("Domain Hierarchy")

    # PlasmaPhysicsProblem stubs
    check("RadiationProblem exists", RadiationProblem is not None)
    check("NuclearMaterialProblem exists", NuclearMaterialProblem is not None)

    rp = RadiationProblem()
    try:
        rp.pde_residual(None, None)
        check("RadiationProblem.pde_residual raises NotImplementedError", False)
    except NotImplementedError:
        check("RadiationProblem.pde_residual raises NotImplementedError", True)

    mp = NuclearMaterialProblem()
    try:
        mp.pde_residual(None, None)
        check("NuclearMaterialProblem.pde_residual raises NotImplementedError", False)
    except NotImplementedError:
        check("NuclearMaterialProblem.pde_residual raises NotImplementedError", True)


def test_solvers():
    section("Solvers")

    batch = make_batch()

    # AutogradPDESolver — solve() mode
    solver = AutogradPDESolver(equation=REGISTRY.get("heat_equation_1d"))
    pinn   = BasePINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16)
    pinn.eval()

    with torch.no_grad():
        output = pinn.forward(batch)

    solver_out = solver.solve(output, batch)
    check("AutogradPDESolver.solve returns SolverOutput",
          isinstance(solver_out, SolverOutput))
    check("AutogradPDESolver.solve has 'u' in quantities",
          "u" in solver_out.quantities)

    # AutogradPDESolver — solve_with_grad() mode
    solver_out = solver.solve_with_grad(pinn, batch)
    check("AutogradPDESolver.solve_with_grad returns SolverOutput",
          isinstance(solver_out, SolverOutput))
    check("AutogradPDESolver has pde residual", "pde" in solver_out.residuals)
    check("AutogradPDESolver has bc residual", "bc" in solver_out.residuals)
    check("AutogradPDESolver has du_dx", "du_dx" in solver_out.quantities)
    check("AutogradPDESolver has du_dt", "du_dt" in solver_out.quantities)

    # EulerODESolver
    euler      = EulerODESolver()
    solver_out = euler.solve_with_grad(pinn, batch)
    check("EulerODESolver.solve_with_grad returns SolverOutput",
          isinstance(solver_out, SolverOutput))
    check("EulerODESolver has u_next", "u_next" in solver_out.quantities)

    # RK4ODESolver
    rk4        = RK4ODESolver()
    solver_out = rk4.solve_with_grad(pinn, batch)
    check("RK4ODESolver.solve_with_grad returns SolverOutput",
          isinstance(solver_out, SolverOutput))
    check("RK4ODESolver has u_next", "u_next" in solver_out.quantities)
    check("RK4ODESolver iterations = 4",
          solver_out.solver_info.iterations == 4)

    # SolverInfo
    info = SolverInfo(solver_type="test", converged=True, iterations=1)
    check("SolverInfo created", info.solver_type == "test")


def test_callbacks():
    section("Callbacks")

    # Callback base
    cb = Callback()
    try:
        cb.on_train_start(trainer=None)
        cb.on_epoch_start(trainer=None, epoch=1)
        cb.on_epoch_end(trainer=None, epoch=1, history=None)
        cb.on_train_end(trainer=None, history=None)
        check("Callback base methods callable without error", True)
    except Exception as e:
        check("Callback base methods callable without error", False, str(e))

    # EarlyStopping
    es = EarlyStopping(patience=3, min_delta=1e-4)
    check("EarlyStopping created", es.patience == 3)

    # CheckPointing
    cp = CheckPointing(save_dir="/tmp/aiplasma_test_checkpoints/")
    check("CheckPointing created", cp.save_dir == "/tmp/aiplasma_test_checkpoints/")


def test_compiled_problem():
    section("CompiledProblem")

    pinn   = BasePINN(input_dim=2, output_dim=1)
    source = SyntheticDataSource(
        solution_fn=lambda x, t: torch.sin(math.pi * x),
        config=DataSourceConfig(
            coord_system=CoordinateSystem.CARTESIAN,
            units=UnitSystem(spatial="m", temporal="s", field="normalized"),
            domain=Domain(x_range=(0.0, 1.0), t_range=(0.0, 1.0)),
        ),
        n_points=50,
    )

    # Use a simple concrete problem for testing compile()
    class _TestProblem(PhysicsProblem):
        def pde_residual(self, batch, pred):
            return torch.zeros_like(pred)
        def boundary_conditions(self, batch, pred):
            return pred[batch.boundary_mask] - batch.fields[batch.boundary_mask]

    problem  = _TestProblem()
    compiled = problem.compile(model=pinn, data_source=source)

    check("CompiledProblem created", isinstance(compiled, CompiledProblem))
    check("CompiledProblem has model", compiled.model is pinn)
    check("CompiledProblem data_source is list", isinstance(compiled.data_source, list))
    check("CompiledProblem describe() runs", len(compiled.describe()) > 0)


def test_trainers():
    section("Trainers")

    batch   = make_batch()
    batches = [make_batch(50, i) for i in range(2)]

    # PINNTrainer — minimal train
    class _HeatProblem(PhysicsProblem):
        w_data, w_pde, w_bc = 1.0, 0.0, 1.0
        def pde_residual(self, b, p):
            return torch.zeros_like(p)
        def boundary_conditions(self, b, p):
            return p[b.boundary_mask] - b.fields[b.boundary_mask]
        def fidelity_weights(self):
            return [0.5, 0.5]

    problem = _HeatProblem()
    pinn    = BasePINN(input_dim=2, output_dim=1, hidden_layers=2, hidden_size=16)
    solver  = AutogradPDESolver()
    config  = TrainerConfig(max_epochs=3, val_frequency=1, log_frequency=3)

    trainer = PINNTrainer(model=pinn, problem=problem, solver=solver, config=config)
    check("PINNTrainer created", trainer.device is not None)

    history = trainer.fit(train_batches=batches, val_batches=[batch])
    check("PINNTrainer fit returns TrainingHistory", isinstance(history, TrainingHistory))
    check("PINNTrainer history has epochs", len(history.epochs) == 3)
    check("PINNTrainer loss decreased or stayed",
          history.train_loss[-1] <= history.train_loss[0] * 1.5)

    # BayesianTrainer
    bayes_model = BayesianNN(input_dim=2, output_dim=1, hidden_size=16, n_samples=3)
    b_trainer   = BayesianTrainer(
        model=bayes_model, problem=problem, solver=solver, config=config
    )
    check("BayesianTrainer created", b_trainer is not None)

    b_history = b_trainer.fit(train_batches=batches, val_batches=[batch])
    check("BayesianTrainer fit returns history", isinstance(b_history, TrainingHistory))
    check("BayesianTrainer uncertainty_history populated",
          len(b_trainer.uncertainty_history()) > 0)

    try:
        BayesianTrainer(model=pinn, problem=problem, solver=solver, config=config)
        check("BayesianTrainer rejects non-BayesianNN", False)
    except ValueError:
        check("BayesianTrainer rejects non-BayesianNN", True)

    # MultiFidelityTrainer
    mf_model   = MultiFidelityPINN(input_dim=2, output_dim=1, n_fidelity_levels=2,
                                    hidden_layers=2, hidden_size=16)
    mf_config  = MultiFidelityConfig(curriculum=False, n_fidelity_levels=2)
    mf_trainer = MultiFidelityTrainer(
        model=mf_model, problem=problem, solver=solver,
        config=config, mf_config=mf_config,
    )
    check("MultiFidelityTrainer created", mf_trainer is not None)
    check("MultiFidelityTrainer active_levels", len(mf_trainer.active_levels()) == 2)

    mf_history = mf_trainer.fit(train_batches=batches, val_batches=[batch])
    check("MultiFidelityTrainer fit returns history", isinstance(mf_history, TrainingHistory))


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run_integration_smoke():
    print("=" * 60)
    print("AIPlasma Integration Smoke Test")
    print("Verifies all core interfaces using real framework classes")
    print("=" * 60)

    start = time.perf_counter()

    test_data_layer()
    test_pde_registry()
    test_models()
    test_domain_hierarchy()
    test_solvers()
    test_callbacks()
    test_compiled_problem()
    test_trainers()

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    print(f"REZULTATI: {PASSED} passed, {FAILED} failed | {elapsed:.2f}s")
    print("=" * 60)

    if FAILED == 0:
        print("INTEGRATION SMOKE TEST PROSAO ✓")
    else:
        print(f"INTEGRATION SMOKE TEST NIJE PROSAO ✗ — {FAILED} test(ova) palo")
    print("=" * 60)


if __name__ == "__main__":
    run_integration_smoke()