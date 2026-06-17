from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time

import torch
from torch import Tensor
from data.preprocessing.feature_pipeline import FeatureBatch
from core.base_model import PhysicsModel, ModelOutput


@dataclass
class SolverInfo:
    solver_type: str
    converged: Optional[bool] = None
    iterations: Optional[int] = None
    wall_time: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class SolverOutput:
    quantities: dict[str, Tensor]
    residuals: dict[str, Tensor]
    solver_info: SolverInfo


class PhysicsSolver(ABC):

    # ── Korisnik MORA da implementira ───────────────────────────

    @abstractmethod
    def solve(self, output: ModelOutput, batch: FeatureBatch) -> SolverOutput:
        """Post-processing i evaluacija bez autograd-a."""
        ...

    @abstractmethod
    def solve_with_grad(self, model: PhysicsModel, batch: FeatureBatch) -> SolverOutput:
        """PDE residual tokom treninga — autograd kroz model."""
        ...

    # ── Framework pruža ─────────────────────────────────────────

    def timed_solve(self, output: ModelOutput, batch: FeatureBatch) -> SolverOutput:
        """solve() sa automatskim merenjem wall_time."""
        start = time.perf_counter()
        result = self.solve(output, batch)
        result.solver_info.wall_time = time.perf_counter() - start
        return result

    def timed_solve_with_grad(self, model: PhysicsModel, batch: FeatureBatch) -> SolverOutput:
        """solve_with_grad() sa automatskim merenjem wall_time."""
        start = time.perf_counter()
        result = self.solve_with_grad(model, batch)
        result.solver_info.wall_time = time.perf_counter() - start
        return result

    def validate_output(self, output: SolverOutput) -> None:
        """Proverava da SolverOutput ima očekivane ključeve."""
        if not output.quantities:
            raise ValueError("SolverOutput.quantities ne sme biti prazan.")
        if not output.residuals:
            raise ValueError("SolverOutput.residuals ne sme biti prazan.")


class ODESolver(PhysicsSolver, ABC):
    """Baza za solver-e vremenski zavisnih problema (ODE sistemi)."""

    @abstractmethod
    def step(self, model: PhysicsModel, batch: FeatureBatch, dt: float) -> SolverOutput:
        """Jedan vremenski korak integracije."""
        ...


class PDESolver(PhysicsSolver, ABC):
    """Baza za solver-e prostorno-vremenskih problema (PDE sistemi)."""

    @abstractmethod
    def compute_derivatives(self, pred: Tensor, coords: Tensor) -> dict[str, Tensor]:
        """Prostorne i vremenske derivacije kroz autograd."""
        ...