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

    # ── User must implement ───────────────────────────

    @abstractmethod
    def solve(self, output: ModelOutput, batch: FeatureBatch) -> SolverOutput:
        """Post-processing and evaluation without autograd-a."""
        ...

    @abstractmethod
    def solve_with_grad(self, model: PhysicsModel, batch: FeatureBatch) -> SolverOutput:
        """PDE residual during training - autograd without model."""
        ...

    # ── Framework has ─────────────────────────────────────────

    def timed_solve(self, output: ModelOutput, batch: FeatureBatch) -> SolverOutput:
        """solve() sa automatskim merenjem wall_time."""
        start = time.perf_counter()
        result = self.solve(output, batch)
        result.solver_info.wall_time = time.perf_counter() - start
        return result

    def timed_solve_with_grad(self, model: PhysicsModel, batch: FeatureBatch) -> SolverOutput:
        start = time.perf_counter()
        result = self.solve_with_grad(model, batch)
        result.solver_info.wall_time = time.perf_counter() - start
        return result

    def validate_output(self, output: SolverOutput) -> None:
        if not output.quantities:
            raise ValueError("SolverOutput.quantities ne sme biti prazan.")
        if not output.residuals:
            raise ValueError("SolverOutput.residuals ne sme biti prazan.")


class ODESolver(PhysicsSolver, ABC):
    """Base solver for time dependent(ODE system)."""

    @abstractmethod
    def step(self, model: PhysicsModel, batch: FeatureBatch, dt: float) -> SolverOutput:
        """one step of integration."""
        ...

#Base for Solver for Partial Differential Equations
class PDESolver(PhysicsSolver, ABC):
    @abstractmethod
    def compute_derivatives(self, pred: Tensor, coords: Tensor) -> dict[str, Tensor]:
       pass