from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import Tensor

from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import Tensor
from data.preprocessing.feature_pipeline import FeatureBatch
class PhysicsProblem(ABC):

    # User must Define

    @abstractmethod
    def pde_residual(self, batch: FeatureBatch, pred: Tensor) -> Tensor:
        """Ostatak PDE-a u collocation tačkama."""
        ...

    @abstractmethod
    def boundary_conditions(self, batch: FeatureBatch, pred: Tensor) -> Tensor:
        """Ostatak boundary conditions."""
        ...

    # ── Korisnik MOŽE da override-uje ───────────────────────────

    def loss(self, batch: FeatureBatch, pred: Tensor) -> Tensor:
        """Default multi-fidelity loss. Override for custom behaviour."""
        w = self.fidelity_weights()
        i = batch.fidelity_level

        data_loss = torch.mean((pred - batch.fields) ** 2)
        pde_loss  = torch.mean(self.pde_residual(batch, pred) ** 2)
        bc_loss   = torch.mean(self.boundary_conditions(batch, pred) ** 2)

        return w[i] * (self.lambda1 * data_loss
                     + self.lambda2 * pde_loss
                     + self.lambda3 * bc_loss)

    def physics_params(self) -> dict:
        return {}

    def fidelity_weights(self) -> list[float]:
        """Default: uniform weights."""
        levels = self._infer_fidelity_levels()
        return [1.0 / levels] * levels

    # ── Framework interno ────────────────────────────────────────

    def validate(self, batch: FeatureBatch) -> None:
        ...

    def compile(self) -> "CompiledProblem":
        ...

    # ── Loss coeficient (class-level defaults) ─────────────────

    lamda1: float = 1.0   # data loss weight
    lamda2: float = 1.0   # PDE loss weight
    lambda3: float = 1.0   # BC loss weight