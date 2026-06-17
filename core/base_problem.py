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

    # ── Korisnik MORA da definiše ────────────────────────────────

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
        """Default multi-fidelity loss. Override za custom ponašanje."""
        w = self.fidelity_weights()
        i = batch.fidelity_level

        data_loss = torch.mean((pred - batch.fields) ** 2)
        pde_loss  = torch.mean(self.pde_residual(batch, pred) ** 2)
        bc_loss   = torch.mean(self.boundary_conditions(batch, pred) ** 2)

        return w[i] * (self.λ1 * data_loss
                     + self.λ2 * pde_loss
                     + self.λ3 * bc_loss)

    def physics_params(self) -> dict:
        return {}

    def fidelity_weights(self) -> list[float]:
        """Default: uniformne težine."""
        levels = self._infer_fidelity_levels()
        return [1.0 / levels] * levels

    # ── Framework interno ────────────────────────────────────────

    def validate(self, batch: FeatureBatch) -> None:
        ...

    def compile(self) -> "CompiledProblem":
        ...

    # ── Loss koeficijenti (class-level defaults) ─────────────────

    λ1: float = 1.0   # data loss weight
    λ2: float = 1.0   # PDE loss weight
    λ3: float = 1.0   # BC loss weight