from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import Tensor

from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import Tensor
from data.preprocessing.feature_pipeline import FeatureBatch


class CompiledProblem:
    def __init__(self,problem,model,data_source,adaptive_weights,weight_scheduler):
        self.problem = problem
        self.model = model
        self.data_source = data_source
        self.adaptive_weights = adaptive_weights
        self.weight_scheduler = weight_scheduler


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

        return w[i] * (self.lamda1 * data_loss
                     + self.lamda2 * pde_loss
                     + self.lamda3 * bc_loss)
    def physics_params(self) -> dict:
        return {}
    def fidelity_weights(self) -> list[float]:
        """Default: uniform weights."""
        levels = self._infer_fidelity_levels()
        return [1.0 / levels] * levels

    # ── Framework interno ────────────────────────────────────────

    def validate(self, batch: FeatureBatch) -> None:
        """
        Proverava konzistentnost FeatureBatch-a pre treninga.
        Baca ValueError ako nešto nije ispravno.
        """
        # 1. boundary_mask i collocation_mask moraju biti disjunktni
        overlap = batch.boundary_mask & batch.collocation_mask
        if overlap.any():
            raise ValueError(
                "boundary_mask i collocation_mask se preklapaju. "
                "Tačka ne može biti i BC i collocation."
            )

        # 2. fidelity_weights suma mora biti blizu 1.0
        weights = self.fidelity_weights()
        total = sum(weights)
        if not (0.99 < total < 1.01):
            raise ValueError(
                f"fidelity_weights suma mora biti 1.0, dobijeno: {total:.4f}"
            )

        # 3. physics_params ne sme biti prazan ako PDE to zahteva
        # (PlasmaPhysicsProblem će ovo proširiti sa domenski specifičnom validacijom)

    def compile(
            self,
            model,
            data_source,
            adaptive_weights: bool = False,
            weight_scheduler=None,
    ) -> "CompiledProblem":
        """
        Spaja problem, model i data_source u trainable objekat.

        Args:
            model: PhysicsModel instanca.
            data_source: DataSource instanca ili lista DataSource instanci.
            adaptive_weights: Ako True, koristi GradNorm za automatsko
                             balansiranje loss komponenti. Default: False.
            weight_scheduler: Custom LossWeightScheduler. Override-uje
                             adaptive_weights ako je prosleđen.

        Priority: weight_scheduler > adaptive_weights > statični w_data/w_pde/w_bc
        """
        return CompiledProblem(
            problem=self,
            model=model,
            data_source=data_source,
            adaptive_weights=adaptive_weights,
            weight_scheduler=weight_scheduler,
        )

    # ── Loss coeficient (class-level defaults) ─────────────────

    lamda1: float = 1.0   # data loss weight
    lamda2: float = 1.0   # PDE loss weight
    lamda3: float = 1.0   # BC loss weight