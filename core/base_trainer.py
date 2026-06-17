from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from core.base_model import PhysicsModel, ModelOutput
from core.base_solver import PhysicsSolver, SolverOutput
from core.base_problem import PhysicsProblem
from data.preprocessing.feature_pipeline import FeatureBatch


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    epochs:     list[int]   = field(default_factory=list)
    extra:      dict        = field(default_factory=dict)


@dataclass
class TrainerConfig:
    max_epochs:      int   = 100
    val_frequency:   int   = 1        # validacija svake N epoha
    log_frequency:   int   = 10       # log svake N epoha
    distributed:     bool  = False
    devices:         list[int] = field(default_factory=lambda: [0])
    checkpoint_dir:  Optional[str] = None


class BaseTrainer(ABC):

    # ── Deklarativni callback registar ──────────────────────────

    callbacks: list = []

    def __init__(
        self,
        model:    PhysicsModel,
        problem:  PhysicsProblem,
        solver:   PhysicsSolver,
        config:   TrainerConfig,
    ):
        self.model   = model
        self.problem = problem
        self.solver  = solver
        self.config  = config
        self.history = TrainingHistory()
        self._setup_device()

    # ── Korisnik MORA da implementira ───────────────────────────

    @abstractmethod
    def train_step(self, batch: FeatureBatch) -> Tensor:
        """Jedan korak treninga. Vraća loss za tu iteraciju."""
        ...

    @abstractmethod
    def val_step(self, batch: FeatureBatch) -> Tensor:
        """Jedan korak validacije. Vraća loss za tu iteraciju."""
        ...

    # ── Framework pruža — fit() je sealed ───────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   Optional[DataLoader] = None,
    ) -> TrainingHistory:

        self._fire("on_train_start")

        for epoch in range(1, self.config.max_epochs + 1):
            self._fire("on_epoch_start", epoch=epoch)

            # ── Training ────────────────────────────────────────
            self.model.train()
            train_losses = []
            for batch in train_loader:
                batch = self._to_device(batch)
                loss  = self.train_step(batch)
                train_losses.append(loss.item())
            epoch_train_loss = sum(train_losses) / len(train_losses)

            # ── Validacija ──────────────────────────────────────
            epoch_val_loss = None
            if val_loader and epoch % self.config.val_frequency == 0:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = self._to_device(batch)
                        loss  = self.val_step(batch)
                        val_losses.append(loss.item())
                epoch_val_loss = sum(val_losses) / len(val_losses)

            # ── History ─────────────────────────────────────────
            self.history.epochs.append(epoch)
            self.history.train_loss.append(epoch_train_loss)
            if epoch_val_loss is not None:
                self.history.val_loss.append(epoch_val_loss)

            # ── Logging ─────────────────────────────────────────
            if epoch % self.config.log_frequency == 0:
                self._log(epoch, epoch_train_loss, epoch_val_loss)

            self._fire("on_epoch_end", epoch=epoch, history=self.history)

        self._fire("on_train_end", history=self.history)
        return self.history

    # ── Interni framework metodi ─────────────────────────────────

    def _setup_device(self) -> None:
        if self.config.distributed:
            raise NotImplementedError("Distributed training coming soon.")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _to_device(self, batch: FeatureBatch) -> FeatureBatch:
        return FeatureBatch(
            coords           = batch.coords.to(self.device),
            fields           = batch.fields.to(self.device),
            boundary_mask    = batch.boundary_mask.to(self.device),
            collocation_mask = batch.collocation_mask.to(self.device),
            physics_params   = batch.physics_params,
            fidelity_level   = batch.fidelity_level,
            fidelity_weight  = batch.fidelity_weight,
        )

    def _fire(self, event: str, **kwargs) -> None:
        for cb in self.callbacks:
            if hasattr(cb, event):
                getattr(cb, event)(trainer=self, **kwargs)

    def _log(
        self,
        epoch:     int,
        train_loss: float,
        val_loss:  Optional[float],
    ) -> None:
        val_str = f"val_loss={val_loss:.6f}" if val_loss is not None else "val_loss=N/A"
        print(f"[Epoch {epoch:>4}] train_loss={train_loss:.6f}  {val_str}")