"""
mf_trainer.py
Multi-Fidelity Trainer for AIPlasma framework.

Lives in: trainers/mf_trainer.py

Note: moved from trainers/callbacks/mf_trainer.py to trainers/
as per implementation plan - MultiFidelityTrainer is a trainer,
not a callback.
"""

import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional

from core.base_trainer import BaseTrainer, TrainerConfig
from core.base_model import PhysicsModel
from core.base_problem import PhysicsProblem
from core.base_solver import PhysicsSolver
from data.preprocessing.feature_pipeline import FeatureBatch
from trainers.pinn_trainer import PINNTrainer


# ════════════════════════════════════════════════════════════════════════════
# MultiFidelityConfig
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MultiFidelityConfig:
    """
    Configuration for multi-fidelity training behavior.

    Args:
        curriculum:                 If True, enables curriculum learning -
                                    training starts with low-fidelity data
                                    and gradually introduces higher levels.
                                    Default: False (uniform training).
        curriculum_epochs_per_level: Number of epochs before introducing
                                    the next fidelity level. Only used
                                    when curriculum=True.
        n_fidelity_levels:          Total number of fidelity levels.
                                    Must match DataPipeline configuration.

    Example (uniform - default):
        config = MultiFidelityConfig()

    Example (curriculum):
        config = MultiFidelityConfig(
            curriculum=True,
            curriculum_epochs_per_level=30,
            n_fidelity_levels=3,
        )
    """
    curriculum:                  bool = False
    curriculum_epochs_per_level: int  = 30
    n_fidelity_levels:           int  = 3


# ════════════════════════════════════════════════════════════════════════════
# MultiFidelityTrainer
# ════════════════════════════════════════════════════════════════════════════

class MultiFidelityTrainer(PINNTrainer):
    """
    Trainer for multi-fidelity PINN training.

    Extends PINNTrainer with fidelity-aware loss weighting and
    optional curriculum learning (ADR-012).

    Default behavior (curriculum=False):
        All fidelity levels trained uniformly every epoch.
        Loss is weighted by batch.fidelity_weight from FeatureBatch.

    Optional curriculum learning (curriculum=True):
        Training starts with low-fidelity data only.
        Higher fidelity levels are introduced every
        curriculum_epochs_per_level epochs.

    Args:
        model:     PhysicsModel instance (preferably MultiFidelityPINN).
        problem:   PhysicsProblem instance.
        solver:    PhysicsSolver instance.
        config:    TrainerConfig with training hyperparameters.
        mf_config: MultiFidelityConfig. Default: uniform training.
        lr:        Learning rate. Default: 1e-3.
        scheduler: If True, uses ReduceLROnPlateau. Default: True.

    Example:
        trainer = MultiFidelityTrainer(
            model=MultiFidelityPINN(input_dim=2, output_dim=1),
            problem=TokamakTurbulence(),
            solver=AutogradPDESolver(equation=REGISTRY.get("hasegawa_wakatani")),
            config=TrainerConfig(max_epochs=200),
            mf_config=MultiFidelityConfig(curriculum=True),
        )
        history = trainer.fit(train_batches, val_batches)
    """

    callbacks: list = []

    def __init__(
        self,
        model:     PhysicsModel,
        problem:   PhysicsProblem,
        solver:    PhysicsSolver,
        config:    TrainerConfig,
        mf_config: Optional[MultiFidelityConfig] = None,
        lr:        float = 1e-3,
        scheduler: bool  = True,
    ):
        super().__init__(
            model=model,
            problem=problem,
            solver=solver,
            config=config,
            lr=lr,
            scheduler=scheduler,
        )
        self.mf_config       = mf_config or MultiFidelityConfig()
        self._active_levels  = self._initial_active_levels()

    # ── Abstract implementations ─────────────────────────────────────────────

    def train_step(self, batch: FeatureBatch) -> Tensor:
        """
        Single multi-fidelity training step.

        Skips batches whose fidelity_level is not yet active
        when curriculum learning is enabled.
        Loss is weighted by batch.fidelity_weight.

        Args:
            batch: FeatureBatch with fidelity_level and fidelity_weight.

        Returns:
            Loss tensor for this step, or zero tensor if batch is skipped.
        """
        # Curriculum: skip batch if its fidelity level is not yet active
        if (self.mf_config.curriculum and
                batch.fidelity_level not in self._active_levels):
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        self.optimizer.zero_grad()

        output = self.model.forward(batch)

        # fidelity_weight is already embedded in problem.loss()
        # via batch.fidelity_weight - no additional weighting needed
        loss = self.problem.loss(batch, output.pred)

        loss.backward()
        self.optimizer.step()

        return loss

    def val_step(self, batch: FeatureBatch) -> Tensor:
        """
        Single multi-fidelity validation step.

        Evaluates on all fidelity levels regardless of curriculum
        state - validation always uses the full dataset.

        Args:
            batch: FeatureBatch at current validation step.

        Returns:
            Loss tensor for this step.
        """
        output = self.model.forward(batch)
        return self.problem.loss(batch, output.pred)

    # ── Curriculum learning ───────────────────────────────────────────────────

    def _schedule_fidelity(self, epoch: int) -> None:
        """
        Updates active fidelity levels based on current epoch.

        No-op when curriculum=False (ADR-012 default).
        When curriculum=True: introduces one new fidelity level
        every curriculum_epochs_per_level epochs.

        Args:
            epoch: Current training epoch (1-indexed).
        """
        if not self.mf_config.curriculum:
            return

        epl = self.mf_config.curriculum_epochs_per_level
        n   = self.mf_config.n_fidelity_levels

        # Number of levels to activate based on current epoch
        n_active = min(n, (epoch - 1) // epl + 1)
        self._active_levels = set(range(n_active))

    def _initial_active_levels(self) -> set:
        """
        Returns initial set of active fidelity levels.

        Curriculum: only level 0 active at start.
        Uniform: all levels active from epoch 1.
        """
        if self.mf_config.curriculum:
            return {0}
        return set(range(self.mf_config.n_fidelity_levels))

    # ── Scheduler + curriculum hook ───────────────────────────────────────────

    def _fire(self, event: str, **kwargs) -> None:
        """
        Extends PINNTrainer._fire() to update curriculum schedule.
        """
        super()._fire(event, **kwargs)

        if event == "on_epoch_end":
            epoch = kwargs.get("epoch", 0)
            self._schedule_fidelity(epoch)

    # ── Status ───────────────────────────────────────────────────────────────

    def active_levels(self) -> set:
        """
        Returns currently active fidelity levels.

        Useful for logging and debugging curriculum progress.

        Returns:
            Set of active fidelity level integers.
        """
        return self._active_levels