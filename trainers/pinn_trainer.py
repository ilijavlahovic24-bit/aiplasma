"""
pinn_trainer.py
PINN Trainer for AIPlasma framework.

Lives in: trainers/pinn_trainer.py
"""

import torch
from torch import Tensor
from typing import Optional

from core.base_trainer import BaseTrainer, TrainerConfig
from core.base_model import PhysicsModel
from core.base_problem import PhysicsProblem
from core.base_solver import PhysicsSolver
from data.preprocessing.feature_pipeline import FeatureBatch
from trainers.callbacks.check_pointing import CheckPointing
from trainers.callbacks.physics_monitor import PhysicsMonitor


class PINNTrainer(BaseTrainer):
    """
    Concrete trainer for Physics-Informed Neural Networks.

    Implements train_step() and val_step() for standard PINN training:
        1. Forward pass through model
        2. Compute multi-fidelity loss via problem.loss()
        3. Backward pass and optimizer step

    Default callbacks: PhysicsMonitor and CheckPointing.
    Override the callbacks class variable to customize.

    Args:
        model:     PhysicsModel instance (BasePINN, ResidualPINN, etc.)
        problem:   PhysicsProblem instance defining the physics.
        solver:    PhysicsSolver instance for PDE residual computation.
        config:    TrainerConfig with training hyperparameters.
        lr:        Learning rate for Adam optimizer. Default: 1e-3.
        scheduler: If True, uses ReduceLROnPlateau scheduler. Default: True.

    Example:
        trainer = PINNTrainer(
            model=BasePINN(input_dim=2, output_dim=1),
            problem=HeatEquationProblem(),
            solver=AutogradPDESolver(equation=REGISTRY.get("heat_equation_1d")),
            config=TrainerConfig(max_epochs=100),
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
        lr:        float = 1e-3,
        scheduler: bool  = True,
    ):
        super().__init__(model, problem, solver, config)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr
        )

        self.lr_scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                verbose=False,
            )
            if scheduler else None
        )

    # ── Abstract implementations ─────────────────────────────────────────────

    def train_step(self, batch: FeatureBatch) -> Tensor:
        """
        Single PINN training step.

        Steps:
            1. Zero gradients
            2. Forward pass through model
            3. Compute multi-fidelity loss via problem.loss()
            4. Backward pass
            5. Optimizer step

        Args:
            batch: FeatureBatch at current training step.

        Returns:
            Loss tensor for this step.
        """
        self.optimizer.zero_grad()

        output = self.model.forward(batch)
        loss   = self.problem.loss(batch, output.pred)

        loss.backward()
        self.optimizer.step()

        return loss

    def val_step(self, batch: FeatureBatch) -> Tensor:
        """
        Single PINN validation step.
        No backward pass - only forward and loss computation.
        Args:
            batch: FeatureBatch at current validation step.
        Returns:
            Loss tensor for this step.
        """
        output = self.model.forward(batch)
        return self.problem.loss(batch, output.pred)

    # ── Scheduler hook ───────────────────────────────────────────────────────

    def _fire(self, event: str, **kwargs) -> None:
        """
        Extends base _fire() to step lr_scheduler on epoch end.
        """
        super()._fire(event, **kwargs)

        if event == "on_epoch_end" and self.lr_scheduler is not None:
            history = kwargs.get("history")
            if history and history.val_loss:
                self.lr_scheduler.step(history.val_loss[-1])