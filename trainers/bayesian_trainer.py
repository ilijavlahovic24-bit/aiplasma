"""
bayesian_trainer.py
Bayesian Neural Network Trainer for AIPlasma framework.

Lives in: trainers/bayesian_trainer.py
"""

import torch
from torch import Tensor

from core.base_trainer import BaseTrainer, TrainerConfig
from core.base_model import PhysicsModel
from core.base_problem import PhysicsProblem
from core.base_solver import PhysicsSolver
from data.preprocessing.feature_pipeline import FeatureBatch
from models.bayesian.bayesian_nn import BayesianNN


class BayesianTrainer(BaseTrainer):
    """
    Concrete trainer for BayesianNN with MC Dropout.

    Key difference from PINNTrainer (ADR-011):
        train_step() - identical to PINNTrainer, Bayesian behavior
                       comes from model architecture (MC Dropout),
                       not from an explicit KL term in the loss.
        val_step()   - uses sample_predictions() for uncertainty-aware
                       evaluation instead of a single forward pass.
                       N times slower than PINNTrainer validation.

    Requires model to be a BayesianNN instance - raises ValueError
    otherwise, since sample_predictions() is BayesianNN-specific.

    Args:
        model:     BayesianNN instance.
        problem:   PhysicsProblem instance defining the physics.
        solver:    PhysicsSolver instance.
        config:    TrainerConfig with training hyperparameters.
        lr:        Learning rate for Adam optimizer. Default: 1e-3.
        scheduler: If True, uses ReduceLROnPlateau scheduler. Default: True.

    Example:
        trainer = BayesianTrainer(
            model=BayesianNN(input_dim=2, output_dim=1, n_samples=50),
            problem=DriftDiffusionProblem(),
            solver=AutogradPDESolver(equation=REGISTRY.get("drift_diffusion_1d")),
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
        if not isinstance(model, BayesianNN):
            raise ValueError(
                "BayesianTrainer requires a BayesianNN model instance. "
                f"Got: {type(model).__name__}. "
                "Use PINNTrainer for non-Bayesian models."
            )

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
        Single Bayesian training step.

        Identical to PINNTrainer.train_step() - Bayesian behavior
        comes from MC Dropout in the model, not from the loss (ADR-011).

        Steps:
            1. Zero gradients
            2. Single forward pass (dropout active during training)
            3. Compute loss via problem.loss()
            4. Backward pass
            5. Optimizer step

        Args:
            batch: FeatureBatch at current training step.

        Returns:
            Loss tensor for this step.
        """
        self.optimizer.zero_grad()

        # model.training=True here - MC Dropout active, single forward pass
        output = self.model.forward(batch)
        loss   = self.problem.loss(batch, output.pred)

        loss.backward()
        self.optimizer.step()

        return loss

    def val_step(self, batch: FeatureBatch) -> Tensor:
        """
        Single Bayesian validation step with uncertainty estimation.

        Uses sample_predictions() instead of a single forward pass -
        runs N MC Dropout samples and evaluates loss on the mean
        prediction. N times slower than PINNTrainer.val_step().

        Args:
            batch: FeatureBatch at current validation step.

        Returns:
            Loss tensor computed on mean prediction across MC samples.
        """
        # sample_predictions() keeps dropout active internally
        mean, std = self.model.sample_predictions(batch)

        # Evaluate loss on mean prediction
        loss = self.problem.loss(batch, mean)

        # Store uncertainty in history.extra for PhysicsMonitor
        avg_uncertainty = float(std.mean())
        if not hasattr(self, "_uncertainty_history"):
            self._uncertainty_history = []
        self._uncertainty_history.append(avg_uncertainty)

        return loss

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

    # ── Uncertainty access ───────────────────────────────────────────────────

    def uncertainty_history(self) -> list[float]:
        """
        Returns average uncertainty per validation step.

        Useful for tracking how model uncertainty evolves during training -
        well-calibrated models show decreasing uncertainty as training
        progresses and the model becomes more confident.

        Returns:
            List of average std values per val_step() call.
            Empty if no validation has been performed yet.
        """
        return getattr(self, "_uncertainty_history", [])