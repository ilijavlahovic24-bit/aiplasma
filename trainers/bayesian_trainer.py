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
            )
            if scheduler else None
        )

    # ── Abstract implementations ─────────────────────────────────────────────

    def train_step(self, batch: FeatureBatch) -> Tensor:
        self.optimizer.zero_grad()

        batch  = self._with_grad_coords(batch)    # PDE residuals need d(pred)/d(coords)
        output = self.model.forward(batch)        # model.training=True: MC Dropout active
        loss   = self.problem.loss(batch, output.pred)

        loss.backward()
        self.optimizer.step()

        return loss

    def val_step(self, batch: FeatureBatch) -> Tensor:
        # fit() runs validation under torch.no_grad(), but the PDE residual needs autograd.
        with torch.enable_grad():
            batch     = self._with_grad_coords(batch)
            mean, std = self.model.sample_predictions(batch)
            loss      = self.problem.loss(batch, mean)

        avg_uncertainty = float(std.mean())
        if not hasattr(self, "_uncertainty_history"):
            self._uncertainty_history = []
        self._uncertainty_history.append(avg_uncertainty)

        return loss.detach()


    # ── Scheduler hook ───────────────────────────────────────────────────────

    def _fire(self, event: str, **kwargs) -> None:
        """
        Extends base _fire() to step lr_scheduler on epoch end.
        """
        super()._fire(event, **kwargs)

        if event == "on_epoch_end" and self.lr_scheduler is not None:
            epoch = kwargs.get("epoch")
            history = kwargs.get("history")
            # Step only on epochs where validation ran (see PINNTrainer._fire)
            if history and history.val_loss and epoch % self.config.val_frequency == 0:
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