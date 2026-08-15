"""
ensemble_model.py
Ensemble model for AIPlasma framework.

Lives in: models/ensemble/ensemble_model.py

Aggregates predictions from multiple PhysicsModel instances.
Does not train directly - receives already-trained models.
"""

import torch
from torch import Tensor
from typing import Optional

from core.base_model import PhysicsModel, ModelOutput
from data.preprocessing.feature_pipeline import FeatureBatch


class EnsembleModel(PhysicsModel):
    """
    Aggregates predictions from multiple PhysicsModel instances.

    Does not have its own training loop - receives already-trained
    models and combines their predictions at inference time.

    Supports two aggregation modes:
        'mean':          Simple average of all model predictions.
        'weighted_mean': Weighted average using per-model weights.

    Uncertainty is estimated as the standard deviation across
    model predictions - higher disagreement = higher uncertainty.

    Args:
        models:      List of trained PhysicsModel instances.
        weights:     Optional list of weights for weighted_mean aggregation.
                     Must have the same length as models.
                     If None and aggregation='weighted_mean', uniform weights
                     are used.
        aggregation: Aggregation strategy - 'mean' or 'weighted_mean'.

    Example:
        pinn    = BasePINN(input_dim=2, output_dim=1)
        bayes   = BayesianNN(input_dim=2, output_dim=1)
        ensemble = EnsembleModel(models=[pinn, bayes], aggregation='mean')
        output   = ensemble.forward(batch)
        # output.pred        - aggregated prediction
        # output.uncertainty - std across model predictions
    """

    AGGREGATIONS = ["mean", "weighted_mean"]

    def __init__(
        self,
        models:      list[PhysicsModel],
        weights:     Optional[list[float]] = None,
        aggregation: str                   = "mean",
    ):
        super().__init__()

        if not models:
            raise ValueError("EnsembleModel requires at least one model.")

        if aggregation not in self.AGGREGATIONS:
            raise ValueError(
                f"Unsupported aggregation '{aggregation}'. "
                f"Choose from: {self.AGGREGATIONS}"
            )

        if weights is not None and len(weights) != len(models):
            raise ValueError(
                f"weights length ({len(weights)}) must match "
                f"models length ({len(models)})."
            )

        # Register models as a ModuleList so PyTorch tracks parameters
        self.models      = torch.nn.ModuleList(models)
        self.aggregation = aggregation
        self.weights     = self._normalize_weights(weights, len(models))


    def forward(self, batch: FeatureBatch) -> ModelOutput:
        """
        Aggregates predictions from all models.

        Each model's forward() is called independently.
        Uncertainty is computed as std across model predictions.

        Args:
            batch: FeatureBatch passed to each model.

        Returns:
            ModelOutput with:
                pred:        Aggregated prediction, shape (N, output_dim).
                uncertainty: Std across model predictions, shape (N, output_dim).
        """
        # Collect predictions from all models
        preds = []
        for model in self.models:
            output = model.forward(batch)
            preds.append(output.pred)

        # Stack: (n_models, N, output_dim)
        stacked = torch.stack(preds, dim=0)

        # Aggregate
        if self.aggregation == "mean":
            aggregated = stacked.mean(dim=0)
        else:
            # weighted_mean: weights shape (n_models, 1, 1) for broadcasting
            w          = self.weights.to(stacked.device)
            w          = w.view(-1, 1, 1)
            aggregated = (stacked * w).sum(dim=0)

        # Uncertainty: std across model predictions
        uncertainty = stacked.std(dim=0)

        return ModelOutput(pred=aggregated, uncertainty=uncertainty)

    def add_model(
        self,
        model:  PhysicsModel,
        weight: float = 1.0,
    ) -> None:
        """
        Adds a model to the ensemble at runtime.

        Weights are re-normalized after adding the new model
        to ensure they sum to 1.0.

        Args:
            model:  Trained PhysicsModel instance to add.
            weight: Relative weight for weighted_mean aggregation.
                    Ignored if aggregation='mean'.
        """
        self.models.append(model)

        # Append new weight and re-normalize
        current = self.weights.tolist()
        current.append(weight)
        self.weights = self._normalize_weights(current, len(current))

    def summary(self) -> str:
        total  = sum(p.numel() for p in self.parameters())
        names  = [m.__class__.__name__ for m in self.models]
        return (
            f"EnsembleModel: {total:,} parameters | "
            f"aggregation={self.aggregation} | "
            f"models={names}"
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_weights(
        weights:  Optional[list[float]],
        n_models: int,
    ) -> Tensor:
        """
        Normalizes weights to sum to 1.0.

        If weights is None, returns uniform weights.

        Args:
            weights:  List of raw weights, or None for uniform.
            n_models: Number of models in the ensemble.

        Returns:
            Normalized weight tensor of shape (n_models,).
        """
        if weights is None:
            return torch.ones(n_models) / n_models

        w = torch.tensor(weights, dtype=torch.float32)

        if w.sum() == 0:
            raise ValueError("Weights must not all be zero.")

        return w / w.sum()