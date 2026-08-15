"""
bayesian_nn.py
Bayesian Neural Network with MC Dropout for AIPlasma framework.

Lives in: models/bayesian/bayesian_nn.py

Uses Monte Carlo Dropout for approximate Bayesian inference.
During inference, dropout remains active and N forward passes
are performed to obtain a distribution over predictions.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from core.base_model import PhysicsModel, ModelOutput
from data.preprocessing.feature_pipeline import FeatureBatch


# ════════════════════════════════════════════════════════════════════════════
# 1. BayesianNN
# ════════════════════════════════════════════════════════════════════════════

class BayesianNN(PhysicsModel):
    """
    Bayesian Neural Network using Monte Carlo Dropout.

    During training: standard forward pass with dropout active.
    During inference: N forward passes with dropout kept active,
    producing a distribution over predictions from which mean
    and uncertainty (std) are computed.

    Dropout is applied only between hidden layers (ADR-007):
        Linear(input) → Activation →
        [Dropout → Linear → Activation] * hidden_layers →
        Linear(output)

    Args:
        input_dim:    Number of input dimensions.
        output_dim:   Number of output fields.
        hidden_size:  Neurons per hidden layer.
        hidden_layers: Number of hidden layers.
        dropout_rate: Dropout probability. Higher = more uncertainty.
        n_samples:    Number of MC forward passes during inference.
        activation:   Activation function — 'tanh', 'relu', or 'silu'.

    Example:
        model  = BayesianNN(input_dim=2, output_dim=1, n_samples=50)
        output = model.forward(batch)
        # output.pred        — mean prediction, shape (N, 1)
        # output.uncertainty — std of predictions, shape (N, 1)
    """

    ACTIVATIONS = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }

    def __init__(
        self,
        input_dim:     int,
        output_dim:    int,
        hidden_size:   int   = 64,
        hidden_layers: int   = 4,
        dropout_rate:  float = 0.1,
        n_samples:     int   = 50,
        activation:    str   = "tanh",
    ):
        super().__init__()

        if activation not in self.ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from: {list(self.ACTIVATIONS.keys())}"
            )

        if not 0.0 < dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must be in (0, 1), got {dropout_rate}."
            )

        self.input_dim     = input_dim
        self.output_dim    = output_dim
        self.hidden_size   = hidden_size
        self.hidden_layers = hidden_layers
        self.dropout_rate  = dropout_rate
        self.n_samples     = n_samples
        self.activation    = activation
        self.net           = self._build_network()

    def _build_network(self) -> nn.Sequential:
        """
        Builds the network with MC Dropout between hidden layers.

        Architecture (Option B):
            Linear(input_dim → hidden_size) → Activation
            [Dropout → Linear(hidden_size → hidden_size) → Activation] * hidden_layers
            Linear(hidden_size → output_dim)

        Returns:
            nn.Sequential model.
        """
        act_cls = self.ACTIVATIONS[self.activation]

        # Input layer — no dropout before first hidden layer
        layers = [
            nn.Linear(self.input_dim, self.hidden_size),
            act_cls(),
        ]

        # Hidden layers with dropout between them
        for _ in range(self.hidden_layers):
            layers.append(nn.Dropout(p=self.dropout_rate))
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(act_cls())

        # Output layer — no dropout after last hidden layer
        layers.append(nn.Linear(self.hidden_size, self.output_dim))
        return nn.Sequential(*layers)

    def forward(self, batch: FeatureBatch) -> ModelOutput:
        """
        Forward pass with uncertainty estimation.

        During training (self.training=True):
            Single forward pass — standard behavior.
            uncertainty is None.

        During inference (self.training=False):
            Calls sample_predictions() for MC Dropout estimation.
            Returns mean as pred and std as uncertainty.

        Args:
            batch: FeatureBatch containing coords of shape (N, D).

        Returns:
            ModelOutput with:
                pred:        Mean prediction, shape (N, output_dim).
                uncertainty: Std of predictions, shape (N, output_dim).
                             None during training.
        """
        if self.training:
            pred = self.net(batch.coords)
            return ModelOutput(pred=pred, uncertainty=None)

        mean, std = self.sample_predictions(batch)
        return ModelOutput(pred=mean, uncertainty=std)

    def sample_predictions(
        self,
        batch: FeatureBatch,
    ) -> tuple[Tensor, Tensor]:
        """
        Performs N MC Dropout forward passes to estimate uncertainty.

        Dropout is kept active during inference by calling self.net.train()
        locally — this does not affect the outer training state.

        Args:
            batch: FeatureBatch containing coords of shape (N, D).

        Returns:
            tuple:
                mean: Mean prediction across samples, shape (N, output_dim).
                std:  Standard deviation across samples, shape (N, output_dim).
                      Higher std = higher epistemic uncertainty.
        """
        # Keep dropout active during inference
        self.net.train()

        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.net(batch.coords)
                samples.append(pred)

        # Stack: (n_samples, N, output_dim)
        stacked = torch.stack(samples, dim=0)

        mean = stacked.mean(dim=0)   # (N, output_dim)
        std  = stacked.std(dim=0)    # (N, output_dim)

        # Restore eval mode after sampling
        self.net.eval()

        return mean, std

    def uncertainty(self, batch: FeatureBatch) -> Tensor:
        """
        Returns only the uncertainty (std) from MC Dropout sampling.

        Convenience method when only uncertainty is needed,
        without the mean prediction.

        Args:
            batch: FeatureBatch containing coords of shape (N, D).

        Returns:
            Std tensor of shape (N, output_dim). All values >= 0.
        """
        _, std = self.sample_predictions(batch)
        return std

    def summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return (
            f"BayesianNN: {total:,} parameters | "
            f"dropout={self.dropout_rate} | "
            f"n_samples={self.n_samples}"
        )