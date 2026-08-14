import torch
import torch.nn as nn
from torch import Tensor

from core.base_model import ModelOutput
from data.preprocessing.feature_pipeline import FeatureBatch
from models.pinn.residual_pinn import ResidualPINN


class MultiFidelityPINN(ResidualPINN):
    """
    Physics-Informed Neural Network with multi-fidelity awareness.

    Extends ResidualPINN by concatenating a learned fidelity embedding
    to the input coordinates before passing them through the network.
    This allows the network to adapt its predictions based on the
    resolution level of the input data.

    The fidelity embedding maps an integer fidelity_level to a
    dense vector, which is concatenated with coords:
        network_input = concat([coords, fidelity_embedding(fidelity_level)])

    Args:
        input_dim:          Number of spatial+temporal input dimensions.
                            e.g. 2 for (x, t). The fidelity embedding
                            is added internally — do not include it here.
        output_dim:         Number of output fields.
        n_fidelity_levels:  Total number of fidelity levels. Must match
                            the number of levels used in DataPipeline.
        fidelity_embed_dim: Dimension of the fidelity embedding vector.
        hidden_layers:      Number of residual blocks.
        hidden_size:        Neurons per residual block.
        activation:         Activation function — 'tanh', 'relu', 'silu'.

    Example:
        model = MultiFidelityPINN(
            input_dim=2,
            output_dim=1,
            n_fidelity_levels=3,
            fidelity_embed_dim=8,
        )
        output = model.forward(batch)
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_fidelity_levels: int = 3,
            fidelity_embed_dim: int = 8,
            hidden_layers: int = 6,
            hidden_size: int = 64,
            activation: str = "tanh",
    ):
        self.n_fidelity_levels = n_fidelity_levels
        self.fidelity_embed_dim = fidelity_embed_dim
        self.coord_dim = input_dim

        # Parent builds the network with effective_input_dim
        effective_input_dim = input_dim + fidelity_embed_dim
        super().__init__(
            input_dim=effective_input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            hidden_size=hidden_size,
            activation=activation,
        )

        # Fidelity embedding: int → dense vector of shape (fidelity_embed_dim,)
        self.fidelity_embedding = nn.Embedding(
            num_embeddings=n_fidelity_levels,
            embedding_dim=fidelity_embed_dim,
        )

    def forward(self, batch: FeatureBatch) -> ModelOutput:
        """
        Forward pass with fidelity-aware input.

        Concatenates the fidelity embedding to coords before
        passing through the residual network.

        Args:
            batch: FeatureBatch. Uses batch.coords and batch.fidelity_level.

        Returns:
            ModelOutput with pred of shape (N, output_dim).
        """
        coords = batch.coords
        fidelity_lvl = batch.fidelity_level

        # Fidelity embedding: int → (1, embed_dim) → (N, embed_dim)
        lvl_tensor = torch.tensor(
            [fidelity_lvl], dtype=torch.long, device=coords.device
        )
        embedding = self.fidelity_embedding(lvl_tensor)  # (1, embed_dim)
        embedding = embedding.expand(coords.shape[0], -1)  # (N, embed_dim)

        # Concatenate coords and embedding
        network_input = torch.cat([coords, embedding], dim=1)  # (N, coord_dim + embed_dim)

        pred = self.net(network_input)
        return ModelOutput(pred=pred)

    def compute_gradients(
            self,
            pred: Tensor,
            coords: Tensor,
    ) -> dict[str, Tensor]:
        """
        Computes gradients w.r.t. original coords only.
        The fidelity embedding is treated as a constant conditioning signal.

        Args:
            pred:   Model prediction, shape (N, F).
            coords: Original coordinates (without embedding), shape (N, coord_dim).
                    Must have requires_grad=True.

        Returns:
            Dict with keys 'du_dx', 'du_dt', 'd2u_dx2'.
        """
        return super().compute_gradients(pred, coords)

    def summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return (
            f"MultiFidelityPINN: {total:,} parameters | "
            f"fidelity_levels={self.n_fidelity_levels} | "
            f"embed_dim={self.fidelity_embed_dim}"
        )
