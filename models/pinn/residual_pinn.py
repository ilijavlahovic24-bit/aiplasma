from torch import nn, Tensor

from models.pinn.base_pinn import BasePINN


class ResidualBlock(nn.Module):
    """
    Single residual block: Linear → Activation → Linear + skip connection.

    Improves gradient flow in deeper networks.

    Args:
        size:       Input and output dimension (must match for skip).
        activation: Activation function class (e.g. nn.Tanh).
    """

    def __init__(self, size: int, activation: nn.Module):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            activation,
            nn.Linear(size, size),
        )
        self.act = activation.__class__()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.block(x) + x)


class ResidualPINN(BasePINN):
    """
    PINN with residual (skip) connections for improved convergence
    in deeper networks.

    Replaces standard hidden layers with ResidualBlocks.
    Inherits forward() and compute_gradients() from BasePINN.

    Args:
        input_dim:    Number of input dimensions.
        output_dim:   Number of output fields.
        hidden_layers: Number of residual blocks.
        hidden_size:  Neurons per block (input and output must match).
        activation:   Activation function - 'tanh', 'relu', or 'silu'.

    Example:
        model = ResidualPINN(input_dim=2, output_dim=1, hidden_layers=6)
    """

    def _build_network(self) -> nn.Sequential:
        """
        Builds the residual network.

        Architecture:
            Linear(input_dim → hidden_size) → Activation
            [ResidualBlock(hidden_size)] * hidden_layers
            Linear(hidden_size → output_dim)

        Returns:
            nn.Sequential model with residual blocks.
        """
        act_cls = self.ACTIVATIONS[self.activation]
        act = act_cls()


        layers = [
            nn.Linear(self.input_dim, self.hidden_size),
            act_cls(),
        ]

        for _ in range(self.hidden_layers):
            layers.append(ResidualBlock(self.hidden_size, act))

        layers.append(nn.Linear(self.hidden_size, self.output_dim))
        return nn.Sequential(*layers)