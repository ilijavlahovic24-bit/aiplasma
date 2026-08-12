import torch
from torch import nn, Tensor

from core.base_model import PhysicsModel
from core.base_problem import PhysicsProblem
from core.plasma_problem import PlasmaPhysicsProblem
from core.base_model import PhysicsModel, ModelOutput
from data.preprocessing.feature_pipeline import FeatureBatch


class BasePINN(PhysicsModel):
    """
    Base Physics-Informed Neural Network architecture.

    Fully-connected network that maps (x, t) coordinates to
    a physical field prediction. Autograd is used for computing
    spatial and temporal derivatives through the network.

    Args:
        input_dim:    Number of input dimensions (e.g. 2 for (x, t)).
        output_dim:   Number of output fields (e.g. 1 for scalar u).
        hidden_layers: Number of hidden layers.
        hidden_size:  Number of neurons per hidden layer.
        activation:   Activation function - 'tanh', 'relu', or 'silu'.

    Example:
        model = BasePINN(input_dim=2, output_dim=1, hidden_layers=4)
        output = model.forward(batch)  # ModelOutput(pred=(N,1))
    """

    ACTIVATIONS = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_layers: int = 4,
            hidden_size: int = 64,
            activation: str = "tanh",
    ):
        super().__init__()

        if activation not in self.ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from: {list(self.ACTIVATIONS.keys())}"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.net = self._build_network()

    def _build_network(self) -> nn.Sequential:
        """
        Builds the fully-connected network.

        Architecture:
            Linear(input_dim → hidden_size) → Activation
            [Linear(hidden_size → hidden_size) → Activation] * (hidden_layers - 1)
            Linear(hidden_size → output_dim)

        Returns:
            nn.Sequential model.
        """
        act_cls = self.ACTIVATIONS[self.activation]
        layers = [
            nn.Linear(self.input_dim, self.hidden_size),
            act_cls(),
        ]

        for _ in range(self.hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(act_cls())

        layers.append(nn.Linear(self.hidden_size, self.output_dim))
        return nn.Sequential(*layers)

    def forward(self, batch: FeatureBatch) -> ModelOutput:
        """
        Forward pass through the network.

        Args:
            batch: FeatureBatch containing coords of shape (N, D).

        Returns:
            ModelOutput with pred of shape (N, output_dim).
        """
        pred = self.net(batch.coords)
        return ModelOutput(pred=pred)

    def compute_gradients(
            self,
            pred: Tensor,
            coords: Tensor,
    ) -> dict[str, Tensor]:
        """
        Computes spatial and temporal derivatives through autograd.

        Assumes coords layout: [...spatial dims..., t]
            coords[:, 0]  = x
            coords[:, 1]  = y  (if 2D+)
            coords[:, -1] = t

        Args:
            pred:   Model prediction, shape (N, F).
            coords: Input coordinates, shape (N, D).
                    Must have requires_grad=True.

        Returns:
            Dict with keys:
                'du_dx'    - ∂u/∂x,  shape (N, F)
                'du_dt'    - ∂u/∂t,  shape (N, F)
                'd2u_dx2'  - ∂²u/∂x², shape (N, F)

        Raises:
            RuntimeError: If coords does not have requires_grad=True.
        """
        if not coords.requires_grad:
            raise RuntimeError(
                "compute_gradients() requires coords.requires_grad=True. "
                "Set coords = coords.requires_grad_(True) before calling."
            )

        ones = torch.ones_like(pred)

        # First-order gradients w.r.t. all coordinates
        grads = torch.autograd.grad(
            pred, coords,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]  # shape (N, D)

        du_dx = grads[:, 0:1]  # spatial x
        du_dt = grads[:, -1:]  # temporal (last dim)

        # Second-order: ∂²u/∂x²
        d2u_dx2 = torch.autograd.grad(
            du_dx, coords,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        return {
            "du_dx": du_dx,
            "du_dt": du_dt,
            "d2u_dx2": d2u_dx2,
        }