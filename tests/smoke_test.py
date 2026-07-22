import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn


#Generating syntetic data
#local copy of feature batch for isolated testing
#In actual projects: from data.preprocessing.feature_pipeline import FeatureBatch
@dataclass
class FeatureBatch:
    coords: Tensor
    fields: Tensor
    boundary_mask: Tensor
    collocation_mask: Tensor
    physics_params: dict
    fidelity_level: int
    fidelity_weight: float


def make_heat_equation_batch(
        n_points: int = 200,
        n_boundary: int = 20,
        alpha: float = 0.01,
        fidelity_level: int = 0,
        fidelity_weight: float = 1.0,
        device: str = "cpu",
) -> FeatureBatch:
    """
    Generate syntetic olaition points for 1D heat equation:
        ∂u/∂t = α · ∂²u/∂x²
    na domenu x ∈ [0, 1], t ∈ [0, 1].

    Right solution (initial state: sin(πx)):
        u(x, t) = sin(πx) · exp(-α · π² · t)
    """
    # ── Cordinates (x, t) ────────────────────────────────────────────────────
    x = torch.rand(n_points, device=device)
    t = torch.rand(n_points, device=device)
    coords = torch.stack([x, t], dim=1)  # (N, 2)

    import math
    u_exact = torch.sin(math.pi * x) * torch.exp(
        torch.tensor(-alpha * math.pi ** 2) * t
    )
    fields = u_exact.unsqueeze(1)  # (N, 1)

    # ── Boundary mask: x=0 ili x=1 ───────────────────────────────────────────
    boundary_mask = (x < 0.05) | (x > 0.95)  # (N,) bool

    # ── Collocation mask: interior ─────────────────────────────────────
    collocation_mask = ~boundary_mask  # (N,) bool

    return FeatureBatch(
        coords=coords,
        fields=fields,
        boundary_mask=boundary_mask,
        collocation_mask=collocation_mask,
        physics_params={"alpha": alpha},
        fidelity_level=fidelity_level,
        fidelity_weight=fidelity_weight,
    )

def make_multifidelity_batches(
    alpha: float = 0.01,
    device: str = "cpu",
) -> list[FeatureBatch]:
    """
    Vraća tri fidelity nivoa istog problema:
      - Level 0:  coarse approximation, small number of points
      - Level 1: middle approximation, medium number of points
      - Level 2: fine solution, large number of points
    """
    configs = [
        {"n_points": 50,  "fidelity_level": 0, "fidelity_weight": 0.2},
        {"n_points": 150, "fidelity_level": 1, "fidelity_weight": 0.3},
        {"n_points": 300, "fidelity_level": 2, "fidelity_weight": 0.5},
    ]
    return [
        make_heat_equation_batch(alpha=alpha, device=device, **cfg)
        for cfg in configs
    ]


@dataclass
class ModelOutput:
    pred: Tensor
    uncertainty: Optional[Tensor] = None
    aux: dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# 2. PhysicsModel — abstract base
# ════════════════════════════════════════════════════════════════════════════

class PhysicsModel(nn.Module, ABC):

    @abstractmethod
    def forward(self, batch: FeatureBatch) -> ModelOutput:
        ...

    def predict(self, batch: FeatureBatch) -> ModelOutput:
        with torch.no_grad():
            return self.forward(batch)

    def preprocess(self, batch: FeatureBatch) -> FeatureBatch:
        return batch

    def summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"{self.__class__.__name__}: {total:,} parameters"


# ════════════════════════════════════════════════════════════════════════════
# 3. PhysicsProblem — abstract base
# ════════════════════════════════════════════════════════════════════════════

class PhysicsProblem(ABC):
    lambda1: float = 1.0  # data loss
    lambda2: float = 1.0  # PDE loss
    lambda3: float = 1.0  # BC loss

    @abstractmethod
    def pde_residual(self, batch: FeatureBatch, pred: Tensor) -> Tensor:
        ...

    @abstractmethod
    def boundary_conditions(self, batch: FeatureBatch, pred: Tensor) -> Tensor:
        ...

    def loss(self, batch: FeatureBatch, pred: Tensor) -> Tensor:
        w = batch.fidelity_weight

        data_loss = torch.mean((pred - batch.fields) ** 2)
        pde_loss = torch.mean(self.pde_residual(batch, pred) ** 2)
        bc_loss = torch.mean(self.boundary_conditions(batch, pred) ** 2)

        return w * (self.lambda1 * data_loss
                    + self.lambda2 * pde_loss
                    + self.lambda3 * bc_loss)

    def physics_params(self) -> dict:
        return {}

    def fidelity_weights(self) -> list:
        return [1.0]


# ════════════════════════════════════════════════════════════════════════════
# 4. BaseTrainer — training loop
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainerConfig:
    max_epochs: int = 20
    val_frequency: int = 5
    log_frequency: int = 5
    lr: float = 1e-3


@dataclass
class TrainingHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    epochs: list = field(default_factory=list)


class BaseTrainer(ABC):
    callbacks: list = []

    def __init__(
            self,
            model: PhysicsModel,
            problem: PhysicsProblem,
            config: TrainerConfig,
    ):
        self.model = model
        self.problem = problem
        self.config = config
        self.history = TrainingHistory()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @abstractmethod
    def train_step(self, batch: FeatureBatch) -> Tensor:
        ...

    @abstractmethod
    def val_step(self, batch: FeatureBatch) -> Tensor:
        ...

    def fit(
            self,
            train_batches: list[FeatureBatch],
            val_batches: Optional[list[FeatureBatch]] = None,
    ) -> TrainingHistory:

        self._fire("on_train_start")

        for epoch in range(1, self.config.max_epochs + 1):
            self._fire("on_epoch_start", epoch=epoch)

            # ── Training ────────────────────────────────────────
            self.model.train()
            train_losses = [
                self.train_step(self._to_device(b)).item()
                for b in train_batches
            ]
            epoch_train_loss = sum(train_losses) / len(train_losses)

            # ── Validacija ──────────────────────────────────────
            epoch_val_loss = None
            if val_batches and epoch % self.config.val_frequency == 0:
                self.model.eval()
                with torch.no_grad():
                    val_losses = [
                        self.val_step(self._to_device(b)).item()
                        for b in val_batches
                    ]
                epoch_val_loss = sum(val_losses) / len(val_losses)

            # ── History ─────────────────────────────────────────
            self.history.epochs.append(epoch)
            self.history.train_loss.append(epoch_train_loss)
            if epoch_val_loss is not None:
                self.history.val_loss.append(epoch_val_loss)

            # ── Log ─────────────────────────────────────────────
            if epoch % self.config.log_frequency == 0:
                val_str = f"  val={epoch_val_loss:.6f}" if epoch_val_loss else ""
                print(f"  [Epoch {epoch:>3}] train={epoch_train_loss:.6f}{val_str}")

            self._fire("on_epoch_end", epoch=epoch, history=self.history)

        self._fire("on_train_end", history=self.history)
        return self.history

    def _to_device(self, batch: FeatureBatch) -> FeatureBatch:
        return FeatureBatch(
            coords=batch.coords.to(self.device),
            fields=batch.fields.to(self.device),
            boundary_mask=batch.boundary_mask.to(self.device),
            collocation_mask=batch.collocation_mask.to(self.device),
            physics_params=batch.physics_params,
            fidelity_level=batch.fidelity_level,
            fidelity_weight=batch.fidelity_weight,
        )

    def _fire(self, event: str, **kwargs):
        for cb in self.callbacks:
            if hasattr(cb, event):
                getattr(cb, event)(trainer=self, **kwargs)


# ════════════════════════════════════════════════════════════════════════════
# 5. CConcrete implementation of smoke test
# ════════════════════════════════════════════════════════════════════════════

class HeatEquationProblem(PhysicsProblem):
    """
    1D Heat equation: ∂u/∂t = α · ∂²u/∂x²
    PDE residual: ∂u/∂t - α · ∂²u/∂x² = 0
    BC: u(0,t) = u(1,t) = 0
    """

    lambda1 = 1.0
    lambda2 = 1.0
    lambda3 = 1.0

    def physics_params(self) -> dict:
        return {"alpha": 0.01}

    def pde_residual(self, batch: FeatureBatch, pred: Tensor) -> Tensor:
        alpha = self.physics_params()["alpha"]
        coords = batch.coords.requires_grad_(True)

        # Re-run forward da dobijemo autograd graph
        u = pred[batch.collocation_mask]
        if u.numel() == 0:
            return torch.tensor(0.0, requires_grad=True)

        # Aproksimacija residuala samo na interior tačkama
        # U punoj implementaciji: autograd kroz model
        # Ovde: finite difference aproksimacija za smoke test
        return u - u.detach()  # → 0, samo da proverimo da pipeline radi

    def boundary_conditions(self, batch: FeatureBatch, pred: Tensor) -> Tensor:
        bc_pred = pred[batch.boundary_mask]
        bc_true = batch.fields[batch.boundary_mask]
        return bc_pred - bc_true


class SimpleMLP(PhysicsModel):
    """
    Trivijalna fully-connected mreža: (x, t) → u
    Dovoljna za smoke test pipeline-a.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: FeatureBatch) -> ModelOutput:
        pred = self.net(batch.coords)
        return ModelOutput(pred=pred)


class HeatTrainer(BaseTrainer):
    """
    Concrete trainer za heat equation.
    Implements train_step i val_step.
    """

    def __init__(self, model, problem, config):
        super().__init__(model, problem, config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr
        )

    def train_step(self, batch: FeatureBatch) -> Tensor:
        self.optimizer.zero_grad()
        output = self.model.forward(batch)
        loss = self.problem.loss(batch, output.pred)
        loss.backward()
        self.optimizer.step()
        return loss

    def val_step(self, batch: FeatureBatch) -> Tensor:
        output = self.model.forward(batch)
        return self.problem.loss(batch, output.pred)


# ════════════════════════════════════════════════════════════════════════════
# 6. Smoke test runner
# ════════════════════════════════════════════════════════════════════════════

def run_smoke_test():
    print("=" * 60)
    print("AIPlasma Smoke Test — 1D Heat Equation")
    print("=" * 60)

    # ── Podaci ──────────────────────────────────────────────────
    print("\n[1/5] Generate synthetic data...")
    train_batches = make_multifidelity_batches(alpha=0.01)
    val_batch = [make_heat_equation_batch(n_points=100)]
    print(f"      Train: {len(train_batches)} fidelity level")
    print(f"      Val:   {len(val_batch[0].coords)} points")

    # ── Problem ─────────────────────────────────────────────────
    print("\n[2/5] Initialise HeatEquationProblem...")
    problem = HeatEquationProblem()
    print(f"      physics_params = {problem.physics_params()}")
    print(f"      λ1={problem.lambda1}, λ2={problem.lambda2}, λ3={problem.lambda3}")

    # ── Model ───────────────────────────────────────────────────
    print("\n[3/5] Initialise SimpleMLP...")
    model = SimpleMLP(hidden=32)
    print(f"      {model.summary()}")

    # ── Trainer ─────────────────────────────────────────────────
    print("\n[4/5] Initialise HeatTrainer...")
    config = TrainerConfig(max_epochs=20, val_frequency=5, log_frequency=5, lr=1e-3)
    trainer = HeatTrainer(model=model, problem=problem, config=config)
    print(f"      device = {trainer.device}")

    # ── Trening ─────────────────────────────────────────────────
    print("\n[5/5] Starting fit()...")
    start = time.perf_counter()
    history = trainer.fit(train_batches=train_batches, val_batches=val_batch)
    elapsed = time.perf_counter() - start

    # ── Rezultati ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Epochs:          {len(history.epochs)}")
    print(f"  Start loss:    {history.train_loss[0]:.6f}")
    print(f"  End loss:    {history.train_loss[-1]:.6f}")
    print(f"  Val loss (last): {history.val_loss[-1]:.6f}" if history.val_loss else "  Val loss: N/A")
    print(f"  Elasped time:    {elapsed:.2f}s")

    loss_improved = history.train_loss[-1] < history.train_loss[0]
    print(f"\n  Loss decressed: {'YES' if loss_improved else 'NO ✗'}")

    print("\n" + "=" * 60)
    if loss_improved:
        print("SMOKE TEST Passed")
        print("All core interfaces works end-to-end.")
    else:
        print("SMOKE TEST Failed")
        print("Loss did not decrease — check learning rate or architecture.")
    print("=" * 60)

    return history


if __name__ == "__main__":
    run_smoke_test()
