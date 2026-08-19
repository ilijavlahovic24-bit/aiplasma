import os

import torch

from trainers.callbacks.base_callback import Callback


class PhysicsMonitor(Callback):
    """
    Tracks physics-specific metrics independently via solver (ADR-010).

    Computes data_loss, pde_loss, and bc_loss separately by calling
    solver.solve_with_grad() on a reference batch every log_frequency
    epochs. Does not depend on trainer internals.

    Args:
        solver:          PhysicsSolver instance with equation set.
        reference_batch: FeatureBatch used for monitoring.
                         Typically a fixed validation batch.
        log_frequency:   Compute metrics every N epochs. Default: 10.
        plot:            If True, saves loss component plots on train end.
        plot_dir:        Directory for saving plots. Default: 'analysis/metrics/'.
        verbose:         If True, prints metrics when computed.

    Example:
        monitor = PhysicsMonitor(
            solver=AutogradPDESolver(equation=REGISTRY.get("heat_equation_1d")),
            reference_batch=val_batch,
            log_frequency=10,
        )
    """

    def __init__(
            self,
            solver,
            reference_batch,
            log_frequency: int = 10,
            plot: bool = False,
            plot_dir: str = "analysis/metrics/",
            verbose: bool = True,
    ):
        self.solver = solver
        self.reference_batch = reference_batch
        self.log_frequency = log_frequency
        self.plot = plot
        self.plot_dir = plot_dir
        self.verbose = verbose

        # History of physics metrics
        self._epochs: list[int] = []
        self._data_loss: list[float] = []
        self._pde_loss: list[float] = []
        self._bc_loss: list[float] = []

    def on_epoch_end(self, trainer, epoch: int, history, **kwargs) -> None:
        """
        Computes physics metrics every log_frequency epochs.

        Moves reference batch to trainer device, calls solve_with_grad(),
        and logs data_loss, pde_loss, bc_loss separately.
        """
        if epoch % self.log_frequency != 0:
            return

        model = trainer.model
        problem = trainer.problem
        device = trainer.device

        # Move reference batch to device
        batch = self._to_device(self.reference_batch, device)

        # Independent metric computation via solver (ADR-011)
        model.eval()
        with torch.no_grad():
            solver_output = self.solver.solve_with_grad(model, batch)

        pred = solver_output.quantities.get("u", model.forward(batch).pred)

        # Compute loss components separately
        data_loss = float(torch.mean((pred - batch.fields) ** 2))
        pde_res = solver_output.residuals.get("pde", torch.zeros(1))
        bc_res = solver_output.residuals.get("bc", torch.zeros(1))
        pde_loss = float(torch.mean(pde_res ** 2))
        bc_loss = float(torch.mean(bc_res ** 2))

        # Store history
        self._epochs.append(epoch)
        self._data_loss.append(data_loss)
        self._pde_loss.append(pde_loss)
        self._bc_loss.append(bc_loss)

        if self.verbose:
            print(
                f"  [PhysicsMonitor] Epoch {epoch:>4} | "
                f"data={data_loss:.6f}  "
                f"pde={pde_loss:.6f}  "
                f"bc={bc_loss:.6f}"
            )

    def on_train_end(self, trainer, history, **kwargs) -> None:
        """Saves physics metric plots if plot=True."""
        if not self.plot or not self._epochs:
            return

        try:
            import matplotlib.pyplot as plt
            os.makedirs(self.plot_dir, exist_ok=True)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            titles = ["Data Loss", "PDE Loss", "BC Loss"]
            values = [self._data_loss, self._pde_loss, self._bc_loss]

            for ax, title, vals in zip(axes, titles, values):
                ax.plot(self._epochs, vals, marker="o", markersize=3)
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            path = os.path.join(self.plot_dir, "physics_metrics.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  [PhysicsMonitor] Saved physics metrics plot: {path}")

        except ImportError:
            print(
                "[PhysicsMonitor] matplotlib not installed. "
                "Install it to enable plotting: pip install matplotlib"
            )

    def get_history(self) -> dict:
        """
        Returns physics metric history as a dict.

        Returns:
            Dict with keys 'epochs', 'data_loss', 'pde_loss', 'bc_loss'.
        """
        return {
            "epochs": self._epochs,
            "data_loss": self._data_loss,
            "pde_loss": self._pde_loss,
            "bc_loss": self._bc_loss,
        }

    @staticmethod
    def _to_device(batch, device):
        """Moves FeatureBatch tensors to the specified device."""
        from data.preprocessing.feature_pipeline import FeatureBatch
        return FeatureBatch(
            coords=batch.coords.to(device),
            fields=batch.fields.to(device),
            boundary_mask=batch.boundary_mask.to(device),
            collocation_mask=batch.collocation_mask.to(device),
            physics_params=batch.physics_params,
            fidelity_level=batch.fidelity_level,
            fidelity_weight=batch.fidelity_weight,
        )