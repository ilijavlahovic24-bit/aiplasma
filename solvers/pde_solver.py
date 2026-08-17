"""
pde_solver.py
PDE solver for AIPlasma framework.

Lives in: solvers/pde_solver.py

Provides AutogradPDESolver which uses PyTorch autograd
for computing spatial and temporal derivatives through the model.
"""

import torch
from torch import Tensor
from typing import Optional

from core.base_model import PhysicsModel, ModelOutput
from core.base_solver import PDESolver, SolverOutput, SolverInfo
from data.preprocessing.feature_pipeline import FeatureBatch


class AutogradPDESolver(PDESolver):
    """
    PDE solver using PyTorch autograd for derivative computation.

    Primary solver for PINN training - computes spatial and temporal
    derivatives through the model's computational graph, enabling
    physics-informed loss computation.

    Two modes (ADR-003):
        solve()           - post-processing without autograd.
                            Receives ModelOutput, returns physical quantities.
        solve_with_grad() - training mode with full autograd through model.
                            Computes PDE residual via PhysicsEquation.residual().

    Args:
        equation: Optional PhysicsEquation from REGISTRY.
                  If None, solve_with_grad() only returns derivatives
                  without computing PDE residual.

    Example:
        from solvers.physics_constraints import REGISTRY

        solver = AutogradPDESolver(
            equation=REGISTRY.get("heat_equation_1d")
        )
        output = solver.solve_with_grad(model, batch)
        # output.quantities["du_dx"]   -spatial derivative
        # output.quantities["du_dt"]   - temporal derivative
        # output.residuals["pde"]      - PDE residual
    """

    def __init__(self, equation=None):
        self.equation = equation

    # ── Public API ───────────────────────────────────────────────────────────

    def solve(
        self,
        output: ModelOutput,
        batch:  FeatureBatch,
    ) -> SolverOutput:
        """
        Post-processing without autograd.

        Computes basic quantities from model output without
        differentiating through the model. Useful for evaluation
        and visualization after training.

        Args:
            output: ModelOutput from model.predict() or model.forward().
            batch:  Corresponding FeatureBatch.

        Returns:
            SolverOutput with quantities:
                'u'           - model prediction
                'u_boundary'  - prediction at boundary points
                'u_interior'  - prediction at collocation points
        """
        pred = output.pred

        return SolverOutput(
            quantities={
                "u":          pred,
                "u_boundary": pred[batch.boundary_mask],
                "u_interior": pred[batch.collocation_mask],
            },
            residuals={},
            solver_info=SolverInfo(
                solver_type="AutogradPDE",
                converged=None,
                iterations=None,
            ),
        )

    def solve_with_grad(
        self,
        model: PhysicsModel,
        batch: FeatureBatch,
    ) -> SolverOutput:
        """
        Full PDE solve with autograd through model.

        Steps:
            1. Forward pass with coords.requires_grad=True
            2. Compute spatial and temporal derivatives
            3. If equation is set: compute PDE residual
            4. Compute boundary residual

        Args:
            model: PhysicsModel instance.
            batch: FeatureBatch. coords.requires_grad is set internally.

        Returns:
            SolverOutput with:
                quantities:
                    'u'       - full prediction
                    'du_dx'   - ∂u/∂x
                    'du_dt'   - ∂u/∂t
                    'd2u_dx2' - ∂²u/∂x²
                residuals:
                    'pde' - PDE residual (if equation is set, else zeros)
                    'bc'  - boundary condition residual
        """
        # Enable autograd on coordinates
        coords = batch.coords.clone().requires_grad_(True)

        local_batch = FeatureBatch(
            coords=coords,
            fields=batch.fields,
            boundary_mask=batch.boundary_mask,
            collocation_mask=batch.collocation_mask,
            physics_params=batch.physics_params,
            fidelity_level=batch.fidelity_level,
            fidelity_weight=batch.fidelity_weight,
        )

        # Forward pass
        output = model.forward(local_batch)
        pred   = output.pred

        # Compute derivatives
        derivatives = self.compute_derivatives(pred, coords)

        # PDE residual
        if self.equation is not None:
            self.equation.validate_params(batch.physics_params)
            pde_residual = self.equation.residual(
                coords, pred, batch.physics_params
            )
        else:
            pde_residual = torch.zeros_like(pred)

        # Boundary residual
        bc_residual = self._compute_bc_residual(pred, batch)

        return SolverOutput(
            quantities={
                "u":       pred,
                **derivatives,
            },
            residuals={
                "pde": pde_residual,
                "bc":  bc_residual,
            },
            solver_info=SolverInfo(
                solver_type="AutogradPDE",
                converged=True,
                iterations=1,
            ),
        )

    def compute_derivatives(
        self,
        pred:   Tensor,
        coords: Tensor,
    ) -> dict[str, Tensor]:
        """
        Computes spatial and temporal derivatives through autograd.

        Assumes coords layout: [x, (y), (z), t]
            coords[:, 0]  = x   (spatial)
            coords[:, 1]  = y   (spatial, if 2D+)
            coords[:, -1] = t   (temporal, always last)

        Args:
            pred:   Model prediction, shape (N, F).
                    Must be connected to coords via autograd graph.
            coords: Input coordinates, shape (N, D).
                    Must have requires_grad=True.

        Returns:
            Dict with keys:
                'du_dx'   - ∂u/∂x,   shape (N, F)
                'du_dt'   - ∂u/∂t,   shape (N, F)
                'd2u_dx2' - ∂²u/∂x², shape (N, F)

        Raises:
            RuntimeError: If coords does not have requires_grad=True.
        """
        if not coords.requires_grad:
            raise RuntimeError(
                "compute_derivatives() requires coords.requires_grad=True."
            )

        ones = torch.ones_like(pred)

        # First-order gradients w.r.t. all coordinates
        first_order = torch.autograd.grad(
            pred, coords,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]  # (N, D)

        du_dx = first_order[:, 0:1]    # spatial x - first dim
        du_dt = first_order[:, -1:]    # temporal  - last dim

        # Second-order: ∂²u/∂x²
        d2u_dx2 = torch.autograd.grad(
            du_dx, coords,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        return {
            "du_dx":   du_dx,
            "du_dt":   du_dt,
            "d2u_dx2": d2u_dx2,
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_bc_residual(
        pred:  Tensor,
        batch: FeatureBatch,
    ) -> Tensor:
        """
        Computes boundary condition residual.

        BC residual = pred at boundary points - true values at boundary.
        Returns zeros if no boundary points exist in batch.

        Args:
            pred:  Model prediction, shape (N, F).
            batch: FeatureBatch with boundary_mask and fields.

        Returns:
            BC residual tensor, shape (N_bc, F) where N_bc = boundary points.
        """
        if not batch.boundary_mask.any():
            return torch.zeros(1, pred.shape[-1], device=pred.device)

        bc_pred = pred[batch.boundary_mask]
        bc_true = batch.fields[batch.boundary_mask]
        return bc_pred - bc_true