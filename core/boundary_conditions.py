"""
boundary_conditions.py
Helper functions that build boundary-condition residuals (ADR-017).

Lives in: core/boundary_conditions.py

They are building blocks: boundary_conditions() in a PhysicsProblem stays
overridable. Use the helpers as they are, combine them per boundary, or
write your own residual.

Every helper returns a tensor of shape (N_b, F): N_b boundary points
selected by `mask`, F output channels. If the mask selects nothing, a
zero residual of shape (1, F) is returned, the same convention as
AutogradPDESolver.

neumann() and robin() differentiate pred with respect to coords, so they
need coords.requires_grad=True and a pointwise model (ADR-015).

Example (fixed temperature on the left end, convective cooling on the
right end, where the outward normal points along +x):

    def boundary_conditions(self, batch, pred):
        left  = batch.boundary_mask & (batch.coords[:, 0] < 0.5)
        right = batch.boundary_mask & (batch.coords[:, 0] >= 0.5)
        return torch.cat([
            dirichlet(pred, 300.0, left),
            robin(pred, batch.coords, right, axis=0,
                  alpha=h, beta=k, target=h * T_inf),
        ], dim=0)
"""

from typing import Union

import torch
from torch import Tensor

Target = Union[float, Tensor]


def _zero_residual(pred: Tensor) -> Tensor:
    return torch.zeros(1, pred.shape[-1], dtype=pred.dtype, device=pred.device)


def _select(target: Target, mask: Tensor):
    """A tensor target has one row per point, like pred. A float broadcasts."""
    return target[mask] if isinstance(target, Tensor) else target


def _normal_derivative(pred: Tensor, coords: Tensor, axis: int, sign: float) -> Tensor:
    """sign * d(pred)/d(coords[:, axis]) for every channel, shape (N, F)."""
    if not coords.requires_grad:
        raise RuntimeError(
            "neumann()/robin() need coords.requires_grad=True, and pred "
            "computed from these coords."
        )
    columns = []
    for c in range(pred.shape[1]):
        f = pred[:, c:c + 1]
        if not f.requires_grad:
            raise RuntimeError("pred is not connected to coords through autograd.")
        g = torch.autograd.grad(
            f, coords,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if g is None:
            raise RuntimeError("pred is not connected to coords through autograd.")
        columns.append(g[:, axis:axis + 1])
    return sign * torch.cat(columns, dim=1)


def dirichlet(pred: Tensor, target: Target, mask: Tensor) -> Tensor:
    """
    u - g on the boundary.

    Args:
        pred:   Model prediction, shape (N, F).
        target: g. A float, or a tensor of shape (N, F) like pred.
        mask:   Boolean mask (N,) selecting the boundary points.
    """
    if not mask.any():
        return _zero_residual(pred)
    return pred[mask] - _select(target, mask)


def neumann(
    pred:   Tensor,
    coords: Tensor,
    mask:   Tensor,
    axis:   int,
    target: Target = 0.0,
    sign:   float = 1.0,
) -> Tensor:
    """
    du/dn - g on the boundary (insulated or reflecting boundary: g = 0).

    Args:
        axis: Coordinate column along which the outward normal points.
        sign: +1 if the outward normal points along +axis, -1 if along -axis.
    """
    if not mask.any():
        return _zero_residual(pred)
    dn = _normal_derivative(pred, coords, axis, sign)
    return dn[mask] - _select(target, mask)


def robin(
    pred:   Tensor,
    coords: Tensor,
    mask:   Tensor,
    axis:   int,
    alpha:  float,
    beta:   float,
    target: Target = 0.0,
    sign:   float = 1.0,
) -> Tensor:
    """
    alpha * u + beta * du/dn - g on the boundary.

    Convective cooling, -k dT/dn = h (T - T_inf):
        alpha = h, beta = k, target = h * T_inf
    Vacuum boundary for neutron diffusion (Marshak condition in the
    diffusion approximation), phi + 2 D dphi/dn = 0:
        alpha = 1, beta = 2 * D, target = 0
    """
    if not mask.any():
        return _zero_residual(pred)
    dn = _normal_derivative(pred, coords, axis, sign)
    return alpha * pred[mask] + beta * dn[mask] - _select(target, mask)