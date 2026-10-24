# ADR-015: PDE Derivatives Assume Pointwise Models

## Status
Accepted

## Context
Equations obtain derivatives with the autograd "sum trick":
grad(pred, coords, grad_outputs=ones) returns d(sum_i u_i)/dx_j.
This equals du_j/dx_j only when pred[j] depends on coords[j] alone
(a pointwise model). In a graph or attention model such as
SE3Transformer, pred[i] also depends on its neighbours'
coordinates, so the Jacobian is not diagonal and the result is not
the field derivative. A PDE residual built on it would be wrong
without any error being raised.

Options:
1. A query/context variant: collocation points query a fixed,
   detached context cloud, so the output depends only on the
   query coordinate.
2. Keep such models data-driven (no PDE residual).

## Decision
Option 2. Models whose output at a point depends on other points'
coordinates are used with the data loss (and with RL), not with
PhysicsEquation residuals. SE3Transformer does not implement
compute_gradients(). The query/context variant is deferred.

## Rationale
- Option 1 needs a separate forward path and its own tests, and
  no current problem requires it.
- Omitting the capability is safer than a plausible but wrong
  residual.

## Consequences
- PDE residuals are valid for pointwise models (BasePINN family).
- Equation docstrings state the pointwise assumption.
- AutogradPDESolver cannot detect a non-pointwise model at runtime.
  A "pointwise" attribute on PhysicsModel that solvers check is a
  candidate for v3.