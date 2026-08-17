"""
ode_solver.py
ODE solvers for AIPlasma framework.

Lives in: solvers/ode_solver.py

Provides three solvers (ADR-008):
    EulerODESolver    - fixed dt, first-order
    RK4ODESolver      - fixed dt, fourth-order
    AdaptiveODESolver - adaptive dt via RK45 error control
"""

import torch
from torch import Tensor
from typing import Optional

from core.base_model import PhysicsModel, ModelOutput
from core.base_solver import ODESolver, SolverOutput, SolverInfo
from data.preprocessing.feature_pipeline import FeatureBatch


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _get_time_derivative(
    model: PhysicsModel,
    batch: FeatureBatch,
) -> Tensor:
    """
    Computes du/dt through the model using autograd.

    Creates a local FeatureBatch with coords.requires_grad=True,
    runs a forward pass, and extracts the temporal derivative.

    Args:
        model: PhysicsModel instance.
        batch: Current FeatureBatch.

    Returns:
        du/dt tensor of shape (N, output_dim).
    """
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

    output = model.forward(local_batch)
    pred   = output.pred

    ones   = torch.ones_like(pred)
    grads  = torch.autograd.grad(
        pred, coords,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
    )[0]

    return grads[:, -1:]   # temporal coordinate is last dim


def _advance_batch(batch: FeatureBatch, u_new: Tensor, dt: float) -> FeatureBatch:
    """
    Returns a new FeatureBatch with updated fields and time coordinate.

    Args:
        batch: Current FeatureBatch.
        u_new: Updated field values, shape (N, output_dim).
        dt:    Time step used.

    Returns:
        New FeatureBatch at t + dt.
    """
    coords_new          = batch.coords.clone()
    coords_new[:, -1]  += dt   # advance time coordinate

    return FeatureBatch(
        coords=coords_new,
        fields=u_new.detach(),
        boundary_mask=batch.boundary_mask,
        collocation_mask=batch.collocation_mask,
        physics_params=batch.physics_params,
        fidelity_level=batch.fidelity_level,
        fidelity_weight=batch.fidelity_weight,
    )


# ════════════════════════════════════════════════════════════════════════════
# 1. EulerODESolver
# ════════════════════════════════════════════════════════════════════════════

class EulerODESolver(ODESolver):
    """
    First-order Euler ODE solver with fixed step size.

    Integration scheme:
        u(t + dt) = u(t) + dt * f(u(t))

    where f(u(t)) = du/dt computed through model autograd.

    Suitable for smooth, well-behaved problems where simplicity
    is preferred over accuracy. Use RK4ODESolver for better accuracy
    at the same dt.

    Example:
        solver = EulerODESolver()
        output = solver.step(model, batch, dt=0.01)
    """

    def step(
        self,
        model: PhysicsModel,
        batch: FeatureBatch,
        dt:    float,
    ) -> SolverOutput:
        """
        Performs one Euler integration step.

        Args:
            model: PhysicsModel for computing du/dt.
            batch: Current FeatureBatch at time t.
            dt:    Time step size.

        Returns:
            SolverOutput with quantities containing 'u_next'
            and residuals containing 'step_size'.
        """
        dudt  = _get_time_derivative(model, batch)
        u_cur = model.forward(batch).pred
        u_new = u_cur + dt * dudt

        return SolverOutput(
            quantities={"u_next": u_new, "dudt": dudt},
            residuals={"step_size": torch.tensor(dt)},
            solver_info=SolverInfo(
                solver_type="EulerODE",
                converged=True,
                iterations=1,
            ),
        )

    def solve(
        self,
        output: ModelOutput,
        batch:  FeatureBatch,
    ) -> SolverOutput:
        """
        Post-processing without autograd.
        Returns model output quantities directly.
        """
        return SolverOutput(
            quantities={"u": output.pred},
            residuals={},
            solver_info=SolverInfo(solver_type="EulerODE"),
        )

    def solve_with_grad(
        self,
        model: PhysicsModel,
        batch: FeatureBatch,
    ) -> SolverOutput:
        """
        Full Euler solve with autograd through model.
        Calls step() with a default dt=0.01.
        Override dt by calling step() directly.
        """
        return self.step(model, batch, dt=0.01)


# ════════════════════════════════════════════════════════════════════════════
# 2. RK4ODESolver
# ════════════════════════════════════════════════════════════════════════════

class RK4ODESolver(ODESolver):
    """
    Fourth-order Runge-Kutta ODE solver with fixed step size.

    Integration scheme:
        k1 = f(u(t))
        k2 = f(u(t) + dt/2 * k1)
        k3 = f(u(t) + dt/2 * k2)
        k4 = f(u(t) + dt   * k3)
        u(t+dt) = u(t) + dt/6 * (k1 + 2k2 + 2k3 + k4)

    Significantly more accurate than Euler at the same dt.
    Preferred for problems where accuracy matters and dt is fixed.

    Example:
        solver = RK4ODESolver()
        output = solver.step(model, batch, dt=0.01)
    """

    def step(
        self,
        model: PhysicsModel,
        batch: FeatureBatch,
        dt:    float,
    ) -> SolverOutput:
        """
        Performs one RK4 integration step.

        Args:
            model: PhysicsModel for computing du/dt.
            batch: Current FeatureBatch at time t.
            dt:    Time step size.

        Returns:
            SolverOutput with quantities containing 'u_next'.
        """
        u = model.forward(batch).pred

        # k1 = f(u(t))
        k1 = _get_time_derivative(model, batch)

        # k2 = f(u(t) + dt/2 * k1)
        batch_k2 = _advance_batch(batch, u + 0.5 * dt * k1, 0.5 * dt)
        k2       = _get_time_derivative(model, batch_k2)

        # k3 = f(u(t) + dt/2 * k2)
        batch_k3 = _advance_batch(batch, u + 0.5 * dt * k2, 0.5 * dt)
        k3       = _get_time_derivative(model, batch_k3)

        # k4 = f(u(t) + dt * k3)
        batch_k4 = _advance_batch(batch, u + dt * k3, dt)
        k4       = _get_time_derivative(model, batch_k4)

        # RK4 combination
        u_new = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return SolverOutput(
            quantities={"u_next": u_new, "k1": k1, "k4": k4},
            residuals={"step_size": torch.tensor(dt)},
            solver_info=SolverInfo(
                solver_type="RK4ODE",
                converged=True,
                iterations=4,
            ),
        )

    def solve(
        self,
        output: ModelOutput,
        batch:  FeatureBatch,
    ) -> SolverOutput:
        return SolverOutput(
            quantities={"u": output.pred},
            residuals={},
            solver_info=SolverInfo(solver_type="RK4ODE"),
        )

    def solve_with_grad(
        self,
        model: PhysicsModel,
        batch: FeatureBatch,
    ) -> SolverOutput:
        return self.step(model, batch, dt=0.01)


# ════════════════════════════════════════════════════════════════════════════
# 3. AdaptiveODESolver (RK45)
# ════════════════════════════════════════════════════════════════════════════

class AdaptiveODESolver(ODESolver):
    """
    Adaptive step size ODE solver using Runge-Kutta-Fehlberg (RK45).

    Computes both RK4 and RK5 solutions at each step and uses
    their difference as a local error estimate to adapt dt:

        error = ||u_RK5 - u_RK4|| / tol
        if error < 1: accept step, increase dt by factor s
        if error > 1: reject step, decrease dt by factor s
        s = 0.9 * (1 / error) ^ 0.2   (standard safety factor)

    step() returns a tuple (SolverOutput, new_dt) - the second
    value is the suggested dt for the next step (ADR-008).

    Args:
        dt_min: Minimum allowed step size. Raises if dt falls below.
        dt_max: Maximum allowed step size.
        safety: Safety factor for dt adjustment. Default: 0.9.

    Example:
        solver = AdaptiveODESolver()
        output, new_dt = solver.step(model, batch, dt=0.1, tol=1e-4)
        # Use new_dt for the next step
    """

    # Dormand-Prince RK45 coefficients
    _A = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    _C4 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    _C5 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]

    def __init__(
        self,
        dt_min: float = 1e-6,
        dt_max: float = 1.0,
        safety: float = 0.9,
    ):
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.safety = safety

    def step(
        self,
        model: PhysicsModel,
        batch: FeatureBatch,
        dt:    float,
        tol:   float = 1e-4,
    ) -> tuple[SolverOutput, float]:
        """
        Performs one adaptive RK45 step.

        Args:
            model: PhysicsModel for computing du/dt.
            batch: Current FeatureBatch at time t.
            dt:    Initial time step size.
            tol:   Error tolerance for step size control.

        Returns:
            tuple:
                SolverOutput with quantities 'u_next', 'error_norm'.
                new_dt: Suggested dt for the next step.

        Raises:
            ValueError: If dt falls below dt_min (step rejected too many times).
        """
        u   = model.forward(batch).pred
        accepted   = False
        iterations = 0

        while not accepted:
            iterations += 1

            # Compute 6 stage derivatives (Dormand-Prince)
            k1 = _get_time_derivative(model, batch)
            k2 = _get_time_derivative(model, _advance_batch(batch, u + dt * (1/5) * k1, dt/5))
            k3 = _get_time_derivative(model, _advance_batch(batch, u + dt * (3/40*k1 + 9/40*k2), dt*3/10))
            k4 = _get_time_derivative(model, _advance_batch(batch, u + dt * (44/45*k1 - 56/15*k2 + 32/9*k3), dt*4/5))
            k5 = _get_time_derivative(model, _advance_batch(batch, u + dt * (19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4), dt*8/9))
            k6 = _get_time_derivative(model, _advance_batch(batch, u + dt * (9017/3168*k1 - 355/33*k2 + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5), dt))

            # RK4 solution
            u_rk4 = u + dt * (
                35/384 * k1 + 500/1113 * k3 +
                125/192 * k4 - 2187/6784 * k5 +
                11/84 * k6
            )

            # RK5 solution (one extra stage)
            k7    = _get_time_derivative(model, _advance_batch(batch, u_rk4, dt))
            u_rk5 = u + dt * (
                5179/57600 * k1 + 7571/16695 * k3 +
                393/640 * k4 - 92097/339200 * k5 +
                187/2100 * k6 + 1/40 * k7
            )

            # Error estimate
            error_norm = float(torch.norm(u_rk5 - u_rk4) / (tol * u_rk4.numel() ** 0.5))

            if error_norm < 1.0:
                # Accept step
                accepted = True
                new_dt   = min(
                    self.dt_max,
                    dt * self.safety * (1.0 / max(error_norm, 1e-10)) ** 0.2
                )
            else:
                # Reject step - reduce dt
                dt = max(
                    self.dt_min,
                    dt * self.safety * (1.0 / error_norm) ** 0.25
                )
                if dt <= self.dt_min:
                    raise ValueError(
                        f"AdaptiveODESolver: dt fell below dt_min={self.dt_min}. "
                        f"Consider increasing tol or checking model stability."
                    )

        return (
            SolverOutput(
                quantities={"u_next": u_rk4, "error_norm": torch.tensor(error_norm)},
                residuals={"step_size": torch.tensor(dt)},
                solver_info=SolverInfo(
                    solver_type="AdaptiveRK45",
                    converged=True,
                    iterations=iterations,
                ),
            ),
            new_dt,
        )

    def solve(
        self,
        output: ModelOutput,
        batch:  FeatureBatch,
    ) -> SolverOutput:
        return SolverOutput(
            quantities={"u": output.pred},
            residuals={},
            solver_info=SolverInfo(solver_type="AdaptiveRK45"),
        )

    def solve_with_grad(
        self,
        model: PhysicsModel,
        batch: FeatureBatch,
    ) -> SolverOutput:
        """
        Calls step() with default dt=0.1 and tol=1e-4.
        Returns only SolverOutput - discards new_dt.
        Use step() directly when new_dt is needed.
        """
        output, _ = self.step(model, batch, dt=0.1, tol=1e-4)
        return output