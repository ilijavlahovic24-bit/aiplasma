"""
nuclear_equations.py
Equations for nuclear-material and radiation problems.

Lives in: solvers/nuclear_equations.py

Registered in REGISTRY on import. solvers/physics_constraints.py imports
this module at its very bottom, so REGISTRY always contains them.

Conventions (same as HasegawaWakatani):
    - spatial coordinate columns first, time last
    - every equation documents the channel order of pred
    - the residual has one column per equation
    - shapes and requires_grad are validated, with explicit errors
    - pointwise models only (ADR-015)
"""

import torch
from torch import Tensor

from solvers.physics_constraints import PhysicsEquation, REGISTRY, _grad


def _check_layout(name: str, coords: Tensor, pred: Tensor, n_coords: int, n_channels: int) -> None:
    if coords.shape[1] != n_coords:
        raise ValueError(
            f"[{name}] coords must have {n_coords} column(s), got {coords.shape[1]}."
        )
    if pred.shape[1] != n_channels:
        raise ValueError(
            f"[{name}] pred must have {n_channels} channel(s), got {pred.shape[1]}."
        )
    if not coords.requires_grad:
        raise RuntimeError(
            f"[{name}] coords.requires_grad must be True, and pred must be "
            f"computed from these coords."
        )


def _second_derivative(grad_f: Tensor, coords: Tensor, axis: int) -> Tensor:
    """d2f/dx_axis2 from grad_f = grad(f), shape (N, 1)."""
    return _grad(grad_f[:, axis:axis + 1], coords, allow_zero=True)[:, axis:axis + 1]


# ════════════════════════════════════════════════════════════════════════════
# Nuclear materials
# ════════════════════════════════════════════════════════════════════════════

class HeatConductionSource1D(PhysicsEquation):
    """
    1D heat conduction with a volumetric source (fuel rod, cladding):

        rho_cp dT/dt = k d2T/dx2 + q

    Parameters: k (conductivity), rho_cp (volumetric heat capacity),
                q (volumetric source: float or tensor of shape (N, 1)).
    Layout:     coords (x, t); pred (T,); residual (N, 1).
    """

    def name(self) -> str:
        return "heat_conduction_source_1d"

    def expected_params(self) -> list[str]:
        return ["k", "rho_cp", "q"]

    def description(self) -> str:
        return "1D heat conduction with a source: rho_cp dT/dt = k d2T/dx2 + q"

    def residual(self, coords: Tensor, pred: Tensor, params: dict) -> Tensor:
        self.validate_params(params)
        _check_layout(self.name(), coords, pred, 2, 1)

        g     = _grad(pred, coords)
        T_t   = g[:, 1:2]
        T_xx  = _second_derivative(g, coords, 0)

        return params["rho_cp"] * T_t - params["k"] * T_xx - params["q"]


class FissionGasBooth(PhysicsEquation):
    """
    Booth fission-gas diffusion in an equivalent sphere:

        dc/dt = D (d2c/dr2 + (2/r) dc/dr) - lam c + beta

    Parameters: D (diffusivity), lam (decay constant), beta (production rate).
    Layout:     coords (r, t) with r >= 0; pred (c,).

    The residual is written for u = r c, where the equation becomes a plain
    1D diffusion with a source:

        du/dt = D d2u/dr2 - lam u + beta r

    This removes the 2/r singularity at r = 0. The returned residual (N, 1)
    is therefore r times the residual of the equation for c.
    A perfect sink at the grain boundary is c(a, t) = 0, i.e. u(a, t) = 0.
    """

    def name(self) -> str:
        return "fission_gas_booth"

    def expected_params(self) -> list[str]:
        return ["D", "lam", "beta"]

    def description(self) -> str:
        return (
            "Booth fission-gas diffusion (equivalent sphere), residual for u = r c:\n"
            "  du/dt = D d2u/dr2 - lam u + beta r"
        )

    def residual(self, coords: Tensor, pred: Tensor, params: dict) -> Tensor:
        self.validate_params(params)
        _check_layout(self.name(), coords, pred, 2, 1)

        r     = coords[:, 0:1]
        u     = r * pred
        g     = _grad(u, coords)
        u_t   = g[:, 1:2]
        u_rr  = _second_derivative(g, coords, 0)

        return u_t - params["D"] * u_rr + params["lam"] * u - params["beta"] * r


class PointDefectRateTheory1D(PhysicsEquation):
    """
    Point-defect rate theory with diffusion (irradiated metals):

        dC_v/dt = D_v d2C_v/dx2 + K0 - K_iv C_i C_v - K_v C_v
        dC_i/dt = D_i d2C_i/dx2 + K0 - K_iv C_i C_v - K_i C_i

    Parameters:
        D_v, D_i:  vacancy and interstitial diffusivities
        K0:        defect production rate (for example dpa rate times efficiency)
        K_iv:      recombination coefficient
        K_v, K_i:  sink absorption rates [1/time] (they already include the
                   diffusivity: K_v = D_v * k_v^2)
    Layout:   coords (x, t); pred (C_v, C_i); residual (N, 2) in the same order.

    The system is stiff when the diffusivities and rate constants differ by
    orders of magnitude. Normalize with scales() before training (ADR-017).
    Uniform stationary limit: K_v C_v = K_i C_i, and C_v is the positive root
    of (K_iv K_v / K_i) C_v^2 + K_v C_v - K0 = 0.
    """

    def name(self) -> str:
        return "point_defect_rate_theory_1d"

    def expected_params(self) -> list[str]:
        return ["D_v", "D_i", "K0", "K_iv", "K_v", "K_i"]

    def description(self) -> str:
        return (
            "Point-defect rate theory (fields C_v, C_i):\n"
            "  dC_v/dt = D_v C_v_xx + K0 - K_iv C_i C_v - K_v C_v\n"
            "  dC_i/dt = D_i C_i_xx + K0 - K_iv C_i C_v - K_i C_i"
        )

    def residual(self, coords: Tensor, pred: Tensor, params: dict) -> Tensor:
        self.validate_params(params)
        _check_layout(self.name(), coords, pred, 2, 2)

        C_v, C_i = pred[:, 0:1], pred[:, 1:2]
        g_v, g_i = _grad(C_v, coords), _grad(C_i, coords)
        recombination = params["K_iv"] * C_i * C_v

        res_v = (
            g_v[:, 1:2] - params["D_v"] * _second_derivative(g_v, coords, 0)
            - params["K0"] + recombination + params["K_v"] * C_v
        )
        res_i = (
            g_i[:, 1:2] - params["D_i"] * _second_derivative(g_i, coords, 0)
            - params["K0"] + recombination + params["K_i"] * C_i
        )
        return torch.cat([res_v, res_i], dim=1)


# ════════════════════════════════════════════════════════════════════════════
# Radiation
# ════════════════════════════════════════════════════════════════════════════

class NeutronDiffusion1D(PhysicsEquation):
    """
    One-group time-dependent neutron diffusion:

        (1/v) dphi/dt = D d2phi/dx2 - Sigma_a phi + nu_Sigma_f phi + S

    Parameters: v (neutron speed), D (diffusion coefficient),
                Sigma_a (absorption), nu_Sigma_f (fission production);
                optional S (source, default 0.0).
    Layout:     coords (x, t); pred (phi,); residual (N, 1).

    Slab eigenmode: phi = cos(B x) exp(w t) with
    w = v (nu_Sigma_f - Sigma_a - D B^2).
    """

    def name(self) -> str:
        return "neutron_diffusion_1d"

    def expected_params(self) -> list[str]:
        return ["v", "D", "Sigma_a", "nu_Sigma_f"]

    def description(self) -> str:
        return (
            "One-group neutron diffusion: "
            "(1/v) dphi/dt = D d2phi/dx2 - Sigma_a phi + nu_Sigma_f phi + S"
        )

    def residual(self, coords: Tensor, pred: Tensor, params: dict) -> Tensor:
        self.validate_params(params)
        _check_layout(self.name(), coords, pred, 2, 1)

        S       = params.get("S", 0.0)
        g       = _grad(pred, coords)
        phi_t   = g[:, 1:2]
        phi_xx  = _second_derivative(g, coords, 0)

        return (
            phi_t / params["v"]
            - params["D"] * phi_xx
            + (params["Sigma_a"] - params["nu_Sigma_f"]) * pred
            - S
        )


class BatemanChain(PhysicsEquation):
    """
    Bateman decay chain, nuclide k decays into nuclide k + 1:

        dN_k/dt = lam_{k-1} N_{k-1} - lam_k N_k + S_k      (N_{-1} term is 0)

    Parameters: lambdas (decay constants, one per nuclide, list or tensor);
                optional S (source, default 0.0).
    Layout:     coords (t,), a single column; pred (N_0, ..., N_{K-1});
                residual (N, K).
    """

    def name(self) -> str:
        return "bateman_chain"

    def expected_params(self) -> list[str]:
        return ["lambdas"]

    def description(self) -> str:
        return "Bateman decay chain: dN_k/dt = lam_{k-1} N_{k-1} - lam_k N_k + S_k"

    def residual(self, coords: Tensor, pred: Tensor, params: dict) -> Tensor:
        self.validate_params(params)

        lam = torch.as_tensor(params["lambdas"], dtype=pred.dtype, device=pred.device)
        if lam.ndim != 1:
            raise ValueError(f"[{self.name()}] lambdas must be one-dimensional.")
        _check_layout(self.name(), coords, pred, 1, lam.shape[0])

        dN_dt = torch.cat(
            [_grad(pred[:, k:k + 1], coords)[:, 0:1] for k in range(lam.shape[0])],
            dim=1,
        )
        decay = lam * pred                                              # (N, K)
        gain  = torch.cat([torch.zeros_like(decay[:, :1]), decay[:, :-1]], dim=1)

        return dN_dt - gain + decay - params.get("S", 0.0)


class RadiativeLoss1D(PhysicsEquation):
    """
    Electron energy balance of a plasma with radiative losses. This is a
    toy model: uniform density and constant conductivity.

        1.5 n_e T_unit_J dT/dt = kappa d2T/dx2 + P_heat - P_br - P_line

        P_br   = c_br Z_eff n_e^2 sqrt(T)         (bremsstrahlung)
        P_line = n_e n_imp L_z(T)                 (impurity line radiation)

    Required: n_e (electron density), kappa (thermal conductivity).
    Optional: Z_eff (1.0), P_heat (0.0), n_imp (0.0),
              L_z (callable T -> cooling rate, default None: no line radiation),
              c_br (5.35e-37), T_unit_J (1.602176634e-16).

    Default units: T in keV, n_e in m^-3, powers in W m^-3, so
    T_unit_J = J per keV. c_br = 5.35e-37 W m^3 keV^-1/2 is the textbook
    bremsstrahlung coefficient (Wesson, Tokamaks). Verify it against
    your source before publication, or pass your own value.
    Line radiation takes L_z from the user: no atomic database is bundled.

    Layout: coords (x, t); pred (T,); residual (N, 1).
    Uniform bremsstrahlung cooling: sqrt(T) = sqrt(T0) - A t / 2 with
    A = c_br Z_eff n_e / (1.5 T_unit_J).
    """

    def name(self) -> str:
        return "radiative_loss_1d"

    def expected_params(self) -> list[str]:
        return ["n_e", "kappa"]

    def description(self) -> str:
        return (
            "Electron energy balance with radiative losses:\n"
            "  1.5 n_e T_unit_J dT/dt = kappa T_xx + P_heat - P_br - P_line\n"
            "  P_br = c_br Z_eff n_e^2 sqrt(T),  P_line = n_e n_imp L_z(T)"
        )

    def residual(self, coords: Tensor, pred: Tensor, params: dict) -> Tensor:
        self.validate_params(params)
        _check_layout(self.name(), coords, pred, 2, 1)

        n_e      = params["n_e"]
        Z_eff    = params.get("Z_eff", 1.0)
        c_br     = params.get("c_br", 5.35e-37)
        T_unit_J = params.get("T_unit_J", 1.602176634e-16)
        P_heat   = params.get("P_heat", 0.0)
        n_imp    = params.get("n_imp", 0.0)
        L_z      = params.get("L_z")

        g    = _grad(pred, coords)
        T_t  = g[:, 1:2]
        T_xx = _second_derivative(g, coords, 0)

        T_pos  = pred.clamp(min=1e-12)           # keeps sqrt and L_z(T) defined
        p_br   = c_br * Z_eff * n_e ** 2 * torch.sqrt(T_pos)
        p_line = n_e * n_imp * L_z(T_pos) if L_z is not None else 0.0

        return 1.5 * n_e * T_unit_J * T_t - params["kappa"] * T_xx - P_heat + p_br + p_line


for _equation in (
    HeatConductionSource1D(),
    FissionGasBooth(),
    PointDefectRateTheory1D(),
    NeutronDiffusion1D(),
    BatemanChain(),
    RadiativeLoss1D(),
):
    REGISTRY.register(_equation)