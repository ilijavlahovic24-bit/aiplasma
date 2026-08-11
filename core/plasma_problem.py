"""
plasma_problem.py
Domain layer for plasma physics problems in AIPlasma framework.

Lives in: core/plasma_problem.py

User inherits PlasmaPhysicsProblem for all plasma problems:
example:
    class TokamakTurbulence(PlasmaPhysicsProblem):
        equation = REGISTRY.get("hasegawa_wakatani")

        def boundary_conditions(self, batch, pred):
            ...
"""

from abc import abstractmethod
from typing import Optional
import torch
from torch import Tensor

from core.base_problem import PhysicsProblem
from data.preprocessing.feature_pipeline import FeatureBatch
from data.sources.physical_tensor import (
    PhysicalTensor, CoordinateSystem, UnitSystem, Domain
)
from solvers.physics_constraints import REGISTRY, PhysicsEquation


# ════════════════════════════════════════════════════════════════════════════
# 1. PlasmaPhysicsProblem
# ════════════════════════════════════════════════════════════════════════════

class PlasmaPhysicsProblem(PhysicsProblem):
    """
    Domain layer for plasma physics problems.

    It inherits PhysicsProblem and adds plasma-specific abstractions
    shared by all plasma problems:
    - Bohm normalization of units
    - Toroidal coordinate system as default
    - Plasma operators (Poisson bracket, Grad-Shafranov)
    - Validation of plasma-specific parameters

    User inherits PlasmaPhysicsProblem, not PhysicsProblem directly.
    It must define:
    - equation: PhysicsEquation from REGISTRY
    - boundary_conditions() method

    Can override:
    - bohm_params() for specific experimental conditions
    - physics_params() for PDE parameters
    - pde_residual() for custom logic (default delegates to equation)

    Example:
    class TokamakTurbulence(PlasmaPhysicsProblem):
    equation = REGISTRY.get("hasegawa_wakatani")

    def physics_params(self) -> dict:
    return {"D": 0.1, "C": 1.0, "nu": 0.01}

    def boundary_conditions(self, batch, pred) -> Tensor:
    bc_pred = pred[batch.boundary_mask]
    bc_true = batch.fields[batch.boundary_mask]
    return bc_pred - bc_true
    """

    # ── User defines per class ────────────────────────────────────
    equation:     Optional[PhysicsEquation] = None
    coord_system: CoordinateSystem          = CoordinateSystem.TOROIDAL

    # ── User MUST implement ───────────────────────────────────────

    @abstractmethod
    def boundary_conditions(self, batch: FeatureBatch, pred: Tensor) -> Tensor:
        ...

    # ── User can override ───────────────────────────────────────

    def pde_residual(self, batch: FeatureBatch, pred: Tensor) -> Tensor:
        if self.equation is None:
            raise ValueError(
                f"[{self.__class__.__name__}] equation not defined. "
                f"Add: equation = REGISTRY.get('equation_name')"
            )

        coords = batch.coords.requires_grad_(True)
        params = self.physics_params()
        self.equation.validate_params(params)

        return self.equation.residual(coords, pred, params)

    def bohm_params(self) -> dict:
        """
        Referentne vrednosti za Bohm normalizaciju.

        Default vrednosti su tipični tokamak uslovi.
        Override za specifičan eksperiment.

        Returns:
            Dict sa:
                T_e: Temperatura elektrona [eV]
                B:   Magnetno polje [T]
                n_0: Referentna gustina [m⁻³]
                m_i: Masa iona [kg] (default: deuterijum)

        Primer override-a:
            def bohm_params(self) -> dict:
                return {"T_e": 2000.0, "B": 5.3, "n_0": 1e20, "m_i": 3.34e-27}
        """
        return {
            "T_e":  1000.0,    # eV — tipična vrednost za mid-size tokamak
            "B":    2.0,       # T  — magnetno polje
            "n_0":  1e19,      # m⁻³ — referentna gustina
            "m_i":  3.34e-27,  # kg — masa deuterijum iona
        }

    def physics_params(self) -> dict:
        """
        PDE parametri. Override u subklasi za konkretan problem.
        Odvojeno od bohm_params() — ADR-005.
        """
        return {}

    # ── Bohm normalization ───────────────────────────────────────────────────

    def to_bohm_units(self, tensor: PhysicalTensor) -> PhysicalTensor:
        """
        Konvertuje PhysicalTensor iz SI u Bohm normalizovane jedinice.

        Bohm jedinice:
            Dužina:   ρ_s = sqrt(T_e · m_i) / (e · B)  [Larmor radius]
            Brzina:   c_s = sqrt(T_e / m_i)              [zvučna brzina]
            Vreme:    t_B = ρ_s / c_s
            Gustina:  normalizovano na n_0

        Args:
            tensor: PhysicalTensor u SI jedinicama.

        Returns:
            PhysicalTensor u Bohm normalizovanim jedinicama.
        """
        bp   = self.bohm_params()
        T_e  = bp["T_e"]
        B    = bp["B"]
        n_0  = bp["n_0"]
        m_i  = bp["m_i"]
        e    = 1.602e-19   # elementarno naelektrisanje [C]

        # Bohm (Larmor) radius: ρ_s = sqrt(T_e [J] * m_i) / (e * B)
        T_e_J  = T_e * e
        rho_s  = (T_e_J * m_i) ** 0.5 / (e * B)

        # Normalizacija koordinata: x → x / ρ_s
        coords_norm = tensor.coordinates.clone()
        coords_norm[:, :-1] = coords_norm[:, :-1] / rho_s  # prostorne dim.

        # Zvučna brzina: c_s = sqrt(T_e [J] / m_i)
        c_s   = (T_e_J / m_i) ** 0.5
        t_B   = rho_s / c_s

        # Normalizacija vremena: t → t / t_B
        coords_norm[:, -1] = coords_norm[:, -1] / t_B

        # Normalizacija vrednosti: n → n / n_0
        values_norm = tensor.values / n_0

        return PhysicalTensor(
            values=values_norm,
            coordinates=coords_norm,
            units=UnitSystem(
                spatial="bohm",
                temporal="normalized",
                field="normalized",
            ),
            coord_system=tensor.coord_system,
            domain=tensor.domain,
            metadata={
                **tensor.metadata,
                "bohm_normalized": True,
                "rho_s": rho_s,
                "c_s":   c_s,
                "t_B":   t_B,
            },
        )

    def from_bohm_units(self, tensor: PhysicalTensor) -> PhysicalTensor:
        """
        Inverz to_bohm_units() — vraća iz Bohm u SI jedinice.
        Korisno za interpretaciju rezultata.

        Args:
            tensor: PhysicalTensor u Bohm normalizovanim jedinicama.

        Returns:
            PhysicalTensor u SI jedinicama.

        Raises:
            ValueError: Ako tensor nije Bohm normalizovan.
        """
        if not tensor.metadata.get("bohm_normalized", False):
            raise ValueError(
                "from_bohm_units(): tensor nije Bohm normalizovan. "
                "Pozovi to_bohm_units() prvo."
            )

        rho_s = tensor.metadata["rho_s"]
        c_s   = tensor.metadata["c_s"]
        t_B   = tensor.metadata["t_B"]
        n_0   = self.bohm_params()["n_0"]

        coords_si = tensor.coordinates.clone()
        coords_si[:, :-1] = coords_si[:, :-1] * rho_s
        coords_si[:, -1]  = coords_si[:, -1]  * t_B

        values_si = tensor.values * n_0

        bp = self.bohm_params()
        return PhysicalTensor(
            values=values_si,
            coordinates=coords_si,
            units=UnitSystem(
                spatial="m",
                temporal="s",
                field="m⁻³",
            ),
            coord_system=tensor.coord_system,
            domain=tensor.domain,
            metadata={
                **tensor.metadata,
                "bohm_normalized": False,
            },
        )

    # ── Plasma operatori ─────────────────────────────────────────────────────

    def poisson_bracket(
        self,
        f:      Tensor,
        g:      Tensor,
        coords: Tensor,
    ) -> Tensor:
        """
        Poisson bracket: [f, g] = ∂f/∂x·∂g/∂y - ∂f/∂y·∂g/∂x

        Koristi se u Hasegawa-Wakatani i sličnim sistemima.
        Zahteva 2D prostorne koordinate (x, y).

        Args:
            f:      Tensor oblika (N, 1).
            g:      Tensor oblika (N, 1).
            coords: Koordinate oblika (N, D) sa D >= 2.
                    coords[:, 0] = x, coords[:, 1] = y.

        Returns:
            Poisson bracket [f, g], oblik (N, 1).
        """
        if coords.shape[1] < 2:
            raise ValueError(
                "poisson_bracket() zahteva najmanje 2 prostorne koordinate. "
                f"Dobijeno: {coords.shape[1]} dimenzija."
            )

        ones = torch.ones_like(f)

        grad_f = torch.autograd.grad(
            f, coords, grad_outputs=ones,
            create_graph=True, retain_graph=True
        )[0]
        grad_g = torch.autograd.grad(
            g, coords, grad_outputs=ones,
            create_graph=True, retain_graph=True
        )[0]

        df_dx = grad_f[:, 0:1]
        df_dy = grad_f[:, 1:2]
        dg_dx = grad_g[:, 0:1]
        dg_dy = grad_g[:, 1:2]

        return df_dx * dg_dy - df_dy * dg_dx

    def grad_shafranov_op(
        self,
        psi:    Tensor,
        coords: Tensor,
    ) -> Tensor:
        """
        Grad-Shafranov operator za toroidal geometriju.

        Δ*ψ = R · ∂/∂R(1/R · ∂ψ/∂R) + ∂²ψ/∂Z²

        gde je:
            ψ — poloidalni magnetni fluks
            R — toroidalni radius
            Z — vertikalna koordinata

        Args:
            psi:    Magnetni fluks, oblik (N, 1).
            coords: Koordinate oblika (N, D).
                    coords[:, 0] = R, coords[:, 1] = Z.

        Returns:
            Grad-Shafranov operator primenjen na psi, oblik (N, 1).
        """
        ones = torch.ones_like(psi)
        R    = coords[:, 0:1]

        grad_psi = torch.autograd.grad(
            psi, coords, grad_outputs=ones,
            create_graph=True, retain_graph=True
        )[0]

        dpsi_dR = grad_psi[:, 0:1]
        dpsi_dZ = grad_psi[:, 1:2]

        # ∂/∂R(1/R · ∂ψ/∂R) = 1/R · ∂²ψ/∂R² - 1/R² · ∂ψ/∂R
        d2psi_dR2 = torch.autograd.grad(
            dpsi_dR, coords, grad_outputs=ones,
            create_graph=True, retain_graph=True
        )[0][:, 0:1]

        d2psi_dZ2 = torch.autograd.grad(
            dpsi_dZ, coords, grad_outputs=ones,
            create_graph=True, retain_graph=True
        )[0][:, 1:2]

        return R * (d2psi_dR2 / R - dpsi_dR / (R ** 2)) + d2psi_dZ2

    # ── Validacija ───────────────────────────────────────────────────────────

    def validate_plasma(self, batch: FeatureBatch) -> None:
        """
        Plasma-specific validation of FeatureBatch.
        """
        # Opšta validacija iz PhysicsProblem
        self.validate(batch)

        if self.equation is None:
            raise ValueError(
                f"[{self.__class__.__name__}] equation not defined."
            )

        # Parametri moraju pokriti sve što equation zahteva
        self.equation.validate_params(self.physics_params())