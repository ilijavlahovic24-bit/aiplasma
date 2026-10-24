"""
material_problem.py
Domain layer for nuclear-material problems in AIPlasma (ADR-016).

Lives in: core/material_problem.py

Sibling of PlasmaPhysicsProblem. The user subclasses it, sets `equation`
from REGISTRY and overrides physics_params() and, if needed, scales() and
boundary_conditions().

    class FuelRod(NuclearMaterialProblem):
        equation = REGISTRY.get("heat_conduction_source_1d")

        def physics_params(self) -> dict:          # illustrative values
            return {"k": 3.0, "rho_cp": 3.0e6, "q": 3.0e8}

        def scales(self) -> dict:
            return {"length": 5e-3, "time": 10.0, "value": 1000.0}

Equations available for this domain: heat_conduction_source_1d,
fission_gas_booth, point_defect_rate_theory_1d.

Nondimensionalizing the equation parameters with scales() is up to the user
(ADR-017).
"""

import math
from typing import Optional

import torch
from torch import Tensor

from core.base_problem import PhysicsProblem
from core.boundary_conditions import dirichlet
from core.normalization import denormalize_tensor, normalize_tensor
from data.preprocessing.feature_pipeline import FeatureBatch
from data.sources.physical_tensor import PhysicalTensor
from solvers.physics_constraints import PhysicsEquation

BOLTZMANN_EV = 8.617333262e-5    # eV / K


# ── Helpers ──────────────────────────────────────────────────────────────────

def arrhenius(prefactor, activation_energy_eV, temperature_K):
    """
    Thermally activated rate, e.g. a diffusivity: D = D0 exp(-E / (k_B T)).
    Works for floats and tensors.
    """
    exponent = -activation_energy_eV / (BOLTZMANN_EV * temperature_K)
    if isinstance(exponent, Tensor):
        return prefactor * torch.exp(exponent)
    return prefactor * math.exp(exponent)


def dpa_from_flux(flux, sigma_d, time=1.0):
    """
    Displacements per atom: dpa = sigma_d * flux * time.
    With time = 1 this is the dpa rate. Units: flux [m^-2 s^-1],
    sigma_d [m^2] (1 barn = 1e-28 m^2), time [s].
    """
    return sigma_d * flux * time


# ── Problem ──────────────────────────────────────────────────────────────────

class NuclearMaterialProblem(PhysicsProblem):
    """
    Domain layer for nuclear-material problems.

    User defines:
        equation:           PhysicsEquation from REGISTRY
        physics_params():   PDE parameters (merged over batch.physics_params)
    User can override:
        scales():             reference scales for normalization
        boundary_conditions(): default is Dirichlet from data (fields)
        n_fidelity_levels:    number of fidelity levels (default 1)
    """

    equation:          Optional[PhysicsEquation] = None
    n_fidelity_levels: int = 1

    # ── User can override ────────────────────────────────────────────────────

    def physics_params(self) -> dict:
        return {}

    def scales(self) -> dict:
        """
        Reference scales: "length", "time" and "value" (a float, or one
        scale per channel). Default: no scaling.
        """
        return {"length": 1.0, "time": 1.0, "value": 1.0}

    def pde_residual(self, batch: FeatureBatch, pred: Tensor) -> Tensor:
        if self.equation is None:
            raise ValueError(
                f"[{self.__class__.__name__}] equation not defined. "
                f"Add: equation = REGISTRY.get('equation_name')"
            )
        params = {**batch.physics_params, **self.physics_params()}
        self.equation.validate_params(params)
        return self.equation.residual(batch.coords, pred, params)

    def boundary_conditions(self, batch: FeatureBatch, pred: Tensor) -> Tensor:
        """Default: match the data at boundary points. Override freely."""
        return dirichlet(pred, batch.fields, batch.boundary_mask)

    def fidelity_weights(self) -> list[float]:
        """Uniform weights over n_fidelity_levels."""
        n = self.n_fidelity_levels
        return [1.0 / n] * n

    # ── Normalization (ADR-017) ──────────────────────────────────────────────

    def to_normalized_units(self, tensor: PhysicalTensor) -> PhysicalTensor:
        return normalize_tensor(tensor, self.scales())

    def from_normalized_units(self, tensor: PhysicalTensor) -> PhysicalTensor:
        return denormalize_tensor(tensor)

    # ── Validation ───────────────────────────────────────────────────────────

    def validate_material(self, batch: FeatureBatch) -> None:
        """Material-specific validation of a FeatureBatch."""
        self.validate(batch)

        if self.equation is None:
            raise ValueError(f"[{self.__class__.__name__}] equation not defined.")

        self.equation.validate_params({**batch.physics_params, **self.physics_params()})