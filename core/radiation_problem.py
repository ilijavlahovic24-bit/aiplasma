"""
radiation_problem.py
Domain layer for radiation problems in AIPlasma (ADR-016).

Lives in: core/radiation_problem.py

Sibling of PlasmaPhysicsProblem. Covers radiation transport and decay and
plasma radiative losses. Equations available for this domain:
neutron_diffusion_1d, bateman_chain, radiative_loss_1d.

    class SlabReactor(RadiationProblem):
        equation = REGISTRY.get("neutron_diffusion_1d")

        def physics_params(self) -> dict:          # illustrative values
            return {"v": 2.2e5, "D": 0.9, "Sigma_a": 0.07, "nu_Sigma_f": 0.09}

For a decay chain the coordinates have a single column (time), so
scales() "length" is unused and "value" may list one scale per nuclide.
"""

import math
from typing import Optional

from torch import Tensor

from core.base_problem import PhysicsProblem
from core.boundary_conditions import dirichlet
from core.normalization import denormalize_tensor, normalize_tensor
from data.preprocessing.feature_pipeline import FeatureBatch
from data.sources.physical_tensor import PhysicalTensor
from solvers.physics_constraints import PhysicsEquation


# ── Helpers ──────────────────────────────────────────────────────────────────

def half_life_to_lambda(half_life):
    """Decay constant: lambda = ln 2 / T_half. Works for floats and tensors."""
    return math.log(2.0) / half_life


def mean_free_path(sigma):
    """Mean free path 1 / Sigma for a macroscopic cross-section Sigma."""
    return 1.0 / sigma


# ── Problem ──────────────────────────────────────────────────────────────────

class RadiationProblem(PhysicsProblem):
    """
    Domain layer for radiation problems.

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

    def validate_radiation(self, batch: FeatureBatch) -> None:
        """Radiation-specific validation of a FeatureBatch."""
        self.validate(batch)

        if self.equation is None:
            raise ValueError(f"[{self.__class__.__name__}] equation not defined.")

        self.equation.validate_params({**batch.physics_params, **self.physics_params()})