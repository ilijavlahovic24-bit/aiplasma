"""
physics_constraints.py
PDE registar i konkretne fizičke jednačine za AIPlasma framework.

Živi u: solvers/physics_constraints.py

Upotreba:
    from solvers.physics_constraints import REGISTRY

    eq  = REGISTRY.get("heat_equation_1d")
    res = eq.residual(coords, pred, params={"alpha": 0.01})

    # Registracija custom jednačine:
    REGISTRY.register(MyEquation())
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import Tensor


# ════════════════════════════════════════════════════════════════════════════
# 1. PhysicsEquation — apstraktna baza
# ════════════════════════════════════════════════════════════════════════════

class PhysicsEquation(ABC):
    """
    Apstraktna reprezentacija jedne fizičke jednačine u PDE registru.

    Svaka jednačina implementira:
        - name()            → jedinstveni identifikator u registru
        - expected_params() → lista parametara koje jednačina zahteva
        - residual()        → PDE residual za PINN trening
        - description()     → matematički opis za dokumentaciju
    """

    @abstractmethod
    def name(self) -> str:
        """
        Jedinstveni identifikator jednačine u registru.
        Konvencija: snake_case, npr. 'heat_equation_1d'.
        """
        ...

    @abstractmethod
    def expected_params(self) -> list[str]:
        """
        Lista imena parametara koje jednačina očekuje u params dict-u.
        Primer: ['alpha'] za heat equation.
        """
        ...

    @abstractmethod
    def residual(
        self,
        coords: Tensor,
        pred:   Tensor,
        params: dict,
    ) -> Tensor:
        """
        Računa PDE residual.

        Autograd mora biti aktivan na coords kada se poziva iz
        solve_with_grad() — coords.requires_grad mora biti True.

        Args:
            coords: Prostorno-vremenske koordinate, oblik (N, D).
            pred:   Predikcija modela, oblik (N, F).
            params: Dict fizičkih parametara (iz physics_params()).

        Returns:
            Residual tensor, oblik (N,) ili (N, F).
            Idealno rešenje daje residual == 0 svuda.
        """
        ...

    # ── Framework pruža ──────────────────────────────────────────────────────

    def validate_params(self, params: dict) -> bool:
        """
        Proverava da params sadrži sve expected_params.

        Args:
            params: Dict parametara za validaciju.

        Returns:
            True ako su svi parametri prisutni.

        Raises:
            ValueError: Ako nedostaje neki parametar.
        """
        missing = [p for p in self.expected_params() if p not in params]
        if missing:
            raise ValueError(
                f"[{self.name()}] Nedostaju parametri: {missing}. "
                f"Očekivano: {self.expected_params()}"
            )
        return True

    def description(self) -> str:
        """
        Matematički opis jednačine.
        Override u subklasi za konkretan opis.
        """
        return f"{self.__class__.__name__} — params: {self.expected_params()}"

    def __repr__(self) -> str:
        return f"PhysicsEquation(name='{self.name()}')"


# ════════════════════════════════════════════════════════════════════════════
# 2. PDERegistry
# ════════════════════════════════════════════════════════════════════════════

class PDERegistry:
    """
    Singleton registar koji čuva sve poznate PhysicsEquation instance.

    Korisnik može da registruje custom jednačine:
        REGISTRY.register(MyCustomEquation())
        eq = REGISTRY.get("my_custom_equation")

    Interne jednačine (HeatEquation1D, DriftDiffusion1D, HasegawaWakatani)
    se registruju automatski na dnu ovog fajla.
    """

    def __init__(self):
        self._equations: dict[str, PhysicsEquation] = {}

    def register(self, equation: PhysicsEquation) -> None:
        """
        Registruje novu jednačinu.

        Args:
            equation: PhysicsEquation instanca za registraciju.

        Raises:
            ValueError: Ako jednačina sa istim imenom već postoji.
        """
        n = equation.name()
        if n in self._equations:
            raise ValueError(
                f"PDERegistry: jednačina '{n}' već postoji. "
                f"Koristite drugo ime ili uklonite postojeću."
            )
        self._equations[n] = equation

    def get(self, name: str) -> PhysicsEquation:
        """
        Vraća jednačinu po imenu.

        Args:
            name: Ime jednačine (isto kao name() metoda).

        Returns:
            PhysicsEquation instanca.

        Raises:
            KeyError: Ako jednačina nije registrovana.
        """
        if name not in self._equations:
            available = self.list_all()
            raise KeyError(
                f"PDERegistry: jednačina '{name}' nije pronađena. "
                f"Dostupne: {available}"
            )
        return self._equations[name]

    def list_all(self) -> list[str]:
        """Vraća listu svih registrovanih imena."""
        return sorted(self._equations.keys())

    def unregister(self, name: str) -> None:
        """
        Uklanja jednačinu iz registra.
        Korisno za testove i dinamičku rekonfiguraciju.

        Args:
            name: Ime jednačine za uklanjanje.

        Raises:
            KeyError: Ako jednačina nije registrovana.
        """
        if name not in self._equations:
            raise KeyError(f"PDERegistry: '{name}' nije pronađena.")
        del self._equations[name]

    def __contains__(self, name: str) -> bool:
        return name in self._equations

    def __repr__(self) -> str:
        return f"PDERegistry(equations={self.list_all()})"


# ════════════════════════════════════════════════════════════════════════════
# 3. Konkretne jednačine
# ════════════════════════════════════════════════════════════════════════════

class HeatEquation1D(PhysicsEquation):
    """
    1D Heat equation: ∂u/∂t = α · ∂²u/∂x²

    Parametri:
        alpha: Toplotna difuzivnost [m²/s]

    Koordinate:
        coords[:, 0] = x  (prostorna koordinata)
        coords[:, 1] = t  (vremenska koordinata)
    """

    def name(self) -> str:
        return "heat_equation_1d"

    def expected_params(self) -> list[str]:
        return ["alpha"]

    def description(self) -> str:
        return "1D Heat Equation: ∂u/∂t = α · ∂²u/∂x²"

    def residual(self, coords: Tensor, pred: Tensor, params: dict) -> Tensor:
        self.validate_params(params)
        alpha = params["alpha"]

        if not coords.requires_grad:
            coords = coords.requires_grad_(True)

        # First-order gradients — single pass
        grads = torch.autograd.grad(
            pred, coords,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if grads is None:
            return torch.zeros_like(pred)

        du_dt = grads[:, 1:2]
        du_dx = grads[:, 0:1]

        # Second-order: ∂²u/∂x²
        grads2 = torch.autograd.grad(
            grads[:, 0:1].sum(), coords,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if grads2 is None:
            return du_dt

        d2u_dx2 = grads2[:, 0:1]
        return du_dt - alpha * d2u_dx2


class DriftDiffusion1D(PhysicsEquation):
    """
    1D Drift-Diffusion: ∂u/∂t + v · ∂u/∂x = D · ∂²u/∂x²

    Parametri:
        D: Koeficijent difuzije [m²/s]
        v: Drift brzina [m/s]

    Koordinate:
        coords[:, 0] = x  (prostorna koordinata)
        coords[:, 1] = t  (vremenska koordinata)
    """

    def name(self) -> str:
        return "drift_diffusion_1d"

    def expected_params(self) -> list[str]:
        return ["D", "v"]

    def description(self) -> str:
        return "1D Drift-Diffusion: ∂u/∂t + v·∂u/∂x = D·∂²u/∂x²"

    def residual(self, coords: Tensor, pred: Tensor, params: dict) -> Tensor:
        self.validate_params(params)
        D = params["D"]
        v = params["v"]

        if not coords.requires_grad:
            coords = coords.requires_grad_(True)

        # First-order gradients — single pass for ∂u/∂t and ∂u/∂x
        grads = torch.autograd.grad(
            pred, coords,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if grads is None:
            return torch.zeros_like(pred)

        du_dt = grads[:, 1:2]
        du_dx = grads[:, 0:1]

        # Second-order: ∂²u/∂x²
        grads2 = torch.autograd.grad(
            grads[:, 0:1].sum(), coords,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if grads2 is None:
            return du_dt + v * du_dx

        d2u_dx2 = grads2[:, 0:1]
        return du_dt + v * du_dx - D * d2u_dx2


def _grad(f: Tensor, coords: Tensor, allow_zero: bool = False) -> Tensor:
    """
    Gradient of f (N, 1) w.r.t. coords (N, D), shape (N, D).

    Uses the sum trick (grad_outputs = ones), so it is exact only for pointwise
    models, where f[i] depends on coords[i] alone. This is the same assumption
    the other equations in this module make.

    allow_zero: a missing graph means "derivative is zero" (valid for the
    second derivative of a linear model). For first derivatives it almost
    always means pred was not computed from coords, so it raises instead.
    """
    g = None
    if f.requires_grad:
        g = torch.autograd.grad(
            f, coords,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

    if g is None:
        if allow_zero:
            return torch.zeros_like(coords)
        raise RuntimeError(
            "pred is not connected to coords through autograd. Compute pred "
            "from the same coords tensor that has requires_grad=True."
        )
    return g


def _laplacian_2d(grad_f: Tensor, coords: Tensor) -> Tensor:
    """f_xx + f_yy from grad_f = ∇f of shape (N, D). Returns (N, 1)."""
    f_xx = _grad(grad_f[:, 0:1], coords, allow_zero=True)[:, 0:1]
    f_yy = _grad(grad_f[:, 1:2], coords, allow_zero=True)[:, 1:2]
    return f_xx + f_yy


def _bracket(grad_f: Tensor, grad_g: Tensor) -> Tensor:
    """Poisson bracket [f, g] = f_x g_y - f_y g_x from precomputed gradients."""
    return grad_f[:, 0:1] * grad_g[:, 1:2] - grad_f[:, 1:2] * grad_g[:, 0:1]


class HasegawaWakatani(PhysicsEquation):
    """
    Hasegawa-Wakatani system for resistive drift-wave turbulence, in the
    mixed form used for PINNs (vorticity is an independent output).

    Equations (original coupling, fixed background density gradient):
        ∂n/∂t + [φ, n] = C(φ - n) - κ ∂φ/∂y + D∇²n + S
        ∂ζ/∂t + [φ, ζ] = C(φ - n) + ν∇²ζ
        ζ = ∇²φ
    with the Poisson bracket [A, B] = A_x B_y - A_y B_x.

    Layout:
        coords: (N, 3) = (x, y, t), time last (same as compute_derivatives()).
        pred:   (N, 3) = (n, φ, ζ).
        Returns residual of shape (N, 3), channels:
            (density equation, vorticity equation, ζ - ∇²φ constraint).

    Parameters:
        D:     density diffusion, multiplies ∇²n.
        C:     adiabaticity (parallel electron response).
        nu:    kinematic viscosity, multiplies ∇²ζ (= ∇⁴φ).
        kappa: optional, default 1.0. Background density gradient,
               κ ≡ -∂ₓ ln n₀. Sign conventions differ between papers, so
               check yours. In Bohm units (to_bohm_units()) κ = ρ_s / L_n.
        S:     optional, default 0.0. Source in the density equation
               (float or tensor broadcastable to (N, 1)).

    Notes:
        - The coupling is the original C(φ - n). The "modified" model subtracts
          zonal (y-averaged) parts, which needs a structured grid in y and is
          not computable from scattered collocation points.
        - Only valid for pointwise models (BasePINN, ResidualPINN, ...), not
          for graph models such as SE3Transformer.
        - Needs second derivatives of the network output, so use a smooth
          activation (tanh or silu). ReLU gives zero second derivatives.
        - Cost: 9 autograd calls per residual (3 first-order, 6 second-order).
    """

    def name(self) -> str:
        return "hasegawa_wakatani"

    def expected_params(self) -> list[str]:
        return ["D", "C", "nu"]

    def description(self) -> str:
        return (
            "Hasegawa-Wakatani drift-wave turbulence (mixed form, fields n, phi, zeta):\n"
            "  ∂n/∂t + [φ, n] = C(φ - n) - κ ∂φ/∂y + D∇²n + S\n"
            "  ∂ζ/∂t + [φ, ζ] = C(φ - n) + ν∇²ζ\n"
            "  ζ = ∇²φ"
        )

    def residual(self, coords: Tensor, pred: Tensor, params: dict) -> Tensor:
        self.validate_params(params)

        if coords.shape[1] != 3:
            raise ValueError(
                f"[{self.name()}] coords must have 3 columns (x, y, t), "
                f"got {coords.shape[1]}."
            )
        if pred.shape[1] != 3:
            raise ValueError(
                f"[{self.name()}] pred must have 3 channels (n, phi, zeta), "
                f"got {pred.shape[1]}."
            )
        if not coords.requires_grad:
            raise RuntimeError(
                f"[{self.name()}] coords.requires_grad must be True, and pred "
                f"must be computed from these coords."
            )

        D, C, nu = params["D"], params["C"], params["nu"]
        kappa    = params.get("kappa", 1.0)
        S        = params.get("S", 0.0)

        n, phi, zeta = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

        # First-order derivatives, shape (N, 3): columns are (x, y, t)
        g_n    = _grad(n, coords)
        g_phi  = _grad(phi, coords)
        g_zeta = _grad(zeta, coords)

        n_t     = g_n[:, 2:3]
        zeta_t  = g_zeta[:, 2:3]
        phi_y   = g_phi[:, 1:2]

        # Second-order derivatives (max order, thanks to the mixed form)
        lap_n    = _laplacian_2d(g_n, coords)
        lap_phi  = _laplacian_2d(g_phi, coords)
        lap_zeta = _laplacian_2d(g_zeta, coords)

        coupling = C * (phi - n)

        res_n = (
            n_t + _bracket(g_phi, g_n)
            - coupling + kappa * phi_y
            - D * lap_n - S
        )
        res_zeta = (
            zeta_t + _bracket(g_phi, g_zeta)
            - coupling - nu * lap_zeta
        )
        res_constraint = zeta - lap_phi

        return torch.cat([res_n, res_zeta, res_constraint], dim=1)

# ════════════════════════════════════════════════════════════════════════════
# 4. Globalni registar — singleton na nivou modula
# ════════════════════════════════════════════════════════════════════════════

REGISTRY = PDERegistry()

REGISTRY.register(HeatEquation1D())
REGISTRY.register(DriftDiffusion1D())
REGISTRY.register(HasegawaWakatani())
# Registers the nuclear-material and radiation equations. It must stay at the
# very bottom: solvers/nuclear_equations.py imports names defined above.

import solvers.nuclear_equations