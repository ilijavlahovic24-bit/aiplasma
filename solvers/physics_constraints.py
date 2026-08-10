"""
physics_constraints.py
PDE registar i konkretne fizičke jednačine za AIPlasma framework.


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
        """
        Residual: ∂u/∂t - α · ∂²u/∂x² = 0

        Args:
            coords: (N, 2) — [x, t]
            pred:   (N, 1) — predikcija u
            params: {"alpha": float}

        Returns:
            Residual tensor oblika (N, 1).
        """
        self.validate_params(params)
        alpha = params["alpha"]

        # coords mora imati requires_grad=True za autograd
        if not coords.requires_grad:
            coords = coords.requires_grad_(True)

        u = pred

        # ∂u/∂t
        du_dt = torch.autograd.grad(
            u, coords,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]  # vremenska koordinata je indeks 1

        # ∂u/∂x
        du_dx = torch.autograd.grad(
            u, coords,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]  # prostorna koordinata je indeks 0

        # ∂²u/∂x²
        d2u_dx2 = torch.autograd.grad(
            du_dx, coords,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

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
        """
        Residual: ∂u/∂t + v·∂u/∂x - D·∂²u/∂x² = 0

        Args:
            coords: (N, 2) — [x, t]
            pred:   (N, 1) — predikcija u
            params: {"D": float, "v": float}

        Returns:
            Residual tensor oblika (N, 1).
        """
        self.validate_params(params)
        D = params["D"]
        v = params["v"]

        if not coords.requires_grad:
            coords = coords.requires_grad_(True)

        u = pred

        # ∂u/∂t
        du_dt = torch.autograd.grad(
            u, coords,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        # ∂u/∂x
        du_dx = torch.autograd.grad(
            u, coords,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        # ∂²u/∂x²
        d2u_dx2 = torch.autograd.grad(
            du_dx, coords,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        return du_dt + v * du_dx - D * d2u_dx2


class HasegawaWakatani(PhysicsEquation):
    """
    Hasegawa-Wakatani sistem za turbulenciju plazme.

    Sistem od dve spregnute PDE:
        ∂n/∂t + [φ, n] = D·∇²n + S
        ∂∇²φ/∂t + [φ, ∇²φ] = C(φ - n) + ν·∇⁴φ

    gde je:
        n  — normalizovana gustina plazme
        φ  — elektrostatički potencijal
        D  — koeficijent difuzije
        C  — koeficijent adiabatskog sparivanja
        nu — koeficijent viskoznosti

    Parametri:
        D:  Koeficijent difuzije
        C:  Koeficijent adiabatskog sparivanja
        nu: Koeficijent viskoznosti

    Status: Stub — implementacija dolazi u Fazi 3 (PlasmaPhysicsProblem).
    Poisson bracket i ∇⁴ operator zahtevaju plasma-specifične
    utility metode iz PlasmaPhysicsProblem.
    """

    def name(self) -> str:
        return "hasegawa_wakatani"

    def expected_params(self) -> list[str]:
        return ["D", "C", "nu"]

    def description(self) -> str:
        return (
            "Hasegawa-Wakatani plasma turbulence system:\n"
            "  ∂n/∂t + [φ, n] = D·∇²n + S\n"
            "  ∂∇²φ/∂t + [φ, ∇²φ] = C(φ - n) + ν·∇⁴φ"
        )

    def residual(self, coords: Tensor, pred: Tensor, params: dict) -> Tensor:
        raise NotImplementedError(
            "HasegawaWakatani.residual() dolazi u Fazi 3.\n"
            "Implementacija zahteva PlasmaPhysicsProblem.poisson_bracket() "
            "i plasma-specifične operatore."
        )


# ════════════════════════════════════════════════════════════════════════════
# 4. Globalni registar — singleton na nivou modula
# ════════════════════════════════════════════════════════════════════════════

REGISTRY = PDERegistry()

REGISTRY.register(HeatEquation1D())
REGISTRY.register(DriftDiffusion1D())
REGISTRY.register(HasegawaWakatani())