#Data transformation is a critical part of the data integration process in which raw data is converted into a unified format or structure.
# Data transformation ensures compatibility with target systems and enhances data quality and usability.
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from feature_pipeline import FeatureBatch
from physical_tensor import PhysicalTensor, Domain


@dataclass
class FidelityConfig:
    spatial_weight:  float = 0.7
    temporal_weight: float = 0.3
    normalize:       bool  = True

class FidelityAssigner:
    def __init__(self, config: Optional[FidelityConfig] = None):
        self.config = config or FidelityConfig()
    def assign(self, sources:list[PhysicalTensor]):
        if not sources:
            raise ValueError("FidelityAssigner.assign() prima nepraznu listu.")

        if len(sources) == 1:
            return [0]

        if not self.validate(sources):
            raise ValueError(
                "FidelityAssigner validacija nije prošla. "
                "Proveri da su svi izvori u kompatibilnim koordinatnim sistemima."
            )

        scores = [self.compute_score(pt) for pt in sources]

        if self.config.normalize:
            scores = self._normalize(scores)

        # Sortiraj indekse po score-u — viši score = grublja rezolucija = niži fidelity
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Assignuj fidelity nivoe: pozicija u sortiranom nizu = fidelity level
        levels = [0] * len(sources)
        for fidelity, original_idx in enumerate(sorted_indices):
            levels[original_idx] = fidelity

        return levels
    def compute_score(self,pt:PhysicalTensor):
        spatial_res = pt.spatial_resolution()
        temporal_res = pt.temporal_resolution()

        # Zaštita od inf i 0 vrednosti
        if spatial_res == float("inf") or spatial_res <= 0:
            spatial_res = 1.0
        if temporal_res <= 0:
            temporal_res = 1.0

        return (spatial_res ** self.config.spatial_weight *
                temporal_res ** self.config.temporal_weight)
    def validate(self,sources: list[PhysicalTensor]):
        if not sources:
            return False

        ref_coord_system = sources[0].coord_system
        for i, pt in enumerate(sources[1:], start=1):
            if pt.coord_system != ref_coord_system:
                print(
                    f"[FidelityAssigner] Warning: source {i} has "
                    f"coord_system={pt.coord_system.value}, expected "
                    f"{ref_coord_system.value}."
                )
                return False
        return True

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        min_s = min(scores)
        max_s = max(scores)

        if max_s == min_s:
            return [0.0] * len(scores)

        return [(s - min_s) / (max_s - min_s) for s in scores]

@dataclass
class TransformerConfig:
    target_units:        Optional[object]  = None   # UnitSystem
    boundary_threshold:  float             = 0.05
    noise_filter:        bool              = True
    fidelity_config:     FidelityConfig    = field(default_factory=FidelityConfig)
    val_ratio:           float             = 0.2



class DataTransformer(object):
    """docstring for DataTransformer"""

    def __init__(self, config: Optional[TransformerConfig] = None):
        self.config = config or TransformerConfig()
        self.assigner = FidelityAssigner(config=self.config.fidelity_config)

        # ──  API ────────────────────────────────────────────────────────────
    def transform(self, sources: list[PhysicalTensor]) -> list[FeatureBatch]:
        """
        Kompletni preprocessing: Prolaz 1 + Prolaz 2 → lista FeatureBatch.

        Args:
            sources: Lista sirovih PhysicalTensor objekata iz DataSource-a.

        Returns:
            Lista FeatureBatch objekata spremnih za model.
        """
        # ── Prolaz 1: lokalna normalizacija ─────────────────────────────────
        normalized = [self._pass1_normalize(pt) for pt in sources]

        # ── Prolaz 2: fidelity assignment ───────────────────────────────────
        fidelity_levels  = self._pass2_fidelity(normalized)
        fidelity_weights = self._compute_fidelity_weights(fidelity_levels)

        # ── Konverzija u FeatureBatch ────────────────────────────────────────
        batches = [
            self._to_feature_batch(pt, level, fidelity_weights[level])
            for pt, level in zip(normalized, fidelity_levels)
        ]

        return batches

    def _pass1_normalize(self, pt: PhysicalTensor) -> PhysicalTensor:
        """
        Lokalna normalizacija jednog PhysicalTensor-a.
        Radi nezavisno od ostalih izvora.

        Koraci:
            1. Normalizacija vrednosti na [0, 1] po svakom polju
            2. Opciono filtriranje šuma (Gaussian smoothing)
            3. Koordinate se ne menjaju — samo vrednosti

        Args:
            pt: Sirovi PhysicalTensor.

        Returns:
            Normalizovani PhysicalTensor.
        """
        values = pt.values.clone().float()

        # Normalizacija po svakom polju posebno
        for f in range(values.shape[1] if values.dim() > 1 else 1):
            col = values[:, f] if values.dim() > 1 else values
            col_min = col.min()
            col_max = col.max()
            if col_max > col_min:
                if values.dim() > 1:
                    values[:, f] = (col - col_min) / (col_max - col_min)
                else:
                    values = (col - col_min) / (col_max - col_min)

        # Opciono filtriranje šuma — Gaussian kernel po vrednostima
        if self.config.noise_filter and values.shape[0] > 10:
            values = self._gaussian_filter(values)

        return PhysicalTensor(
            values=values,
            coordinates=pt.coordinates.clone().float(),
            units=pt.units,
            coord_system=pt.coord_system,
            domain=pt.domain,
            metadata={**pt.metadata, "normalized": True},
        )

    def _gaussian_filter(self, values: Tensor, sigma: float = 1.0) -> Tensor:
        """
        Jednostavno Gaussian glačanje vrednosti.
        Koristi 1D konvoluciju sa Gaussian kernelom.

        Args:
            values: Tensor oblika (N,) ili (N, F).
            sigma:  Širina Gaussian kernela.

        Returns:
            Uglačani tensor istog oblika.
        """
        kernel_size = 5
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        if values.dim() == 1:
            v = values.unsqueeze(0).unsqueeze(0)
            k = kernel.unsqueeze(0).unsqueeze(0)
            pad = kernel_size // 2
            return torch.nn.functional.conv1d(v, k, padding=pad).squeeze()

        result = values.clone()
        for f in range(values.shape[1]):
            col = values[:, f].unsqueeze(0).unsqueeze(0)
            k = kernel.unsqueeze(0).unsqueeze(0)
            pad = kernel_size // 2
            result[:, f] = torch.nn.functional.conv1d(
                col, k, padding=pad
            ).squeeze()
        return result

        # ── Prolaz 2 ─────────────────────────────────────────────────────────────

    def _pass2_fidelity(self, sources: list[PhysicalTensor]) -> list[int]:
        """
        Relacioni prolaz — assignuje fidelity nivoe svim izvorima zajedno.

        Args:
            sources: Lista normalizovanih PhysicalTensor objekata.

        Returns:
            Lista fidelity nivoa, iste dužine kao sources.
        """
        return self.assigner.assign(sources)

    @staticmethod
    def _compute_fidelity_weights(levels: list[int]) -> dict[int, float]:
        """
        Računa fidelity_weight za svaki nivo.
        Viši fidelity nivo dobija veću težinu u loss-u.

        Formula: weight[i] = (i + 1) / sum(1..N)
        Normalizovano na sumu 1.0.

        Args:
            levels: Lista fidelity nivoa.

        Returns:
            Dict koji mapira fidelity_level → fidelity_weight.
        """
        n_levels = max(levels) + 1
        raw = {i: float(i + 1) for i in range(n_levels)}
        total = sum(raw.values())
        return {i: w / total for i, w in raw.items()}

    # ── Konverzija u FeatureBatch ─────────────────────────────────────────────

    def _to_feature_batch(
            self,
            pt: PhysicalTensor,
            fidelity_level: int,
            fidelity_weight: float,
    ) -> FeatureBatch:
        """
        Konvertuje PhysicalTensor u FeatureBatch.
        Generiše boundary_mask i collocation_mask iz domene.

        Args:
            pt:              Normalizovani PhysicalTensor.
            fidelity_level:  Assignovani fidelity nivo.
            fidelity_weight: Težina ovog nivoa u loss-u.

        Returns:
            FeatureBatch spreman za model.
        """
        boundary_mask, collocation_mask = self._compute_masks(
            pt.coordinates, pt.domain
        )

        return FeatureBatch(
            coords=pt.coordinates,
            fields=pt.values,
            boundary_mask=boundary_mask,
            collocation_mask=collocation_mask,
            physics_params=pt.metadata.get("physics_params", {}),
            fidelity_level=fidelity_level,
            fidelity_weight=fidelity_weight,
        )

    def _compute_masks(
            self,
            coords: Tensor,
            domain: Domain,
    ) -> tuple[Tensor, Tensor]:
        """
        Generiše boundary_mask i collocation_mask.

        Tačka je boundary ako je unutar boundary_threshold
        od bilo koje granice domene.

        Args:
            coords: Koordinate tačaka, oblik (N, D).
            domain: Domena iz PhysicalTensor-a.

        Returns:
            tuple: (boundary_mask, collocation_mask), oba oblika (N,) bool.
        """
        n = coords.shape[0]
        boundary = torch.zeros(n, dtype=torch.bool)
        thr = self.config.boundary_threshold

        dim = 0
        for range_ in [domain.x_range, domain.y_range, domain.z_range]:
            if range_ is not None and dim < coords.shape[1] - 1:
                lo, hi = range_
                span = hi - lo
                col = coords[:, dim]
                boundary |= (col < lo + thr * span)
                boundary |= (col > hi - thr * span)
                dim += 1

        # Vremenska koordinata — poslednja dimenzija
        t_lo, t_hi = domain.t_range
        t_span = t_hi - t_lo
        t_col = coords[:, -1]
        boundary |= (t_col < t_lo + thr * t_span)
        boundary |= (t_col > t_hi - thr * t_span)

        collocation = ~boundary
        return boundary, collocation