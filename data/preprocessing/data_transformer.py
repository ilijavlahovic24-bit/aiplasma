#Data transformation is a critical part of the data integration process in which raw data is converted into a unified format or structure.
# Data transformation ensures compatibility with target systems and enhances data quality and usability.
from dataclasses import dataclass
from typing import Optional

from physical_tensor import PhysicalTensor

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


class DataTransformer(object):
    """docstring for DataTransformer"""