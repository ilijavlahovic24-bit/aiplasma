#Interni type that contains physical properties od data to pipeline
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import torch
from torch import Tensor


class CoordinateSystem(Enum):
    CARTESIAN = "cartesian"  # (x, y, z, t)       - general purpose
    CYLINDRICAL = "cylindrical"  # (r, z, φ, t)        - axis symetry
    TOROIDAL = "toroidal"  # (r, θ, φ, t)        - tokamak geometry
    CUSTOM = "custom"  # user defines


@dataclass
class UnitSystem:
    """
    Units of Measurement in PhysicalTensor-u.

    Example:
        SI:         spatial="m",    temporal="s",          field="Pa"
        Plasma:     spatial="m",    temporal="s",          field="keV"
        Normalized: spatial="bohm", temporal="normalized", field="normalized"
    """
    spatial: str  # unit of spatial coordinates: "m", "cm", "bohm"
    temporal: str  # unit of time coordinates:  "s", "ms", "normalized"
    field: str  # unit of physical field:        "keV", "Pa", "T", "normalized"

    def is_normalized(self) -> bool:
        return all(v == "normalized" for v in [self.spatial, self.temporal, self.field])

    def __str__(self) -> str:
        return f"UnitSystem(spatial={self.spatial}, temporal={self.temporal}, field={self.field})"


@dataclass
class Domain:
    """
    Space-time domain of data.
    x_range, y_range, z_range are optional and
    coordinates don't have to exist (for example 1D problem only has x_range and t_range).
    """
    t_range: tuple[float, float]
    x_range: Optional[tuple[float, float]] = None
    y_range: Optional[tuple[float, float]] = None
    z_range: Optional[tuple[float, float]] = None

    def spatial_dims(self) -> int:
        """Numer of dimensions in spatial space."""
        return sum(r is not None for r in [self.x_range, self.y_range, self.z_range])

    def contains(self, coords: Tensor) -> Tensor:
        """
        Checks if coordinates are within the domain.
        returns boolean mask as (N,).
        """
        mask = torch.ones(coords.shape[0], dtype=torch.bool)
        dim = 0

        for range_ in [self.x_range, self.y_range, self.z_range]:
            if range_ is not None:
                mask &= (coords[:, dim] >= range_[0]) & (coords[:, dim] <= range_[1])
                dim += 1

        # Time coordinate is always last
        mask &= (coords[:, -1] >= self.t_range[0]) & (coords[:, -1] <= self.t_range[1])
        return mask

    def __str__(self) -> str:
        ranges = []
        for name, r in [("x", self.x_range), ("y", self.y_range), ("z", self.z_range)]:
            if r is not None:
                ranges.append(f"{name}∈{r}")
        ranges.append(f"t∈{self.t_range}")
        return f"Domain({', '.join(ranges)})"



@dataclass
class PhysicalTensor:
    """
    An internal type that carries physical data through the AIPlasma pipeline.

    Each transition between pipeline layers is a transformation of this type.
    fidelity_level is not assigned here
    it is assigned only in Preprocessing phase (FidelityAssigner) because
    it is a relational property, not an absolute one.

    Attributes:
    values - physical fields, form (N, F) where F = field number
    coordinates - space-time input, form (N, D) where D = number dim.
    units - physical units of values and coordinates
    coord_system- coordinate system geometry
    domain - space-time range of data
    metadata - source, instrument, experiment ID, and more
    """
    values: Tensor
    coordinates: Tensor
    units: UnitSystem
    coord_system: CoordinateSystem
    domain: Domain
    metadata: dict = field(default_factory=dict)

    # ── Validation ───────────────────────────────────────────────────────────

    def __post_init__(self):
        if self.values.shape[0] != self.coordinates.shape[0]:
            raise ValueError(
                f"values i coordinates moraju imati isti broj tačaka. "
                f"Dobijeno: values={self.values.shape[0]}, "
                f"coordinates={self.coordinates.shape[0]}"
            )

    # ── Utility methods ───────────────────────────────────────────────────────

    def n_points(self) -> int:
        return self.values.shape[0]

    def n_fields(self) -> int:
        return self.values.shape[1] if self.values.dim() > 1 else 1

    def spatial_resolution(self) -> float:
        if self.domain.x_range is None:
            return float("inf")
        x_span = self.domain.x_range[1] - self.domain.x_range[0]
        return x_span / max(self.n_points(), 1)

    def temporal_resolution(self) -> float:
        t_span = self.domain.t_range[1] - self.domain.t_range[0]
        return t_span / max(self.n_points(), 1)

    def to(self, device: str) -> "PhysicalTensor":
        """Premešta tenzore na zadati device."""
        return PhysicalTensor(
            values=self.values.to(device),
            coordinates=self.coordinates.to(device),
            units=self.units,
            coord_system=self.coord_system,
            domain=self.domain,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        return (
            f"PhysicalTensor(\n"
            f"  points={self.n_points()}, fields={self.n_fields()},\n"
            f"  coord_system={self.coord_system.value},\n"
            f"  {self.units},\n"
            f"  {self.domain},\n"
            f"  metadata_keys={list(self.metadata.keys())}\n"
            f")"
        )