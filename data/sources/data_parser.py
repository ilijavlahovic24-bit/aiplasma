#Abstract interface for sources of physical data in AIPlasma Framework
#CSV files for
#HDF5 from simulations, ROOT files from experiments, netCDF
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator
import torch
from torch import Tensor

from physical_tensor import (
    PhysicalTensor, CoordinateSystem, UnitSystem, Domain
)

@dataclass
class DataSourceConfig:
    """
    Forwarding to the DataSource constructor.
    """
    coord_system: CoordinateSystem = CoordinateSystem.CARTESIAN
    units:        Optional[UnitSystem] = None
    domain:       Optional[Domain]     = None
    max_points:   Optional[int]        = None   # None = load all
    shuffle:      bool                 = False
    metadata:     dict                 = field(default_factory=dict)



class DataSource(ABC):
    """
        Abstract interface for all data sources in AIPlasma.

        Each data source inherits from DataSource and implements:
        - load() → loads all data at once
        - stream() → loads data in chunks (for large datasets)
        - validate() → checks data integrity

        The user does not implement the pipeline logic — only the source description.
        FidelityAssigner in the Preprocessing layer assigns fidelity_level
        only when all sources have been loaded and compared with each other.
        """

    def __init__(self, config: DataSourceConfig):
        self.config = config

    # ── User must implement ───────────────────────────────────────

    @abstractmethod
    def load(self) -> PhysicalTensor:
        """
        Load all data and returns PhysicalTensor.
        It is used for dataset that can fit in memory
        """
        ...

    @abstractmethod
    def validate(self) -> bool:
        """
        Checks if the data source is valid before loading.
        Returns True if everything is fine, False if there are problems.
        """
        ...

    # ── Framework pruža ─────────────────────────────────────────────────────

    def stream(self, chunk_size: int = 1000) -> Iterator[PhysicalTensor]:
        """
        Loads data in chunks.
        Default implementation: loads everything and cuts it into chunks.
        Override for really lazy loading (HDF5, databases).
        """
        full = self.load()
        n = full.n_points()

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            yield PhysicalTensor(
                values=full.values[start:end],
                coordinates=full.coordinates[start:end],
                units=full.units,
                coord_system=full.coord_system,
                domain=full.domain,
                metadata={**full.metadata, "chunk": f"{start}:{end}"},
            )

    def describe(self) -> str:
        """Short description of source for logging."""
        return f"{self.__class__.__name__}(coord_system={self.config.coord_system.value})"



