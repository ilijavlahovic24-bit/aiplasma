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



#Concrete implementations

class SyntheticDataSource(DataSource):
    """
   Generates synthetic data from a given analytical solution.
    Useful for testing and benchmarking.

    Primer:
        source = SyntheticDataSource(
            solution_fn=lambda x, t: torch.sin(math.pi * x) * torch.exp(-0.01 * math.pi**2 * t),
            config=DataSourceConfig(
                coord_system=CoordinateSystem.CARTESIAN,
                units=UnitSystem(spatial="m", temporal="s", field="normalized"),
                domain=Domain(x_range=(0.0, 1.0), t_range=(0.0, 1.0)),
            ),
            n_points=500,
        )
        tensor = source.load()
    """

    def __init__(
            self,
            solution_fn,  # callable: (x: Tensor, t: Tensor) -> Tensor
            config: DataSourceConfig,
            n_points: int = 500,
            noise_std: float = 0.0,
    ):
        super().__init__(config)
        self.solution_fn = solution_fn
        self.n_points = n_points
        self.noise_std = noise_std

    def validate(self) -> bool:
        if self.config.domain is None:
            print("[SyntheticDataSource] ERROR: domain not defined in config.")
            return False
        if self.config.units is None:
            print("[SyntheticDataSource] ERROR: units not defined in config.")
            return False
        return True

    def load(self) -> PhysicalTensor:
        if not self.validate():
            raise ValueError("DataSource validacija nije prošla. Proveri config.")

        domain = self.config.domain

        # ── Generating coordinates───────────────────────────────────────────
        x = torch.rand(self.n_points)
        if domain.x_range:
            x = x * (domain.x_range[1] - domain.x_range[0]) + domain.x_range[0]

        t = torch.rand(self.n_points)
        t = t * (domain.t_range[1] - domain.t_range[0]) + domain.t_range[0]

        coordinates = torch.stack([x, t], dim=1)  # (N, 2)

        # ── Analitical solution as value ────────────────────────────────
        values = self.solution_fn(x, t).unsqueeze(1)  # (N, 1)

        # ── Optional Noise─────────────────────────────────────────────────────
        if self.noise_std > 0.0:
            values = values + torch.randn_like(values) * self.noise_std

        # ── Number of points ─────────────────────────────────────────
        if self.config.max_points is not None:
            idx = torch.randperm(self.n_points)[:self.config.max_points]
            values = values[idx]
            coordinates = coordinates[idx]

        return PhysicalTensor(
            values=values,
            coordinates=coordinates,
            units=self.config.units,
            coord_system=self.config.coord_system,
            domain=domain,
            metadata={
                **self.config.metadata,
                "source_type": "synthetic",
                "n_points": self.n_points,
                "noise_std": self.noise_std,
            },
        )


class CSVDataSource(DataSource):
    """
    Load data from CSV file.

    Expected format :
        x, t, u         ←  1D problem
        x, y, t, u, v  ←  2D problem

    EXample:
        source = CSVDataSource(
            path="data/plasma_measurements.csv",
            field_columns=["density", "temperature"],
            config=DataSourceConfig(
                coord_system=CoordinateSystem.TOROIDAL,
                units=UnitSystem(spatial="m", temporal="s", field="keV"),
                domain=Domain(x_range=(0.0, 1.0), t_range=(0.0, 10.0)),
            )
        )
    """

    def __init__(
            self,
            path: str,
            field_columns: list[str],
            config: DataSourceConfig,
    ):
        super().__init__(config)
        self.path = Path(path)
        self.field_columns = field_columns

    def validate(self) -> bool:
        if not self.path.exists():
            print(f"[CSVDataSource] ERROR: file not found: {self.path}")
            return False
        if self.config.units is None:
            print("[CSVDataSource] ERRROR: units not defined in config-u.")
            return False
        return True

    def load(self) -> PhysicalTensor:
        if not self.validate():
            raise ValueError(f"DataSource validaiton failed: {self.path}")

        import csv
        rows = []
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: float(v) for k, v in row.items()})

        # ── Koordinate i vrednosti iz kolona ────────────────────────────────
        coord_cols = [c for c in rows[0].keys() if c not in self.field_columns]
        coordinates = torch.tensor([[r[c] for c in coord_cols] for r in rows])
        values = torch.tensor([[r[c] for c in self.field_columns] for r in rows])

        if self.config.max_points is not None:
            idx = torch.randperm(len(rows))[:self.config.max_points]
            coordinates = coordinates[idx]
            values = values[idx]

        return PhysicalTensor(
            values=values,
            coordinates=coordinates,
            units=self.config.units,
            coord_system=self.config.coord_system,
            domain=self.config.domain or Domain(t_range=(0.0, 1.0)),
            metadata={
                **self.config.metadata,
                "source_type": "csv",
                "path": str(self.path),
                "field_columns": self.field_columns,
            },
        )


class HDF5DataSource(DataSource):
    """
    Stub for HDF5 izvor — typical format for simulations(GENE, GS2, CGYRO).

    """

    def __init__(self, path: str, config: DataSourceConfig):
        super().__init__(config)
        self.path = Path(path)

    def validate(self) -> bool:
        if not self.path.exists():
            print(f"[HDF5DataSource] GREŠKA: fajl nije pronađen: {self.path}")
            return False
        try:
            import h5py
            with h5py.File(self.path, "r") as f:
                keys = list(f.keys())
                if not keys:
                    print("[HDF5DataSource] GREŠKA: HDF5 fajl je prazan.")
                    return False
            return True
        except ImportError:
            print("[HDF5DataSource] GREŠKA: h5py nije instaliran. pip install h5py")
            return False
        except Exception as e:
            print(f"[HDF5DataSource] GREŠKA pri otvaranju fajla: {e}")
            return False

    def load(self) -> PhysicalTensor:
        if not self.validate():
            raise ValueError(f"HDF5DataSource validacija nije prošla za: {self.path}")

        import h5py
        with h5py.File(self.path, "r") as f:
            # Pretpostavka o strukturi HDF5 fajla iz GENE/GS2/CGYRO:
            #   /coordinates — prostorno-vremenski grid, shape (N, D)
            #   /fields      — fizička polja, shape (N, F)
            # Ako fajl ima drugačiju strukturu, korisnik override-uje _parse_hdf5()
            coords = torch.tensor(f["coordinates"][:], dtype=torch.float32)
            values = torch.tensor(f["fields"][:], dtype=torch.float32)

            # Metadata iz HDF5 atributa ako postoje
            meta = dict(self.config.metadata)
            if "experiment_id" in f.attrs:
                meta["experiment_id"] = str(f.attrs["experiment_id"])
            if "instrument" in f.attrs:
                meta["instrument"] = str(f.attrs["instrument"])

        if self.config.max_points is not None:
            idx = torch.randperm(coords.shape[0])[:self.config.max_points]
            coords = coords[idx]
            values = values[idx]

        return PhysicalTensor(
            values=values,
            coordinates=coords,
            units=self.config.units or UnitSystem(
                spatial="m", temporal="s", field="normalized"
            ),
            coord_system=self.config.coord_system,
            domain=self.config.domain or Domain(t_range=(0.0, 1.0)),
            metadata={**meta, "source_type": "hdf5", "path": str(self.path)},
        )

    def _parse_hdf5(self, f) -> tuple:
        """
        Hook za custom parsiranje HDF5 strukture.
        Override kada Vinča podaci imaju drugačiji layout od pretpostavljenog.

        Returns:
            tuple: (coordinates: Tensor, values: Tensor)
        """
        return (
            torch.tensor(f["coordinates"][:], dtype=torch.float32),
            torch.tensor(f["fields"][:], dtype=torch.float32),
        )
