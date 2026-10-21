#Abstract interface for sources of physical data in AIPlasma Framework
#CSV files for
#HDF5 from simulations, ROOT files from experiments, netCDF
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator
import torch
from torch import Tensor

from data.sources.physical_tensor import (
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

        The user does not implement the pipeline logic - only the source description.
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
    HDF5 source — typical format for simulation output (GENE, GS2, CGYRO).

    Expected default layout:
        /coordinates — space-time grid, shape (N, D)
        /fields      — physical fields, shape (N, F)  (or (N,) for one field)

    Dataset names are configurable, nested paths are supported:
        HDF5DataSource(path, config, coordinates_key="grid/xt", fields_key="phi/data")

    For layouts that cannot be expressed with two dataset names, subclass
    and override BOTH hooks: _n_rows() and _parse_hdf5().

    max_points selects a random subset, consistent with the other sources.
    Rows are read from disk in increasing order (an h5py requirement),
    so the subset is random but not shuffled.

    stream() reads chunks directly from disk, without loading the whole file.
    """

    def __init__(
        self,
        path: str,
        config: DataSourceConfig,
        coordinates_key: str = "coordinates",
        fields_key: str = "fields",
    ):
        super().__init__(config)
        self.path = Path(path)
        self.coordinates_key = coordinates_key
        self.fields_key = fields_key

    # ── DataSource interface ─────────────────────────────────────────────────

    def validate(self) -> bool:
        if not self.path.exists():
            print(f"[HDF5DataSource] ERROR: file not found: {self.path}")
            return False
        try:
            import h5py
        except ImportError:
            print("[HDF5DataSource] ERROR: h5py is not installed. pip install h5py")
            return False
        try:
            with h5py.File(self.path, "r") as f:
                if not list(f.keys()):
                    print("[HDF5DataSource] ERROR: HDF5 file is empty.")
                    return False
                # Fails with KeyError if the configured datasets do not exist
                self._n_rows(f)
            return True
        except Exception as e:
            print(f"[HDF5DataSource] ERROR: could not read layout of {self.path}: {e}")
            return False

    def load(self) -> PhysicalTensor:
        if not self.validate():
            raise ValueError(f"HDF5DataSource validation failed for: {self.path}")

        import h5py
        with h5py.File(self.path, "r") as f:
            rows = self._select_rows(self._n_rows(f))
            coords, values = self._normalize(*self._parse_hdf5(f, rows))
            meta = self._read_attrs(f)

        return self._build(coords, values, meta)

    def stream(self, chunk_size: int = 1000) -> Iterator[PhysicalTensor]:
        """Lazy streaming: each chunk is read from disk on demand."""
        if not self.validate():
            raise ValueError(f"HDF5DataSource validation failed for: {self.path}")

        import h5py
        with h5py.File(self.path, "r") as f:
            n        = self._n_rows(f)
            selected = self._select_rows(n)
            meta     = self._read_attrs(f)

            # selected is slice(None) (everything) or a sorted index array (max_points)
            full  = isinstance(selected, slice)
            total = n if full else len(selected)

            for start in range(0, total, chunk_size):
                end  = min(start + chunk_size, total)
                rows = slice(start, end) if full else selected[start:end]
                coords, values = self._normalize(*self._parse_hdf5(f, rows))
                yield self._build(coords, values, {**meta, "chunk": f"{start}:{end}"})

    # ── Hooks for custom layouts ─────────────────────────────────────────────

    def _n_rows(self, f) -> int:
        """
        Total number of rows in the file.
        Override together with _parse_hdf5() for non-standard layouts.
        """
        return f[self.coordinates_key].shape[0]

    def _parse_hdf5(self, f, rows) -> tuple:
        """
        Reads the given rows from an open h5py file.

        Args:
            f:    Open h5py.File.
            rows: slice, or sorted numpy integer array (h5py requires
                  increasing indices).

        Returns:
            (coordinates: Tensor (n, D), values: Tensor (n, F))
        """
        return (
            torch.tensor(f[self.coordinates_key][rows], dtype=torch.float32),
            torch.tensor(f[self.fields_key][rows], dtype=torch.float32),
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _select_rows(self, n: int):
        """slice(None) if everything is read, else a sorted random index array."""
        limit = self.config.max_points
        if limit is None or limit >= n:
            return slice(None)
        return torch.randperm(n)[:limit].sort().values.numpy()

    @staticmethod
    def _normalize(coords: Tensor, values: Tensor) -> tuple:
        """Validates shapes; a 1-D fields dataset becomes (N, 1)."""
        if values.ndim == 1:
            values = values.unsqueeze(1)
        if coords.ndim != 2 or values.ndim != 2:
            raise ValueError(
                f"Expected coordinates (N, D) and fields (N, F), "
                f"got {tuple(coords.shape)} and {tuple(values.shape)}"
            )
        if coords.shape[0] != values.shape[0]:
            raise ValueError(
                f"Row count mismatch: {coords.shape[0]} coordinate rows "
                f"vs {values.shape[0]} field rows"
            )
        return coords, values

    def _read_attrs(self, f) -> dict:
        """Metadata from HDF5 attributes, if present."""
        meta = dict(self.config.metadata)
        if "experiment_id" in f.attrs:
            meta["experiment_id"] = str(f.attrs["experiment_id"])
        if "instrument" in f.attrs:
            meta["instrument"] = str(f.attrs["instrument"])
        return meta

    def _build(self, coords: Tensor, values: Tensor, meta: dict) -> PhysicalTensor:
        return PhysicalTensor(
            values=values,
            coordinates=coords,
            units=self.config.units or UnitSystem(
                spatial="m", temporal="s", field="normalized"
            ),
            coord_system=self.config.coord_system,
            domain=self.config.domain or Domain(t_range=(0.0, 1.0)),
            metadata={
                **meta,
                "source_type": "hdf5",
                "path": str(self.path),
                "coordinates_key": self.coordinates_key,
                "fields_key": self.fields_key,
            },
        )
