#Test 5

"""
test_data_pipeline_hdf5.py
DataPipeline test using HDF5DataSource.

Generates temporary HDF5 files, runs the pipeline, then cleans up.
Skipped entirely if h5py is not installed (it is an optional dependency).

Default layout (GENE/GS2/CGYRO-style output):
    /coordinates — shape (N, D)
    /fields      — shape (N, F), or (N,) for a single field

Run:
    python tests/test_data_pipeline_hdf5.py
"""

import sys
import os
import time
import tempfile
from contextlib import contextmanager

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import h5py
except ImportError:
    print("h5py is not installed — skipping HDF5 tests (pip install h5py)")
    sys.exit(0)

from data_parser import DataSourceConfig, HDF5DataSource
from data_pipeline import DataPipeline

from data.sources.physical_tensor import (
    PhysicalTensor, UnitSystem, Domain, CoordinateSystem
)

from data.preprocessing.data_transformer import DataTransformer, FeatureBatchDataset

# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n[{title}]")


@contextmanager
def temp_h5():
    """Yields a path to a temporary .h5 file and removes it afterwards."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


def make_h5(
    path: str,
    n_rows: int = 150,
    alpha: float = 0.01,
    coordinates_key: str = "coordinates",
    fields_key: str = "fields",
    flat_fields: bool = False,
) -> None:
    """
    Writes an HDF5 file with the heat equation solution:
        coordinates: (x, t)
        fields:      u = sin(pi*x) * exp(-alpha*pi^2*t)
    """
    x = np.random.rand(n_rows)
    t = np.random.rand(n_rows)
    u = np.sin(np.pi * x) * np.exp(-alpha * np.pi ** 2 * t)

    with h5py.File(path, "w") as f:
        f[coordinates_key] = np.stack([x, t], axis=1)
        f[fields_key]      = u if flat_fields else u[:, None]
        f.attrs["experiment_id"] = "GENE-042"


def make_config(**kwargs) -> DataSourceConfig:
    return DataSourceConfig(
        coord_system=CoordinateSystem.CARTESIAN,
        units=UnitSystem(spatial="m", temporal="s", field="normalized"),
        domain=Domain(x_range=(0.0, 1.0), t_range=(0.0, 1.0)),
        **kwargs,
    )


class SplitLayoutSource(HDF5DataSource):
    """
    Custom layout: x, t and u stored as three separate 1-D datasets.
    Overrides BOTH hooks, as documented on HDF5DataSource.
    """

    def _n_rows(self, f) -> int:
        return f["x"].shape[0]

    def _parse_hdf5(self, f, rows) -> tuple:
        coords = np.stack([f["x"][rows], f["t"][rows]], axis=1)
        return (
            torch.tensor(coords, dtype=torch.float32),
            torch.tensor(f["u"][rows], dtype=torch.float32),
        )


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

def test_hdf5_source_basic():
    section("HDF5DataSource — basic load")

    with temp_h5() as path:
        make_h5(path, n_rows=100)

        source = HDF5DataSource(path=path, config=make_config())

        check("validate() returns True", source.validate())

        tensor = source.load()
        check("load() returns PhysicalTensor", isinstance(tensor, PhysicalTensor))
        check("correct n_points", tensor.n_points() == 100)
        check("correct n_fields", tensor.n_fields() == 1)
        check("coordinates shape (N, 2)", tensor.coordinates.shape == (100, 2))
        check("values shape (N, 1)", tensor.values.shape == (100, 1))
        check("values are float32", tensor.values.dtype == torch.float32)
        check("metadata has source_type hdf5",
              tensor.metadata.get("source_type") == "hdf5")
        check("metadata has path", "path" in tensor.metadata)
        check("metadata has dataset keys",
              tensor.metadata.get("coordinates_key") == "coordinates"
              and tensor.metadata.get("fields_key") == "fields")
        check("HDF5 attributes copied to metadata",
              tensor.metadata.get("experiment_id") == "GENE-042")


def test_hdf5_source_missing_file():
    section("HDF5DataSource — missing / invalid file")

    source = HDF5DataSource(path="/nonexistent/path/data.h5", config=make_config())
    check("validate() False for missing file", not source.validate())

    try:
        source.load()
        check("load() raises ValueError for missing file", False)
    except ValueError:
        check("load() raises ValueError for missing file", True)

    # File exists but contains no datasets
    with temp_h5() as path:
        with h5py.File(path, "w"):
            pass
        check("validate() False for empty file",
              not HDF5DataSource(path=path, config=make_config()).validate())


def test_hdf5_source_max_points():
    section("HDF5DataSource — max_points")

    with temp_h5() as path:
        make_h5(path, n_rows=200)

        tensor = HDF5DataSource(
            path=path, config=make_config(max_points=75)
        ).load()
        check("max_points respected", tensor.n_points() == 75)

        # Coordinates and fields must be subset with the same rows
        x, t     = tensor.coordinates[:, 0], tensor.coordinates[:, 1]
        expected = torch.sin(torch.pi * x) * torch.exp(-0.01 * torch.pi ** 2 * t)
        check("rows stay aligned after subset",
              torch.allclose(tensor.values[:, 0], expected, atol=1e-5))

        # max_points larger than the file → everything is loaded
        full = HDF5DataSource(
            path=path, config=make_config(max_points=1000)
        ).load()
        check("max_points > N loads all rows", full.n_points() == 200)


def test_hdf5_custom_keys():
    section("HDF5DataSource — custom dataset keys")

    with temp_h5() as path:
        make_h5(path, n_rows=100, coordinates_key="grid/xt",
                fields_key="phi/data", flat_fields=True)

        source = HDF5DataSource(
            path=path, config=make_config(),
            coordinates_key="grid/xt", fields_key="phi/data",
        )
        check("validate() True with nested keys", source.validate())

        tensor = source.load()
        check("1-D fields become (N, 1)", tensor.values.shape == (100, 1))
        check("metadata records custom keys",
              tensor.metadata["coordinates_key"] == "grid/xt"
              and tensor.metadata["fields_key"] == "phi/data")

        # Default keys do not exist in this file
        check("validate() False for wrong keys",
              not HDF5DataSource(path=path, config=make_config()).validate())


def test_hdf5_custom_layout_hooks():
    section("HDF5DataSource — custom layout via hooks")

    with temp_h5() as path:
        x = np.random.rand(120)
        t = np.random.rand(120)
        with h5py.File(path, "w") as f:
            f["x"] = x
            f["t"] = t
            f["u"] = np.sin(np.pi * x) * np.exp(-0.01 * np.pi ** 2 * t)

        check("subclass validate() True",
              SplitLayoutSource(path=path, config=make_config()).validate())

        tensor = SplitLayoutSource(path=path, config=make_config()).load()
        check("subclass load n_points", tensor.n_points() == 120)
        check("subclass values shape (N, 1)", tensor.values.shape == (120, 1))

        subset = SplitLayoutSource(
            path=path, config=make_config(max_points=40)
        ).load()
        check("subclass max_points respected", subset.n_points() == 40)


def test_hdf5_multifidelity():
    section("HDF5DataSource — multi-fidelity pipeline")

    paths = []
    try:
        # Create 3 HDF5 files with different resolutions
        for n in [50, 150, 300]:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                paths.append(f.name)
            make_h5(paths[-1], n_rows=n)

        config  = make_config()
        sources = [HDF5DataSource(path=p, config=config) for p in paths]

        pipeline         = DataPipeline(sources=sources)
        train_ds, val_ds = pipeline.build()

        check("pipeline builds successfully",
              isinstance(train_ds, FeatureBatchDataset))
        check("train dataset not empty", len(train_ds) > 0)

        all_batches     = [train_ds[i] for i in range(len(train_ds))] + \
                          [val_ds[i]   for i in range(len(val_ds))]
        fidelity_levels = [b.fidelity_level for b in all_batches]
        check("3 HDF5 sources → 3 fidelity levels", len(set(fidelity_levels)) == 3)

        weights = [b.fidelity_weight for b in all_batches]
        check("fidelity weights sum to 1.0", abs(sum(weights) - 1.0) < 1e-5)

    finally:
        for p in paths:
            if os.path.exists(p):
                os.unlink(p)


def test_hdf5_transformer():
    section("HDF5DataSource — DataTransformer")

    with temp_h5() as path:
        make_h5(path, n_rows=120)

        tensor      = HDF5DataSource(path=path, config=make_config()).load()
        transformer = DataTransformer()
        batches     = transformer.transform([tensor])

        check("transformer produces FeatureBatch", len(batches) == 1)
        b = batches[0]
        check("boundary_mask and collocation_mask disjoint",
              not (b.boundary_mask & b.collocation_mask).any())
        check("coords shape correct", b.coords.shape[1] == 2)
        check("fields shape correct", b.fields.shape[1] == 1)


def test_hdf5_stream():
    section("HDF5DataSource — streaming")

    with temp_h5() as path:
        make_h5(path, n_rows=90)

        source = HDF5DataSource(path=path, config=make_config())
        chunks = list(source.stream(chunk_size=30))

        check("stream produces 3 chunks", len(chunks) == 3)
        check("stream total = n_rows", sum(c.n_points() for c in chunks) == 90)
        check("all chunks are PhysicalTensor",
              all(isinstance(c, PhysicalTensor) for c in chunks))
        check("chunk metadata marks the row range",
              chunks[0].metadata.get("chunk") == "0:30")
        check("stream equals load",
              torch.allclose(torch.cat([c.values for c in chunks]),
                             source.load().values))

        # Streaming together with max_points
        capped = list(HDF5DataSource(
            path=path, config=make_config(max_points=70)
        ).stream(chunk_size=30))
        check("stream + max_points chunk sizes",
              [c.n_points() for c in capped] == [30, 30, 10])


def test_hdf5_errors():
    section("HDF5DataSource — malformed data")

    # coordinates and fields with different row counts
    with temp_h5() as path:
        with h5py.File(path, "w") as f:
            f["coordinates"] = np.random.rand(100, 2)
            f["fields"]      = np.random.rand(60, 1)
        try:
            HDF5DataSource(path=path, config=make_config()).load()
            check("row mismatch raises ValueError", False)
        except ValueError:
            check("row mismatch raises ValueError", True)

    # coordinates that are not 2-D
    with temp_h5() as path:
        with h5py.File(path, "w") as f:
            f["coordinates"] = np.random.rand(100)
            f["fields"]      = np.random.rand(100, 1)
        try:
            HDF5DataSource(path=path, config=make_config()).load()
            check("1-D coordinates raise ValueError", False)
        except ValueError:
            check("1-D coordinates raise ValueError", True)


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("AIPlasma — DataPipeline Test (HDF5)")
    print("=" * 60)

    start = time.perf_counter()

    test_hdf5_source_basic()
    test_hdf5_source_missing_file()
    test_hdf5_source_max_points()
    test_hdf5_custom_keys()
    test_hdf5_custom_layout_hooks()
    test_hdf5_multifidelity()
    test_hdf5_transformer()
    test_hdf5_stream()
    test_hdf5_errors()

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    print(f"REZULTATI: {PASSED} passed, {FAILED} failed | {elapsed:.2f}s")
    if FAILED == 0:
        print("TEST PROSAO ✓")
    else:
        print(f"TEST NIJE PROSAO ✗ — {FAILED} test(ova) palo")
    print("=" * 60)


if __name__ == "__main__":
    run()