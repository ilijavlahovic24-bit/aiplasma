#Test 4


"""
test_data_pipeline_csv.py
DataPipeline test using CSVDataSource.

Generates a temporary CSV file, runs the pipeline, then cleans up.

Run:
    python tests/test_data_pipeline_csv.py
"""

import sys
import os
import math
import time
import csv
import tempfile
import torch

from data_parser import DataSourceConfig, CSVDataSource
from data_pipeline import DataPipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def make_csv(path: str, n_rows: int = 150, alpha: float = 0.01) -> None:
    """
    Writes a CSV file with heat equation solution:
        columns: x, t, u
        u = sin(pi*x) * exp(-alpha*pi^2*t)
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "t", "u"])
        for _ in range(n_rows):
            x = float(torch.rand(1))
            t = float(torch.rand(1))
            u = math.sin(math.pi * x) * math.exp(-alpha * math.pi ** 2 * t)
            writer.writerow([x, t, u])


def make_config(domain=None) -> DataSourceConfig:
    return DataSourceConfig(
        coord_system=CoordinateSystem.CARTESIAN,
        units=UnitSystem(spatial="m", temporal="s", field="normalized"),
        domain=domain or Domain(x_range=(0.0, 1.0), t_range=(0.0, 1.0)),
    )


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

def test_csv_source_basic():
    section("CSVDataSource — basic load")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        csv_path = f.name

    try:
        make_csv(csv_path, n_rows=100)

        source = CSVDataSource(
            path=csv_path,
            field_columns=["u"],
            config=make_config(),
        )

        check("validate() returns True", source.validate())

        tensor = source.load()
        check("load() returns PhysicalTensor", isinstance(tensor, PhysicalTensor))
        check("correct n_points", tensor.n_points() == 100)
        check("correct n_fields", tensor.n_fields() == 1)
        check("coordinates shape (N, 2)", tensor.coordinates.shape == (100, 2))
        check("values shape (N, 1)", tensor.values.shape == (100, 1))
        check("metadata has source_type csv",
              tensor.metadata.get("source_type") == "csv")
        check("metadata has path", "path" in tensor.metadata)
        check("metadata has field_columns", "field_columns" in tensor.metadata)

    finally:
        os.unlink(csv_path)


def test_csv_source_missing_file():
    section("CSVDataSource — missing file")

    source = CSVDataSource(
        path="/nonexistent/path/data.csv",
        field_columns=["u"],
        config=make_config(),
    )
    check("validate() False for missing file", not source.validate())

    try:
        source.load()
        check("load() raises ValueError for missing file", False)
    except ValueError:
        check("load() raises ValueError for missing file", True)


def test_csv_source_max_points():
    section("CSVDataSource — max_points")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        csv_path = f.name

    try:
        make_csv(csv_path, n_rows=200)

        config            = make_config()
        config.max_points = 75

        source = CSVDataSource(
            path=csv_path,
            field_columns=["u"],
            config=config,
        )
        tensor = source.load()
        check("max_points respected", tensor.n_points() == 75)

    finally:
        os.unlink(csv_path)


def test_csv_multifidelity():
    section("CSVDataSource — multi-fidelity pipeline")

    paths = []
    try:
        # Create 3 CSV files with different resolutions
        for n in [50, 150, 300]:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                paths.append(f.name)
            make_csv(paths[-1], n_rows=n)

        config = make_config()
        sources = [
            CSVDataSource(path=p, field_columns=["u"], config=config)
            for p in paths
        ]

        pipeline        = DataPipeline(sources=sources)
        train_ds, val_ds = pipeline.build()

        check("pipeline builds successfully",
              isinstance(train_ds, FeatureBatchDataset))
        check("train dataset not empty", len(train_ds) > 0)

        all_batches     = [train_ds[i] for i in range(len(train_ds))] + \
                          [val_ds[i]   for i in range(len(val_ds))]
        fidelity_levels = [b.fidelity_level for b in all_batches]
        check("3 CSV sources → 3 fidelity levels", len(set(fidelity_levels)) == 3)

        weights = [b.fidelity_weight for b in all_batches]
        check("fidelity weights sum to 1.0", abs(sum(weights) - 1.0) < 1e-5)

    finally:
        for p in paths:
            if os.path.exists(p):
                os.unlink(p)


def test_csv_transformer():
    section("CSVDataSource — DataTransformer")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        csv_path = f.name

    try:
        make_csv(csv_path, n_rows=120)

        source      = CSVDataSource(
            path=csv_path, field_columns=["u"], config=make_config()
        )
        tensor      = source.load()
        transformer = DataTransformer()
        batches     = transformer.transform([tensor])

        check("transformer produces FeatureBatch", len(batches) == 1)
        b = batches[0]
        check("boundary_mask and collocation_mask disjoint",
              not (b.boundary_mask & b.collocation_mask).any())
        check("coords shape correct", b.coords.shape[1] == 2)
        check("fields shape correct", b.fields.shape[1] == 1)

    finally:
        os.unlink(csv_path)


def test_csv_stream():
    section("CSVDataSource — streaming")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        csv_path = f.name

    try:
        make_csv(csv_path, n_rows=90)

        source = CSVDataSource(
            path=csv_path, field_columns=["u"], config=make_config()
        )
        chunks = list(source.stream(chunk_size=30))

        check("stream produces 3 chunks", len(chunks) == 3)
        total = sum(c.n_points() for c in chunks)
        check("stream total = n_rows", total == 90)
        check("all chunks are PhysicalTensor",
              all(isinstance(c, PhysicalTensor) for c in chunks))

    finally:
        os.unlink(csv_path)


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("AIPlasma — DataPipeline Test (CSV)")
    print("=" * 60)

    start = time.perf_counter()

    test_csv_source_basic()
    test_csv_source_missing_file()
    test_csv_source_max_points()
    test_csv_multifidelity()
    test_csv_transformer()
    test_csv_stream()

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