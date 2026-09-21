#Test 3
"""
test_data_pipeline_synthetic.py
DataPipeline test using SyntheticDataSource.

Run:
    python tests/test_data_pipeline_synthetic.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import time
import torch

from data.data_parser import DataSourceConfig, SyntheticDataSource
from data.data_pipeline import DataPipeline
from data.sources.physical_tensor import PhysicalTensor, UnitSystem, Domain, CoordinateSystem
from data.preprocessing.feature_pipeline import FeatureBatch, FeatureBatchDataset
from data.preprocessing.data_transformer import DataTransformer, TransformerConfig, FidelityAssigner, FidelityConfig
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


def make_config(n_points: int = 200) -> DataSourceConfig:
    return DataSourceConfig(
        coord_system=CoordinateSystem.CARTESIAN,
        units=UnitSystem(spatial="m", temporal="s", field="normalized"),
        domain=Domain(x_range=(0.0, 1.0), t_range=(0.0, 1.0)),
    )


def heat_solution(x, t, alpha=0.01):
    return torch.sin(math.pi * x) * torch.exp(
        torch.tensor(-alpha * math.pi ** 2) * t
    )


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

def test_synthetic_source():
    section("SyntheticDataSource")

    config = make_config()
    source = SyntheticDataSource(
        solution_fn=heat_solution,
        config=config,
        n_points=300,
    )

    check("validate() returns True", source.validate())

    tensor = source.load()
    check("load() returns PhysicalTensor", isinstance(tensor, PhysicalTensor))
    check("correct n_points", tensor.n_points() == 300)
    check("correct n_fields", tensor.n_fields() == 1)
    check("coordinates shape (N, 2)", tensor.coordinates.shape == (300, 2))
    check("values shape (N, 1)", tensor.values.shape == (300, 1))
    check("metadata has source_type", tensor.metadata.get("source_type") == "synthetic")
    check("coord_system is CARTESIAN", tensor.coord_system == CoordinateSystem.CARTESIAN)

    # Domain contains check
    mask = tensor.domain.contains(tensor.coordinates)
    check("all points within domain", mask.all().item())

    # With noise
    noisy_source = SyntheticDataSource(
        solution_fn=heat_solution,
        config=config,
        n_points=100,
        noise_std=0.05,
    )
    noisy_tensor = noisy_source.load()
    check("noisy source loads", noisy_tensor.n_points() == 100)
    check("noisy metadata has noise_std", noisy_tensor.metadata.get("noise_std") == 0.05)

    # max_points
    limited_config        = make_config()
    limited_config.max_points = 50
    limited_source = SyntheticDataSource(
        solution_fn=heat_solution,
        config=limited_config,
        n_points=200,
    )
    limited_tensor = limited_source.load()
    check("max_points respected", limited_tensor.n_points() == 50)

    # Missing domain raises
    bad_config = DataSourceConfig(
        coord_system=CoordinateSystem.CARTESIAN,
        units=UnitSystem(spatial="m", temporal="s", field="normalized"),
        domain=None,
    )
    bad_source = SyntheticDataSource(solution_fn=heat_solution, config=bad_config)
    check("validate() False when domain is None", not bad_source.validate())


def test_fidelity_assigner():
    section("FidelityAssigner")

    config = make_config()

    coarse = SyntheticDataSource(solution_fn=heat_solution, config=config, n_points=50).load()
    medium = SyntheticDataSource(solution_fn=heat_solution, config=config, n_points=150).load()
    fine   = SyntheticDataSource(solution_fn=heat_solution, config=config, n_points=400).load()

    assigner = FidelityAssigner()

    # Single source
    levels = assigner.assign([coarse])
    check("single source → level 0", levels == [0])

    # Two sources
    levels = assigner.assign([coarse, fine])
    check("two sources → different levels", levels[0] != levels[1])

    # Three sources
    levels = assigner.assign([coarse, medium, fine])
    check("three sources → 3 distinct levels", len(set(levels)) == 3)
    check("finer source gets higher level", levels[2] > levels[0])

    # Determinism
    levels_a = assigner.assign([coarse, medium, fine])
    levels_b = assigner.assign([coarse, medium, fine])
    check("assigner is deterministic", levels_a == levels_b)

    # Empty list raises
    try:
        assigner.assign([])
        check("empty list raises ValueError", False)
    except ValueError:
        check("empty list raises ValueError", True)

    # Incompatible coord systems
    toroidal = SyntheticDataSource(
        solution_fn=heat_solution,
        config=DataSourceConfig(
            coord_system=CoordinateSystem.TOROIDAL,
            units=UnitSystem(spatial="m", temporal="s", field="normalized"),
            domain=Domain(x_range=(0.0, 1.0), t_range=(0.0, 1.0)),
        ),
        n_points=50,
    ).load()
    check("validate() False for incompatible coord systems",
          not assigner.validate([coarse, toroidal]))


def test_data_transformer():
    section("DataTransformer")

    config = make_config()
    sources = [
        SyntheticDataSource(solution_fn=heat_solution, config=config, n_points=50).load(),
        SyntheticDataSource(solution_fn=heat_solution, config=config, n_points=200).load(),
    ]

    transformer = DataTransformer()
    batches     = transformer.transform(sources)

    check("transform returns list", isinstance(batches, list))
    check("correct number of batches", len(batches) == 2)
    check("all items are FeatureBatch", all(isinstance(b, FeatureBatch) for b in batches))

    for i, b in enumerate(batches):
        overlap = b.boundary_mask & b.collocation_mask
        check(f"batch {i} masks are disjoint", not overlap.any())
        check(f"batch {i} has fidelity_level", isinstance(b.fidelity_level, int))
        check(f"batch {i} fidelity_weight > 0", b.fidelity_weight > 0)
        check(f"batch {i} coords shape", b.coords.shape[1] == 2)

    # Fidelity weights sum to 1.0
    weights = [b.fidelity_weight for b in batches]
    check("fidelity weights sum to 1.0", abs(sum(weights) - 1.0) < 1e-5)

    # Values normalized to [0, 1]
    for i, b in enumerate(batches):
        vals = b.fields
        check(f"batch {i} values in [0, 1]",
              vals.min().item() >= -1e-5 and vals.max().item() <= 1.0 + 1e-5)


def test_data_pipeline():
    section("DataPipeline")

    config = make_config()
    sources = [
        SyntheticDataSource(solution_fn=heat_solution, config=config, n_points=100),
        SyntheticDataSource(solution_fn=heat_solution, config=config, n_points=300),
    ]

    pipeline = DataPipeline(sources=sources)
    train_ds, val_ds = pipeline.build()

    check("build() returns two datasets",
          isinstance(train_ds, FeatureBatchDataset) and
          isinstance(val_ds, FeatureBatchDataset))
    check("train dataset not empty", len(train_ds) > 0)
    check("val dataset not empty", len(val_ds) > 0)
    check("total size = n_sources", len(train_ds) + len(val_ds) == len(sources))

    # describe()
    desc = pipeline.describe()
    check("describe() returns string", isinstance(desc, str) and len(desc) > 0)

    # Multi-fidelity pipeline with 3 sources
    sources_3 = [
        SyntheticDataSource(solution_fn=heat_solution, config=config, n_points=50),
        SyntheticDataSource(solution_fn=heat_solution, config=config, n_points=150),
        SyntheticDataSource(solution_fn=heat_solution, config=config, n_points=400),
    ]
    pipeline_3       = DataPipeline(sources=sources_3)
    train_3, val_3   = pipeline_3.build()
    all_batches      = [train_3[i] for i in range(len(train_3))] + \
                       [val_3[i]   for i in range(len(val_3))]
    fidelity_levels  = [b.fidelity_level for b in all_batches]
    check("3-source pipeline produces 3 fidelity levels",
          len(set(fidelity_levels)) == 3)


def test_stream():
    section("DataSource streaming")

    config = make_config()
    source = SyntheticDataSource(solution_fn=heat_solution, config=config, n_points=100)
    chunks = list(source.stream(chunk_size=30))

    check("stream produces multiple chunks", len(chunks) > 1)
    total = sum(c.n_points() for c in chunks)
    check("stream total points = n_points", total == 100)
    check("each chunk is PhysicalTensor", all(isinstance(c, PhysicalTensor) for c in chunks))


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("AIPlasma — DataPipeline Test (Synthetic)")
    print("=" * 60)

    start = time.perf_counter()

    test_synthetic_source()
    test_fidelity_assigner()
    test_data_transformer()
    test_data_pipeline()
    test_stream()

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