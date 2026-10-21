#Test 12

"""
test_se3_transformer.py
SE3Transformer tests — shapes, validation, and above all equivariance.

Run:
    python tests/test_se3_transformer.py
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing.feature_pipeline import FeatureBatch
from core.base_model import ModelOutput
from models.equivariant.se3_transformer import SE3Transformer

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


def raises_value_error(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def clone_batch(batch: FeatureBatch, coords=None, fields=None) -> FeatureBatch:
    return FeatureBatch(
        coords=batch.coords if coords is None else coords,
        fields=batch.fields if fields is None else fields,
        boundary_mask=batch.boundary_mask,
        collocation_mask=batch.collocation_mask,
        physics_params=batch.physics_params,
        fidelity_level=batch.fidelity_level,
        fidelity_weight=batch.fidelity_weight,
    )


def make_batch(n: int = 40) -> FeatureBatch:
    """
    coords: (x, y, z, t)     — columns 0-2 spatial, column 3 time (invariant)
    fields: (s, vx, vy, vz)  — column 0 a scalar, columns 1-3 a vector
    """
    dtype = torch.float64
    return FeatureBatch(
        coords=torch.rand(n, 4, dtype=dtype),
        fields=torch.randn(n, 4, dtype=dtype),
        boundary_mask=torch.zeros(n, dtype=torch.bool),
        collocation_mask=torch.ones(n, dtype=torch.bool),
        physics_params={},
        fidelity_level=0,
        fidelity_weight=1.0,
    )


def make_model(**kwargs) -> SE3Transformer:
    params = dict(output_dim=1, channels=16, n_layers=2, n_heads=2,
                  k_neighbors=8, invariant_dims=(3,))
    params.update(kwargs)
    return SE3Transformer(**params).double()


def random_rotation() -> torch.Tensor:
    """Random proper rotation (det = +1)."""
    q, r = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rigid_transform(batch: FeatureBatch, R: torch.Tensor, shift: torch.Tensor) -> FeatureBatch:
    """Rotates + translates positions; rotates the vector field (it does not translate)."""
    coords = batch.coords.clone()
    coords[:, :3] = batch.coords[:, :3] @ R.T + shift
    fields = batch.fields.clone()
    fields[:, 1:4] = batch.fields[:, 1:4] @ R.T
    return clone_batch(batch, coords=coords, fields=fields)


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

def test_construction():
    section("SE3Transformer construction")

    model = make_model()
    check("default construction", model is not None)
    check("summary mentions SE3Transformer", "SE3Transformer" in model.summary())
    check("summary mentions parameters", "parameters" in model.summary())

    check("2 spatial dims raises ValueError",
          raises_value_error(lambda: SE3Transformer(spatial_dims=(0, 1))))
    check("overlapping spatial/invariant dims raises ValueError",
          raises_value_error(lambda: SE3Transformer(invariant_dims=(2,))))
    check("vector group of 2 raises ValueError",
          raises_value_error(lambda: SE3Transformer(vector_fields=((0, 1),))))
    check("channels not divisible by heads raises ValueError",
          raises_value_error(lambda: SE3Transformer(channels=10, n_heads=4)))
    check("k_neighbors=0 raises ValueError",
          raises_value_error(lambda: SE3Transformer(k_neighbors=0)))
    check("n_layers=0 raises ValueError",
          raises_value_error(lambda: SE3Transformer(n_layers=0)))


def test_forward():
    section("SE3Transformer forward")

    batch  = make_batch()
    output = make_model().forward(batch)
    check("returns ModelOutput", isinstance(output, ModelOutput))
    check("pred shape (N, 1)", output.pred.shape == (40, 1))
    check("uncertainty is None", output.uncertainty is None)
    check("pred is finite", torch.isfinite(output.pred).all().item())
    check("different nodes get different predictions", output.pred.std().item() > 0)

    full = make_model(output_dim=2, scalar_fields=(0,), vector_fields=((1, 2, 3),))
    check("scalar + vector inputs, output_dim=2",
          full.forward(batch).pred.shape == (40, 2))

    no_invariant = make_model(invariant_dims=())
    check("no scalar inputs at all still works",
          no_invariant.forward(batch).pred.shape == (40, 1))


def test_equivariance():
    section("SE3Transformer equivariance")

    batch = make_batch()
    R     = random_rotation()
    shift = torch.randn(3, dtype=torch.float64)

    configs = [
        ("geometry + time only", {}),
        ("scalar + vector inputs", dict(scalar_fields=(0,), vector_fields=((1, 2, 3),))),
    ]
    for name, kwargs in configs:
        model = make_model(**kwargs)

        out       = model.forward(batch).pred
        out_rigid = model.forward(rigid_transform(batch, R, shift)).pred
        diff      = (out - out_rigid).abs().max().item()
        check(f"{name}: invariant to rotation + translation",
              diff < 1e-7, f"max diff={diff:.2e}")

        out_shift = model.forward(
            rigid_transform(batch, torch.eye(3, dtype=torch.float64), shift)
        ).pred
        diff = (out - out_shift).abs().max().item()
        check(f"{name}: invariant to translation alone",
              diff < 1e-7, f"max diff={diff:.2e}")


def test_sensitivity():
    section("SE3Transformer is not trivially invariant")

    batch = make_batch()
    model = make_model(scalar_fields=(0,), vector_fields=((1, 2, 3),))
    out   = model.forward(batch).pred

    # Changing the geometry must change the output
    coords = batch.coords.clone()
    coords[:, :3] += 0.1 * torch.randn(40, 3, dtype=torch.float64)
    out_geo = model.forward(clone_batch(batch, coords=coords)).pred
    check("perturbed geometry changes output", not torch.allclose(out, out_geo, atol=1e-6))

    # Rotating ONLY the vector inputs (not the geometry) is a relative change
    R      = random_rotation()
    fields = batch.fields.clone()
    fields[:, 1:4] = batch.fields[:, 1:4] @ R.T
    out_vec = model.forward(clone_batch(batch, fields=fields)).pred
    check("rotating vector inputs alone changes output",
          not torch.allclose(out, out_vec, atol=1e-6))


def test_small_clouds():
    section("SE3Transformer small point clouds")

    model = make_model(k_neighbors=8)
    small = make_batch(n=5)
    check("k_neighbors is clamped to N - 1", model.forward(small).pred.shape == (5, 1))
    check("single point raises ValueError",
          raises_value_error(lambda: model.forward(make_batch(n=1))))


def test_backprop():
    section("SE3Transformer backpropagation")

    batch = make_batch()
    model = make_model(scalar_fields=(0,), vector_fields=((1, 2, 3),))
    loss  = model.forward(batch).pred.pow(2).mean()

    try:
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        check("backprop works", True)
        check("all params have gradients", all(g is not None for g in grads))
        check("gradients are finite",
              all(torch.isfinite(g).all().item() for g in grads if g is not None))
    except Exception as e:
        check("backprop works", False, str(e))
        check("all params have gradients", False)
        check("gradients are finite", False)


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("AIPlasma — SE3Transformer Test")
    print("=" * 60)

    start = time.perf_counter()

    test_construction()
    test_forward()
    test_equivariance()
    test_sensitivity()
    test_small_clouds()
    test_backprop()

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