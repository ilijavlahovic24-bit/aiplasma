# ADR-016: SE(3)-Transformer Implemented from Scratch (Degrees 0 and 1)

## Status
Accepted

## Context
Problems with 3D geometry (tokamak or stellarator points, atomic
positions, vector fields) benefit from models that are equivariant
to rotations and translations. Options considered:

1. From scratch in plain PyTorch, features of degree 0 and 1 only
2. e3nn as building blocks, attention layer written by us
3. Wrapping an existing implementation (the NVIDIA reference
   implementation works on DGL graph objects)

A FeatureBatch is a point set without edges, and its coords may
mix space and time.

## Decision
Option 1. Plain PyTorch, no new dependency.
- Features: scalars (degree 0) and 3-vectors (degree 1).
- Edge kernel with five paths: s, v.u, s*u, v, v x u (u is the
  unit edge direction), weighted by radial MLPs on a Gaussian
  basis of the edge length.
- Attention logits are invariant inner products over scalar and
  vector channels, softmax over neighbours, multiple heads.
- Vector channels are mixed without bias and gated by scalars.
  The last layer skips the vector update.
- Neighbours: k nearest by spatial distance (k configurable,
  clamped to N - 1, no self-edges).
- Inputs are declared by the user: spatial_dims (exactly three
  columns), invariant_dims (for example time), scalar_fields and
  vector_fields (groups of three columns).
- Output: a scalar per point, invariant to rotation and
  translation.
- Activation: SiLU.
- The equivariance test (rotation + translation, float64) is the
  primary test.

## Rationale
- Simplest correct implementation first, as in ADR-014.
- Plain PyTorch keeps second-order autograd straightforward and
  avoids a new dependency and a graph-format conversion.
- kNN gives a fixed number of neighbours and one hyper-parameter.
- Declaring field roles keeps target columns from leaking in as
  inputs when fields holds the training targets.

## Consequences
- The model sees only relative geometry, invariant coordinates and
  input fields. It cannot represent functions of absolute
  position, so it fits symmetric or feature-driven problems.
- Degrees 2 and above are unavailable. e3nn is the upgrade path.
- Neighbour search uses a dense distance matrix (O(N^2) memory).
- One point cloud per forward call; fidelity_level is ignored.
- No PDE derivatives are provided (see ADR-017).