# ADR-017: Domain Unit Normalization Protocol

## Status
Accepted

## Context
PlasmaPhysicsProblem provides to_bohm_units() and
from_bohm_units(). Nuclear problems need the same capability:
rate constants and diffusivities in point-defect theory differ by
many orders of magnitude, and a PINN trained on SI values
converges poorly.

Options:
A. Each domain defines scales() and to/from normalized-unit
   methods, following the Bohm pattern
B. The user supplies dimensionless data and parameters directly

## Decision
Option A. Nuclear and radiation problems provide:
- scales(): dict of reference scales (length, time and the
  domain-specific value scales, such as temperature, concentration
  or flux)
- to_normalized_units(tensor) and from_normalized_units(tensor) on
  PhysicalTensor. The scales used are stored in tensor.metadata
  with a "normalized" flag. from_normalized_units() raises
  ValueError for a tensor that was not normalized.
- scales() stays separate from physics_params() (ADR-005).
PlasmaPhysicsProblem keeps its Bohm methods unchanged.

## Rationale
- Reuses a pattern the framework already has; training then works
  on quantities of order one.
- Scales live in one method, so they are easy to inspect and
  override per experiment.

## Consequences
- Nondimensionalizing the PDE parameters (diffusivities, rate
  constants) with scales() is the user's responsibility for now.
- A shared protocol across domains, plasma included, may be
  extracted later.