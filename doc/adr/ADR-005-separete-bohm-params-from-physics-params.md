# ADR-005: Separate bohm_params() from physics_params()

## Status
Accepted

## Context
PlasmaPhysicsProblem needs reference values for Bohm normalization
(T_e, B, n_0) in addition to PDE parameters (D, C, nu).
Two options were considered:
- Option A: Include Bohm parameters in physics_params()
- Option B: Separate bohm_params() method

## Decision
Introduce a dedicated bohm_params() method in PlasmaPhysicsProblem,
separate from physics_params().

## Rationale
- Clear separation of concerns: PDE parameters and normalization
  parameters have fundamentally different roles
- PhysicsEquation.validate_params() works only with PDE parameters,
  without normalization noise
- Easier to test in isolation
- PlasmaPhysicsProblem provides default Bohm values for typical
  tokamak conditions - user overrides only what they need
- Tutorial notebook can demonstrate each method in a separate cell,
  making the framework easier to learn

## Consequences
- Users must be aware that bohm_params() exists - tutorial notebook
  covers this explicitly
- bohm_params() has default values in PlasmaPhysicsProblem so
  forgetting to override it does not cause runtime errors
- physics_params() remains clean - only PDE parameters