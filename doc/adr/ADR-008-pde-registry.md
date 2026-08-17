# ADR-008: PDE Registry as Central Equation Store

## Status
Accepted

## Context
PhysicsEquation instances need to be accessible across the framework -
in PlasmaPhysicsProblem, AutogradPDESolver, and user code.
Two options were considered:

1. Direct instantiation - user creates equation objects directly:
   equation = HeatEquation1D()

2. Central registry - user accesses equations by name:
   equation = REGISTRY.get("heat_equation_1d")

## Decision
Introduce PDERegistry as a module-level singleton in
solvers/physics_constraints.py. All built-in equations are
registered automatically at import time. Users register
custom equations via REGISTRY.register().

## Rationale
- Declarative API: user specifies equation by name, not class -
  consistent with the subclassing philosophy (ADR-001)
- Extensibility: REGISTRY.register(MyEquation()) adds a custom
  equation without modifying framework code
- Validation: REGISTRY.get() raises KeyError with a list of
  available equations if the name is not found, giving a clear
  error message instead of an ImportError
- Tutorial notebook: REGISTRY.list_all() shows all available
  equations in one cell - better discoverability for new users
- AutogradPDESolver and PlasmaPhysicsProblem both receive
  equations through the registry, keeping their interfaces
  consistent

## Consequences
- All built-in equations must have unique snake_case names
  registered at module import time
- Custom equations must be registered before use -
  REGISTRY.register() raises ValueError on duplicate names
- REGISTRY is a global singleton - tests that register custom
  equations should use REGISTRY.unregister() in teardown
  to avoid polluting other tests