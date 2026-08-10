# ADR-002: Domain-Scoped Problem Hierarchy

## Status
Accepted

## Context
AIPlasma is designed as a physics-informed ML framework for nuclear 
physics applications. The initial scope covers plasma physics and fusion,
but the framework architecture should not prevent future expansion into
related domains such as radiation detection or nuclear material modeling.

The base PhysicsProblem interface is intentionally domain-agnostic -
it defines only the structural contract (pde_residual, boundary_conditions,
loss, compile) without any assumptions about the physical domain.

## Decision
Introduce a domain-scoped intermediate layer between PhysicsProblem
and concrete problem implementations:

    PhysicsProblem                  ← domain-agnostic base
        ├── PlasmaPhysicsProblem    ← active domain, full implementation
        │       └── UserProblem     ← end-user subclass
        ├── RadiationProblem        ← stub, reserved for future work
        └── NuclearMaterialProblem  ← stub, reserved for future work

PlasmaPhysicsProblem extends PhysicsProblem with plasma-specific
abstractions shared across all plasma problems:
- Coordinate systems (toroidal, poloidal)
- Bohm unit normalization
- Common plasma operators (Grad-Shafranov, Poisson bracket)
- Plasma-specific validation logic

RadiationProblem and NuclearMaterialProblem are defined as empty
stubs with docstrings describing their intended scope. They contain
no implementation and are not part of the active development roadmap.

## Rationale
- PlasmaPhysicsProblem itself acts as a framework within AIPlasma:
  users derive their own problems from it without modifying core code
- The stub hierarchy signals that the architecture is designed for 
  domain expansion without requiring premature implementation
- Follows the Open/Closed principle: open for extension across domains,
  closed for modification of the base interface
- Domain-specific abstractions (Bohm units, toroidal geometry) belong
  in PlasmaPhysicsProblem, not in the domain-agnostic PhysicsProblem,
  keeping the base interface clean and reusable

## Consequences
- All current development targets PlasmaPhysicsProblem and its subclasses
- Future domain expansion requires implementing a new intermediate class
  at the same level as PlasmaPhysicsProblem, with no changes to
  PhysicsProblem or PlasmaPhysicsProblem
- Stub classes must not contain placeholder logic - only docstrings
  describing intended scope and expected abstractions