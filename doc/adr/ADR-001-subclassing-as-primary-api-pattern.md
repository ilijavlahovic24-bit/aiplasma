# ADR-001: Subclassing as primary API pattern

## Status
Accepted

## Context
AIPlasma needs a user-facing API for declaring physics problems.
Options considered: YAML config, decorators, subclassing.

## Decision
Subclassing of PhysicsProblem as the primary API pattern.

## Rationale
- Consistent with PyTorch nn.Module idiom familiar to target users
- Full IDE support and type checking
- Custom PDE definitions are a core use case, not edge case
- YAML wrapper can be added on top later without redesign

## Consequences
- More boilerplate for simple cases vs YAML
- Users need to understand which methods to override
- Serialization requires additional infrastructure