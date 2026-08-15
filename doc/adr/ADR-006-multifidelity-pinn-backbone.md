# ADR-006: MultiFidelityPINN inherits ResidualPINN

## Status
Accepted

## Context
MultiFidelityPINN needs a base architecture to build upon.
Three options were considered:

1. Inherit BasePINN - simple, user controls depth manually
2. Inherit ResidualPINN - residual connections built-in
3. Inherit BasePINN with backbone argument - maximum flexibility,
   user chooses architecture explicitly

## Decision
MultiFidelityPINN inherits ResidualPINN.

## Rationale
- Multi-fidelity training requires the network to learn mappings
  across different resolution levels, which benefits from deeper
  networks with stable gradient flow
- Residual connections directly address the vanishing gradient
  problem that appears in deeper multi-fidelity architectures
- Simpler API than Option C - user does not need to instantiate
  a separate backbone before creating MultiFidelityPINN
- Option C adds flexibility that is not needed in v1 - if a user
  needs a custom backbone, they can subclass MultiFidelityPINN
  directly, which is consistent with ADR-001

## Consequences
- MultiFidelityPINN always uses residual blocks - no way to
  switch to standard layers without subclassing
- input_dim passed to MultiFidelityPINN must account for the
  fidelity embedding dimension being concatenated to coords:
  effective_input = original_input_dim + fidelity_embed_dim
  This is handled internally - user passes original input_dim
- If Option C is needed in future versions, it can be added
  as a factory method without breaking existing subclasses