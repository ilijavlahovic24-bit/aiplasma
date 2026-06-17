# ADR-003: Dual-Mode Solver Interface

## Status
Accepted

## Context
PhysicsSolver needs to compute physical quantities from model predictions.
Two distinct use cases exist:

1. Post-processing and evaluation — no gradient tracking needed,
   works with already-computed predictions (ModelOutput)
2. PDE residual computation during training — requires autograd
   through the model to compute spatial/temporal derivatives

A single-mode solver would force one of two bad trade-offs:
- Always passing the model → unnecessary coupling for evaluation use cases
- Always passing ModelOutput → impossible to differentiate through the model

## Decision
PhysicsSolver exposes two methods:
- solve(output, batch) → works with ModelOutput, no autograd
- solve_with_grad(model, batch) → works with PhysicsModel, full autograd

Both return SolverOutput. The caller decides which mode is appropriate.

## Rationale
- Modularity: evaluation and training are distinct concerns
- Simplicity: users who only need post-processing never touch the model
- Flexibility: PINN training can differentiate through the model freely
- Consistent return type: both modes return SolverOutput for uniform
  downstream handling

## Consequences
- Subclasses must implement both methods
- solve() cannot compute quantities that require gradients through the model
- solve_with_grad() should not be called during inference — only training