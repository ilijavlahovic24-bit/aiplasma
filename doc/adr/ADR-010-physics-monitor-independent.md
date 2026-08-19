# ADR-011: PhysicsMonitor Computes Metrics Independently via Solver

## Status
Accepted

## Context
PhysicsMonitor callback needs to track physics-specific metrics
during training (data_loss, pde_loss, bc_loss) separately from
the total training loss. Two options were considered:

1. Read from history.extra - PINNTrainer writes loss components
   into history.extra, PhysicsMonitor reads them in on_epoch_end()
2. Compute independently via solver - PhysicsMonitor holds a
   reference to the solver and calls solve_with_grad() on a
   validation batch every N epochs

## Decision
PhysicsMonitor computes physics metrics independently using
the solver (Option B).

## Rationale
- Modularity: PhysicsMonitor does not depend on trainer internals -
  it only needs model, problem, solver, and a reference batch
- Consistent with ADR-003 dual-mode solver: solve_with_grad()
for computing PDE residuals independently
- Option A creates implicit coupling between PINNTrainer and
  PhysicsMonitor - if PINNTrainer changes how it writes to
  history.extra, PhysicsMonitor silently breaks
- Option B works with any trainer that uses a PhysicsSolver -
  not just PINNTrainer
- Independent computation allows PhysicsMonitor to run at a
  different frequency than the training loop

## Consequences
- PhysicsMonitor requires model, problem, solver, and a reference
  batch at construction time
- solve_with_grad() is called every log_frequency epochs -
  adds computational overhead proportional to one forward pass
- history.extra is still available for trainers to write
  custom metrics, but PhysicsMonitor does not depend on it
- BayesianTrainer and MultiFidelityTrainer get physics monitoring
  for free without any changes to their train_step()