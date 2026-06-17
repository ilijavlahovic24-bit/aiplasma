# ADR-004: Abstract Training Steps with Framework-Provided Loop

## Status
Accepted

## Context
BaseTrainer needs to define how the training loop is structured.
Two extreme approaches were considered:

1. Fully abstract loop — each trainer implements the entire training
   loop from scratch, giving maximum freedom but risking code
   duplication across PINNTrainer, BayesianTrainer, and future trainers.

2. Fully concrete loop — BaseTrainer provides the complete loop,
   subclasses only override specific hooks. Reduces duplication but
   limits flexibility for trainers with fundamentally different
   training dynamics (e.g. Bayesian sampling vs. gradient descent).

## Decision
Split responsibilities between the framework and the user:

- train_step(batch) → abstractmethod, user implements
- val_step(batch)   → abstractmethod, user implements
- fit()             → framework provides, orchestrates everything else

fit() handles: epoch loop, validation frequency, callback lifecycle
(on_epoch_start, on_epoch_end, on_train_end), checkpointing triggers,
early stopping checks, and device management.

train_step() and val_step() are the only mandatory abstractions —
they define what happens in a single step, nothing more.

## Rationale
- Eliminates duplication of epoch loop, validation, and callback
  logic across PINNTrainer, BayesianTrainer, and future trainers
- Preserves full freedom over what happens inside each step —
  PINN computes PDE residual, Bayesian samples from posterior,
  Ensemble aggregates — none of this is constrained by the framework
- Consistent with the subclassing philosophy established in ADR-001:
  users override the minimum necessary, framework handles the rest
- Mirrors PyTorch Lightning's proven LightningModule pattern,
  which is already familiar to the target user base
- fit() as a framework method means distributed training support
  (ADR-003 consequence) can be added in one place without touching
  any concrete trainer implementation

## Consequences
- All concrete trainers (PINNTrainer, BayesianTrainer) must implement
  both train_step() and val_step() — neither has a default
- fit() is sealed — subclasses should not override it to preserve
  consistent training behavior across the framework
- Callback lifecycle is tied to fit(), so callbacks only fire when
  fit() is used directly; manual step-by-step training bypasses them