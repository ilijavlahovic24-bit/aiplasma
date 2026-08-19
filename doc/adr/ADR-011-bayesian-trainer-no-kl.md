# ADR-011: BayesianTrainer Uses Standard Loss Without Explicit KL Term

## Status
Accepted

## Context
BayesianTrainer.train_step() needs a loss function for training
BayesianNN. Two options were considered:

1. Standard loss only - problem.loss() with MC Dropout acting as
   implicit regularization. No explicit KL divergence term.
2. KL approximation - add L2 weight regularization as an
   approximation of KL divergence between posterior and prior:
   loss = problem.loss() + kl_weight * l2_reg

## Decision
Use standard problem.loss() without an explicit KL term (Option A).

## Rationale
- Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
  shows that MC Dropout already implicitly minimizes a form of
  KL divergence through the dropout regularization mechanism -
  no explicit KL term is needed
- Option B is the correct approach for Variational Inference,
  which was explicitly rejected in ADR-007 in favor of MC Dropout
- Adding kl_weight introduces a new hyperparameter that users
  would need to tune - inconsistent with the framework goal of
  being accessible to physics researchers
- BayesianTrainer remains consistent with PINNTrainer in its
  loss computation - both delegate to problem.loss(), keeping
  the training interface uniform
- The key difference between BayesianTrainer and PINNTrainer is
  in val_step(): BayesianTrainer uses sample_predictions() for
  uncertainty-aware evaluation instead of a single forward pass

## Consequences
- BayesianTrainer.train_step() is nearly identical to
  PINNTrainer.train_step() - the Bayesian behavior comes from
  the model architecture (MC Dropout), not the trainer
- val_step() uses sample_predictions() which runs N forward
  passes - validation is N times slower than PINNTrainer
- If Variational Inference is added in a future version,
  a VIBayesianTrainer subclass can add the KL term explicitly
  without modifying BayesianTrainer