# ADR-019: RLTrainer Policy and Action Convention

## Status
Accepted (amends ADR-014)

## Context
ADR-014 defines RLTrainer but not the shape contract between the
policy network and PlasmaEnvironment.step(). The first
implementation averaged predictions over points, which worked only
when output_dim happened to equal the number of control points.

## Decision
- The policy is a PhysicsModel with output_dim=1. Its (N, 1)
  output is squeezed to (N,): one control value per point, used as
  the mean of a Gaussian policy with fixed standard deviation 0.1.
- Training samples actions; validation uses the mean action.
- Entropy regularization is averaged over every step collected in
  the episode.
- train_step() performs backward() and the optimizer step itself.
  BaseTrainer.fit() only reads loss.item().
- SyntheticPlasmaEnv takes u_target as an optional constructor
  argument (shape (n_points,), validated). The default remains
  sin(pi * x).

## Consequences
- The environment action_dim() must equal the number of points in
  its state, as in SyntheticPlasmaEnv.
- Learned or state-dependent action variance is left to PPO and
  SAC subclasses (v3).