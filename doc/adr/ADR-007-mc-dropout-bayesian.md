# ADR-007: MC Dropout for Bayesian Inference in BayesianNN

## Status
Accepted

## Context
BayesianNN needs a method for approximate Bayesian inference to
produce uncertainty estimates alongside predictions. Three main
approaches were considered:

1. Monte Carlo Dropout - keep dropout active during inference,
   run N forward passes, compute mean and std across samples
2. Variational Inference - explicitly model weight distributions
   using variational parameters (e.g. Bayes by Backprop)
3. Deep Ensembles - train multiple independent networks and
   aggregate their predictions

## Decision
Use Monte Carlo Dropout (Option 1) with dropout applied only
between hidden layers (not on input or output layer).

Dropout placement (Option B within MC Dropout):
    Linear(input) → Activation →
    [Dropout → Linear → Activation] * hidden_layers →
    Linear(output)

## Rationale
- MC Dropout is the simplest Bayesian approximation that works
  with standard PyTorch - no external libraries required
- Significantly lower implementation complexity than Variational
  Inference, which requires custom loss terms (ELBO) and
  reparameterization tricks
- Lower computational cost than Deep Ensembles, which require
  training N separate models
- Dropout between hidden layers only (not on input/output) is
  the standard practical choice - avoids corrupting raw
  coordinates and final predictions while still producing
  meaningful uncertainty estimates
- Consistent with the framework goal of being accessible to
  physics researchers who may not have ML backgrounds

## Consequences
- Uncertainty quality depends on dropout_rate and n_samples -
  these are user-configurable hyperparameters
- MC Dropout underestimates uncertainty compared to true
  Bayesian inference - acceptable for v1 research use cases
- If higher-quality uncertainty is needed in future versions,
  Variational Inference can be added as a separate model class
  (VIBayesianNN) without modifying BayesianNN
- sample_predictions() keeps dropout active locally via
  self.net.train() - this does not affect the outer training
  state managed by BaseTrainer