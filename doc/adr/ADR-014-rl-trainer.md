# ADR-014: RLTrainer Inherits BaseTrainer with PlasmaEnvironment Abstraction

## Status
Accepted

## Context
AIPlasma needs a Reinforcement Learning trainer for plasma control
problems, directly relevant to plasma control research (e.g. DeepMind
TCV tokamak work). Two approaches were considered:

1. Separate RL loop outside BaseTrainer - custom training loop
   incompatible with existing callback and fit() infrastructure
2. RLTrainer inherits BaseTrainer - train_step() maps RL concepts
   to the existing FeatureBatch interface

The key insight: in plasma control, FeatureBatch naturally represents
environment state (plasma diagnostics), and physics_params carries
control parameters (magnetic field, heating). This makes train_step()
a valid abstraction for RL as well:
    state   = batch           - current plasma state
    action  = policy(state)   - control action
    reward  = env.step(action) - feedback from simulation
    loss    = -reward.mean()  - policy gradient loss

## Decision
RLTrainer inherits BaseTrainer. A new PlasmaEnvironment abstract
class is introduced in trainers/environments/plasma_env.py:

    PlasmaEnvironment (ABC)
        step(action: Tensor) -> Tensor   # returns reward
        reset() -> FeatureBatch          # returns initial state
        state() -> FeatureBatch          # returns current state

RLTrainer uses REINFORCE (policy gradient) as the default algorithm.
train_step() receives a FeatureBatch state and returns policy loss.

## Rationale
- Inheriting BaseTrainer reuses fit(), callback system, CheckPointing,
  EarlyStopping, and PhysicsMonitor without modification
- FeatureBatch as state is natural for plasma problems - diagnostics
  map directly to coords and fields
- REINFORCE is the simplest policy gradient algorithm - consistent
  with the v1 philosophy of simplest correct implementation first
- PlasmaEnvironment abstraction allows users to plug in different
  simulators (analytical, numerical, surrogate model) without
  changing RLTrainer
- Consistent with ADR-001: user subclasses PlasmaEnvironment to
  define their specific plasma control problem

## Consequences
- New trainers/environments/ directory with PlasmaEnvironment ABC
- RLTrainer requires a PlasmaEnvironment instance at construction
- train_step() ignores the batch argument from fit() and instead
  calls env.state() - the batch parameter is kept for interface
  consistency with BaseTrainer
- val_step() evaluates policy by running one full episode
- Policy network is a separate PhysicsModel instance passed to
  RLTrainer - consistent with how PINNTrainer receives a model
- More advanced RL algorithms (PPO, SAC) can be added as subclasses
  of RLTrainer without modifying BaseTrainer