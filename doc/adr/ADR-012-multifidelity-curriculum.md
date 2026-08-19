# ADR-012: MultiFidelityTrainer Curriculum Learning as Optional Mode

## Status
Accepted

## Context
MultiFidelityTrainer can train on multiple fidelity levels in two ways:

1. Uniform training - all fidelity levels trained equally every epoch.
   Simple, predictable, consistent with PINNTrainer behavior.
2. Curriculum learning - training starts with low-fidelity data and
   gradually introduces higher fidelity levels:
   Epochs 1-N:    only fidelity level 0 (coarsest)
   Epochs N-2N:   fidelity level 0 + 1
   Epochs 2N+:    all levels
   Physically motivated - model learns coarse patterns first,
   then refines them with higher-fidelity data.

## Decision
Uniform training (Option A) is the default behavior.
Curriculum learning (Option B) is an optional mode activated
through MultiFidelityConfig:

    config = MultiFidelityConfig(
        curriculum=True,
        curriculum_epochs_per_level=30,
    )
    trainer = MultiFidelityTrainer(..., mf_config=config)

## Rationale
- Consistent with the hybrid approach used throughout the framework:
  simple default, advanced option available without redesign
  (adaptive loss weights, distributed training, etc.)
- Curriculum learning has physics motivation but is not universally
  better - for smooth problems like heat equation it adds complexity
  without benefit
- Making curriculum optional allows users to benchmark both approaches
  on their specific problem before committing
- Default uniform training is easier to reason about and debug -
  important for research reproducibility
- Consistent with ADR-001: the simplest case stays simple,
  advanced users subclass or configure

## Consequences
- MultiFidelityTrainer accepts an optional MultiFidelityConfig
  in addition to TrainerConfig
- _schedule_fidelity() is called every epoch but is a no-op
  when curriculum=False
- Curriculum schedule is linear - equal number of epochs per
  fidelity level introduction. Non-linear schedules can be added
  later via MultiFidelityConfig without breaking existing code
- Users who want custom curriculum logic can override
  _schedule_fidelity() in a subclass