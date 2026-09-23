# ADR-013: AIPlasma CLI as Installable Entry Point

## Status
Accepted

## Context
AIPlasma needs a way for users to scaffold new problems, models,
trainers, and examples without writing boilerplate from scratch.
Two options were considered:

1. Script in scripts/ directory — run as python scripts/aiplasma_cli.py
2. Installable CLI entry point — run as aiplasma after pip install -e .

## Decision
Implement AIPlasma CLI as an installable entry point (Option B)
using Python's click library, registered via pyproject.toml.

Commands:
    aiplasma init                              - scaffolds full project structure
    aiplasma new problem <Name> --equation <>  - generates PhysicsProblem subclass
    aiplasma new model <Name> --type <>        - generates PhysicsModel subclass
    aiplasma new trainer <Name> --type <>      - generates Trainer subclass
    aiplasma new example <Name> --equation <>  - generates full example folder
    aiplasma list equations                    - lists all registered equations
    aiplasma list models                       - lists available model types

Output always goes to current working directory.
Example command generates full folder with README, train, evaluate,
visualize scripts and config.yaml.

## Rationale
- Installable entry point is consistent with the goal of AIPlasma
  being used in scientific institutions - users install once and
  use the CLI naturally without knowing the project structure
- Angular/React-style scaffolding (ng g c, ng g s) is familiar
  to developers and reduces friction for new users
- click library provides clean command composition, help text,
  and error handling with minimal boilerplate
- pyproject.toml entry_points is the modern Python standard for
  CLI tools - consistent with making AIPlasma pip installable
- Output to current directory keeps the interface simple -
  no --out flag needed for v1

## Consequences
- click must be added to project dependencies
- CLI code lives in aiplasma/cli/ package, not scripts/
- pyproject.toml must define [project.scripts] entry point
- aiplasma init creates the full directory structure -
  useful for new users starting from scratch
- Templates are string templates in code, not separate files -
  keeps the package self-contained without data file dependencies