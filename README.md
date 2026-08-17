AIPlasma: AI Framework for Nuclear Physics
## Status: Active Development

## Core Framework Philosophy
Instead of solving one specific physics problem, i'm building the infrastructure to solve many. This is analogous to PyTorch Lightning for deep learning or ROS for robotics.


## What AIPlasma Does
AIPlasma provides a framework for applying machine learning 
to physics simulations, with built-in support for:
- Physics-Informed Neural Networks (PINNs)
- Uncertainty quantification via Bayesian methods  
- Multi-fidelity training across different simulation resolutions
- Ready-to-use examples for plasma physics problems

## Project Structure
AIPlasma/
├── analysis/       # Validation, visualization
├── config/         # Hyperparameters
├── core/           # Base classes, interfaces
├── data/           # Data loaders, transforms
├── models/         # Model zoo
├── solvers/        # Integration with physics codes
├── tests/          # Validation, visualization
├── trainers/       # Training strategies
├── docs/       	# Documentation, references
└── examples/       # Demonstrations with Real-Life Examples

### Completed
- Base Classes,interfaces
- Data pipeline
- Physics-Informed Neural Network (PINN) layer

### In Progress
- Multi-Fidelity training strategy
- Bayesian uncertainty quantification
- 
### Planned
- Chaos testing for model robustness

## References and Documentation
Artificial intelligence-driven advances in nuclear technology: Exploring innovations, applications, and future prospects


## Acknowledgements

AIPlasma was designed and developed by Ilija Vlahović as part of
a personal research initiative during the final year of Computer
Science and Engineering at ETF Belgrade.

**Architectural design and technical decisions:** All framework
architecture, API design, and physics-informed ML methodology
were conceived and directed by the author. Every major decision
is documented in the Architecture Decision Records (ADRs)
in `doc/adr/`.

**AI assistance:** Claude (Anthropic) was used as a development
assistant during implementation. Specifically:
- Code generation and boilerplate implementation based on
  author-specified designs
- Formatting and structuring of ADR documentation after
  author decisions were made
- Technical discussion and option analysis during design phases

All generated code was reviewed, tested, and validated by the
author before integration.

**Note for scientific use:** Users building on AIPlasma for
research purposes should independently verify physical
formulations and numerical implementations against established
literature before publication.
