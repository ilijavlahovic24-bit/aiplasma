AIPlasma: AI Framework for Nuclear Physics
## Status: Maintained

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
```
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
```
### Completed
- Base Classes,interfaces
- Data pipeline
- Physics-Informed Neural Network (PINN) layer
- Multi-Fidelity training strategy
- Bayesian uncertainty quantification
- CLI
- RLTrainer + PlasmaEnvironment + SyntheticPlasmaEnv
- HDF5DataSource
- SE3 Transformer model
- NuclearMaterialProblem
- RadiationProblem
- HasegawaWakatani.residual()

### Planned


## References and Documentation
1. Artificial intelligence-driven advances in nuclear technology: Exploring innovations, applications, and future prospects-Fatima Ezzahra Arhounia, Maged Ahmed Saleh Abdoa, 
Saad Ouakkasa, Mohamed Lhadi Bouhssaa, Aziz Boukhaira
2. Attention is all you need - Ashish Vaswani
3. Physics-Informed Neural Networks- Raissi et al. (2019) 
4. Multi-fidelity learning- Meng & Karniadakis (2020) 
5. MC Dropout / Bayesian approximation - Gal & Ghahramani (2016)
6. Hasegawa & Wakatani (1983) - Plasma turbulence model
7. Degrave et al. (2022) - RL for tokamak plasma control


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
