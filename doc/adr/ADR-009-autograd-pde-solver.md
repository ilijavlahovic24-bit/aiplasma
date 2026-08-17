# ADR-009: AutogradPDESolver as Primary PDE Solver

## Status
Accepted

## Context
AIPlasma needs a PDE solver that can compute spatial and temporal
derivatives for PINN training. Three approaches were considered:

1. Autograd - use PyTorch automatic differentiation through
   the model's computational graph
2. Finite Difference - approximate derivatives numerically
   using u(x+h) - u(x-h) / 2h
3. Spectral Methods - represent solution as sum of basis
   functions (Fourier, Chebyshev) and differentiate analytically

## Decision
Use PyTorch autograd as the primary derivative computation method
in AutogradPDESolver. Finite difference and spectral methods
are not implemented in v1.

## Rationale
- Autograd is exact (up to floating point precision) - no
  discretization error unlike finite difference
- No spatial grid required - PINN operates on scattered
  collocation points, which autograd handles naturally
- Zero additional implementation cost - PyTorch autograd
  is already available through the existing model graph
- Finite difference requires careful choice of h (step size)
  and struggles with irregular point distributions
- Spectral methods require structured grids and are problem-specific -
  incompatible with the framework's goal of being domain-agnostic
- Consistent with standard PINN literature (Raissi et al. 2019)
  which uses autograd for all derivative computations

## Consequences
- coords must have requires_grad=True when solve_with_grad()
  is called - AutogradPDESolver sets this internally
- Autograd through deep networks can be slow for large N -
  acceptable for research use cases in v1
- If performance becomes a bottleneck, a FiniteDiffPDESolver
  subclass can be added without modifying AutogradPDESolver
- Second and higher-order derivatives require multiple autograd
  passes - d2u_dx2 costs two passes, d3u_dx3 costs three