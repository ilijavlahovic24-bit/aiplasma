# ADR-017: Nuclear Domain Problems - Scope, Structure and Boundary Conditions

## Status
Accepted

## Context
NuclearMaterialProblem and RadiationProblem were v1 stubs that
raise NotImplementedError. ADR-002 defines a domain-scoped problem
hierarchy and PlasmaPhysicsProblem is its first domain.

Structure options:
A. Sibling classes of PlasmaPhysicsProblem, each inheriting
   PhysicsProblem, with small helper functions
B. A shared NuclearPhysicsProblem base for common units and
   validation

Scope options for RadiationProblem: neutron transport and decay,
plasma radiative losses, or both.

boundary_conditions() is abstract in PhysicsProblem, so every user
rewrites the same Dirichlet, Neumann and Robin residuals. Neutron
diffusion (vacuum and reflecting boundaries) and heat conduction
(fixed temperature, insulated and convective surfaces) need them
constantly.

## Decision
- Structure A. No shared nuclear base class.
- Both problems follow the PlasmaPhysicsProblem pattern: an
  equation attribute taken from REGISTRY, physics_params() kept
  separate from unit scales (ADR-005), pde_residual() delegating to
  the equation, and a domain validation method.
- NuclearMaterialProblem equations (registered in REGISTRY):
  - heat conduction with a volumetric source
  - Booth fission-gas diffusion (diffusion, decay, production, on
    an equivalent sphere)
  - point-defect rate theory (vacancies and interstitials as two
    coupled fields with production, recombination and sinks)
- RadiationProblem equations (registered in REGISTRY):
  - one-group time-dependent neutron diffusion
  - Bateman decay chains
  - plasma radiative losses: bremsstrahlung from an analytic
    scaling, impurity line radiation from a user-supplied cooling
    rate L_z(T_e); no atomic database is bundled
- Helpers are small free functions: arrhenius() and
  dpa_from_flux() for materials, half_life_to_lambda() and
  mean_free_path() for radiation.
- Equations that couple several fields follow the conventions of
  the Hasegawa-Wakatani equation: spatial coordinates first and
  time last, one residual column per equation in a documented
  order.
- Boundary conditions: helper functions dirichlet(), neumann()
  (normal derivative through autograd) and robin() build residuals
  from pred, coords and boundary_mask. They are available to every
  domain problem, plasma included, and are building blocks only:
  boundary_conditions() stays overridable, so users may use the
  helpers, combine them per boundary, or write their own residual.
  Nuclear and radiation problems ship a default
  boundary_conditions() that matches the data at boundary points
  (Dirichlet from fields).
- Units follow ADR-018.
- Verification against closed forms: steady heat conduction,
  Booth steady state, neutron-diffusion slab eigenmode, Bateman
  solutions, rate theory at its stationary limit.

## Rationale
- A follows the existing pattern and adds no hierarchy depth.
  Shared code is small enough to live in helpers.
- Every chosen equation has an analytic limit, so correctness can
  be tested rather than assumed.
- Line radiation depends on tabulated atomic data. Taking L_z as an
  input avoids pretending to ship data the framework does not have.
- Helpers remove the most repeated user code without taking
  control away.

## Consequences
- The v1 stubs and their NotImplementedError messages are removed.
- The registry grows by six equations; existing registry tests
  are unaffected.
- Rate theory is stiff (diffusivities and rate constants differ by
  orders of magnitude), so it depends on ADR-017 normalization.
- Bateman chains are time-only (one coordinate column) and have one
  output channel per nuclide.
- Neumann and Robin differentiate pred with respect to coords, so
  they need coords.requires_grad and a pointwise model (ADR-015).
- Different boundary types on different parts of the boundary
  require the user to combine masks; the helpers take a mask for
  this.
- Periodic boundaries are not covered. Use an empty boundary_mask
  and enforce periodicity in the model (v3 item).