# ADR 0017: Horizon-penetrating Valencia core

## Status

Accepted as the next one-domain causal architecture. The local flux and rank
prototype passes. Production evolution remains blocked pending the declared
migration gates.

## Context

WP10a repaired the gas+radiation acoustic derivative, but it also exposed a
more fundamental problem. The old low-rate Paczynski-Wiita continuation has

```text
v_phi/c = 0.848 at 4.5 rg
v_phi/c = 1.710 at 3.0 rg
v_phi/c = 357   at 2.0001 rg.
```

Its radial-only characteristic crossing is therefore not a physical
relativistic state. Special-relativistic sound speeds inserted into the
Newtonian PW momentum and energy equations cannot repair the full system.

## Decision

Use one vertically integrated Valencia-type conservative system on an
ingoing Kerr-Schild Schwarzschild background. The intended production domain
extends from an excision inside `2 rg` to the physical Hill/Roche edge. The
outer weak-field region must be recovered as the Newtonian limit of the same
system rather than retained as a separately spliced PW model.

This choice is based on four requirements:

1. the coordinates are regular at the event horizon;
2. radial and transverse velocities share one Lorentz factor;
3. stationary critical rank and time-dependent characteristics come from the
   same conservative flux;
4. an inner boundary inside the horizon requires no exterior physical data.

## Geometry

For `H=2 rg/R`, equatorial ingoing Kerr-Schild Schwarzschild geometry has

```text
alpha        = (1+H)^(-1/2)
beta^R       = H/(1+H)
gamma_RR     = 1+H
gamma_phiphi = R^2.
```

The coordinate radial light speeds are

```text
lambda_light,- = -1
lambda_light,+ = (1-H)/(1+H).
```

The outgoing light speed is zero at `2 rg` and negative inside it. Every
physical fluid characteristic is therefore directed through an inner
excision placed inside the horizon.

## Conservative Column Chart

The primitive chart is

```text
P = (Sigma, v_hat_R/c, v_hat_phi/c, T).
```

The gas+radiation column EOS supplies `e`, `Pi`, and

```text
h = 1 + (e + Pi/Sigma)/c^2.
```

With Eulerian Lorentz factor `W`, the mass-equivalent Valencia variables are

```text
D     = Sigma W
S_R   = Sigma h W^2 v_R
S_phi = Sigma h W^2 v_phi
tau   = Sigma h W^2 - Pi/c^2 - D.
```

Here the coordinate metric is included in the covariant momentum components.
The implementation stores momentum divided by `c` and energy divided by
`c^2`. The radial transport speed is

```text
q^R = alpha v^R - beta^R.
```

The perfect-fluid coordinate flux divided by `c` is

```text
F^R = (
    D q^R,
    S_R q^R + alpha Pi/c^2,
    S_phi q^R,
    tau q^R + alpha (Pi/c^2) v^R
).
```

The finite-volume variables and fluxes are densitized by the proper column
Jacobian `2 pi R sqrt(gamma_RR)`. Geometric, cooling, stream, tide, stress,
and wind terms must enter named source or flux ledgers.

## Exact Flux-primary Count

For `N` cells, keep cell conserved variables, local primitives, and all face
fluxes explicit:

```text
cell conserved variables             4 N
cell primitive variables             4 N
face fluxes                       4(N+1)
total unknowns                    12 N + 4
```

The rows are

```text
backward-Euler conservation           4 N
primitive/conserved map               4 N
interior face fluxes              4(N-1)
inner one-sided flux                     4
outer provider flux                      4
total rows                         12 N + 4.
```

The four inner flux rows define the one-sided constitutive flux. They are not
exterior boundary conditions. The number of physical inner boundary rows is
zero.

If primitives and fluxes are eliminated locally, the differential system has
`4N` unknowns and `4N` conservation rows.

## Stationary Critical Contract

The stationary state is a root of the same finite-volume operator used by
time evolution. No separately defined slim-disk sonic row is inserted.

For a continuous diagnostic, the stationary primitive matrix loses one rank
when one Valencia characteristic is zero. Regularity is the left-null
compatibility condition of that same matrix. The WP10b ideal-gas prototype
finds exactly rank three at a one-acoustic-mode critical point.

## Rejected Alternatives

1. Do not insert the WP10a sound speed into the old PW flux.
2. Do not use Schwarzschild coordinates, whose lapse degenerates at the
   horizon.
3. Do not add another relativistic-inner/Newtonian-outer splice.
4. Do not map the old superluminal PW plunge into the new variables.
5. Do not resume tide, wind, or long loading before the causal baseline passes.

## Production Gates

The architecture is not production ready until it has:

1. gas+radiation column primitive recovery;
2. Kerr-Schild geometric sources and independent conservation audits;
3. a stress and torque-work closure whose full characteristic spectrum
   remains causal;
4. a migrated stream and Hill/Roche boundary ledger;
5. cold weak-field recovery of the certified Newtonian benchmarks;
6. a low-throughput stationary root of the same finite-volume equations;
7. a fixed inner face inside `2 rg` with zero incoming modes;
8. conservative N64/N96 mapping and one accepted tiny implicit step.

## References

- Banyuls et al., [Numerical 3+1 General Relativistic Hydrodynamics:
  A Local Characteristic Approach](https://doi.org/10.1086/303604).
- Montero, Baumgarte, and Mueller,
  [General relativistic hydrodynamics in curvilinear
  coordinates](https://arxiv.org/abs/1309.7808).
- Takahashi, [Radiation Hydrodynamics in Kerr Spacetime: Equations without
  Coordinate Singularity at the Event
  Horizon](https://arxiv.org/abs/0710.3512).
