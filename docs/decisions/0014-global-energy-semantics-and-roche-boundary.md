# ADR 0014: Separate Cell-Average Energy from Physical Face Energy

## Status

Accepted for the global signed conservative preflight. The standalone
Hill/Roche nozzle provider is implemented and passes its manufactured gates;
the disk edge coupling passes its closed/choked preflight. The former
fixed-reference inner characteristic projection is retained as a regression
control, while the selected production preflight boundary is the causally
outgoing `4.5 rg` plunge face.

## Context

The conservative remap stores the true annular total-energy integral. A fixed
cell mechanical quadrature offset reconciles that integral with primitive
variables evaluated at the cell center:

```text
epsilon_stored = epsilon_center + delta_e_mech.
```

The offset is a finite-volume representation term. It is not energy available
to gas crossing a physical radial face. Before this decision, smooth face
reconstruction omitted the offset, while the Rusanov physical flux, conserved
outer donor, and current side of the inner characteristic correction included
it. The projected inner state omitted it. That mixed convention produced a
finite energy-flux jump as a nonzero incoming characteristic amplitude tended
to zero.

The current outer edge at `335 rg` is deeply subsonic and has one incoming
acoustic characteristic. It lies at about `0.4485 R_H`, not at an `L1/L2`
saddle. The checked-in Layer-1 package supplies capture, geometry, and stress
diagnostics but no ambient pressure, entropy, temperature, or Bernoulli/Jacobi
invariant at that edge.

## Energy Decision

Use four explicitly different quantities:

1. **Stored cell-average energy** includes `delta_e_mech` and remains the
   fourth conservative state.
2. **Physical center energy** is stored specific energy minus
   `delta_e_mech`.
3. **Physical face Bernoulli energy** is reconstructed from physical
   primitives and never exports `delta_e_mech`.
4. **Numerical dissipative energy state** remains the stored conservative
   field so the finite-volume update conserves exactly what is stored.

Consequences:

- smooth face reconstruction is unchanged;
- the physical part of the Rusanov energy flux removes the offset;
- Rusanov diffusion continues to act on stored energy;
- the conserved-donor outer face exports physical Bernoulli energy;
- projected and current characteristic states use the same convention;
- a projected stored state adds the same offset before its physical flux
  removes it;
- the local physical-flux eigensystem differentiates physical center energy,
  not the mesh quadrature offset.

The fixed offset is restart state. Its checkpoint stores the full array, grid
edges, schema version, generating-state SHA-256, offset SHA-256, and JSON
provenance. A missing, altered, or mesh-incompatible reference is rejected.

## Characteristic Decision

The analytic gas-radiation acoustic projection remains accepted for the
reference-state preflight only after comparison with the numerical Jacobian of
the actual vertically integrated physical flux. The audit is performed in
scaled conserved variables and reports eigenvalues, incoming-left-vector
alignment, finite-difference refinement, biorthogonality, and eigenpair
residuals.

The reference-state projection is certified only for the small-perturbation
preflight. Long evolution uses the same-equation supersonic continuation to a
`4.5 rg` face where every radial characteristic leaves the domain. That face
applies no incoming invariant or projection. N64/N96/N128 pass the shared
`1.001e-6 t_load` gate. The N64 continuation remains closed and causally
outgoing through `1.430993e-6`; the longer duration gate is blocked by
nonlinear/Jacobian cost rather than the inner or Roche boundary contract.

## Outer-Boundary Decision

Path A, a Layer-1 exterior invariant at `335 rg`, is unavailable in the
current repository. Select Path B:

```text
disk outer annulus -> adiabatic Hill/Roche overflow side channel
                   -> regular sonic throat at a real L1/L2 saddle.
```

The global solver will consume a boundary-provider protocol, but only the
Hill/Roche nozzle provider will be implemented now. A placeholder Layer-1
provider is prohibited until Layer 1 exports a genuine ambient state in the
same frame, equation of state, location, and energy-zero convention.

The first nozzle contract must return one shared state containing outward
mass, radial-momentum, angular-momentum, and inertial-energy fluxes plus sonic
and Jacobi residuals. It must use:

- the disk-edge entropy/contact state;
- the disk-edge angular momentum;
- the binary pattern speed and Hill/Roche effective potential;
- an explicitly declared effective throat area or azimuthal filling factor;
- adiabatic transport in the first model;
- regular sonic passage at the selected saddle;
- the rotating-to-inertial identity pairing energy and angular momentum.

No vacuum state, copied mesh pressure, tuned exterior temperature, wind,
distributed tide, or arbitrary nozzle heating belongs in this work package.

## Gates

Energy and inner-characteristic gates:

```text
nonzero-offset energy correction tends continuously to zero
analytic/numerical acoustic sign count agrees
physical eigensystem finite-difference audit converges
restart reproduces the offset bitwise
all four discrete ledgers remain closed
```

Outer-nozzle gates:

```text
exactly one incoming acoustic relation supplied
sonic and Jacobi residuals pass
mass/J/E/radial fluxes come from one boundary state
N96/N128 boundary flux convergence passes the declared tolerance
no unconfigured inward mass
```

Stop if the result is controlled mainly by an unconstrained throat area. That
would identify missing multidimensional geometry, not justify fitting the
area to the old open-overflow solution.

The first mapped `N=64,96` evaluations are below the adiabatic saddle threshold
by `8.58e16` and `8.76e16 erg/g`. The filling factor is therefore irrelevant
for those states: the channel is closed. Disk coupling must support a
continuous closed-to-choked transition rather than force the old donor
overflow through the nozzle.

## Sequencing

1. Complete and test the energy/reference correction.
2. Implement and certify the standalone adiabatic nozzle. **Completed.**
3. Couple the nozzle to the outer annulus. **Next.**
4. Run no-distributed-tide evolution.
5. Add one physical distributed tide with paired pattern-speed power.
6. Add wind last.
