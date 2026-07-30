# Causal Inner Scattering Geometry Feasibility

## WP10c9d6c7c2a1 — 2026-07-30

Analyzed base:

```text
73f902622834d13981d36e22aa21e13fefb9df8b
```

## Binding classification

```text
manufactured_interface_patch_geometry_selected_
energy_preflight_authorized
```

The package changes no operator and propagates no state. It preserves the
c2a geometry rejection and every c7 classification.

One operator-neutral **method-level** route is selected:

```text
manufactured_variable_coefficient_interface_patch
```

This selection authorizes only implementation and certification of the
manufactured state, invariant energy, characteristic projectors, and complete
energy ledger. It does not authorize uniform propagation, embedded
propagation, nonlinear evolution, production promotion, fixed-Q experiments,
or reduced slow-time evolution.

## Required geometry

The frozen scattering packet needs:

```text
43 parent cells of compact support
 3 parent cells of clearance at each end
49 parent cells per incident side
```

Both incidence directions remain mandatory.

## Option 1 — extended physical domain

A log-spaced domain with 49 cells on each side of the physical coupling
radius would extend over approximately

```text
1.727985 rg to 94.478791 rg.
```

Its geometry is valid and retains the coupling surface exactly, but the
independently certified smooth continuum background covers only

```text
1.8 rg to 12.777242 rg.
```

The proposed inner edge is also inside the certified excision surface, and
its outgoing-causality condition has not been established.

This route is rejected because it would require an unverified stationary
background extension and an uncertified new inner boundary. Spline
extrapolation is explicitly forbidden.

## Option 2 — characteristic injection on the existing domain

The minimum resolved pulse occupies 43 parent-cell crossing times. The
existing coarse side has only 16 cells:

```text
outer-boundary/interface round trip: 32 cell times
postinterface-surface round trip:      6 cell times
```

Both are shorter than the incident pulse. Binding incident and reflected
windows would overlap.

Fine-to-coarse injection through the excision surface is also forbidden
because excision has no incoming characteristics.

This route is rejected. No deconvolution or observed-peak window adjustment
is introduced.

## Option 3 — manufactured variable-coefficient patch

The selected patch has:

```text
98 parent-equivalent cells
49 cells on each side
interface face 49
left packet support  [3, 46]
right packet support [52, 95]
measurement faces    [6, 49, 92]
```

The interface radius is reproduced to relative defect

```text
1.62e-15
```

and log-grid spacing closes to

```text
3.05e-15.
```

The incident/reflected separation capacity is 86 parent-cell crossing times,
twice the 43-cell pulse extent. The geometry and travel-window capacity
therefore pass.

The physical core

```text
original parent faces 42–54
```

maps exactly to

```text
manufactured patch faces 43–55.
```

The coupling state, measures, principal matrices, and numerical stencil must
be identical inside that core.

## Frozen coefficient-extension construction

Outside the exact physical core, c2a2 must:

1. work in the implemented five-field primitive chart;
2. construct fourth-order endpoint Taylor jets;
3. blend each jet to a constant far state over 12 parent cells;
4. use the frozen C4 degree-nine smootherstep

\[
126s^5-420s^6+540s^7-315s^8+70s^9;
\]

5. recompute all storage, principal, lower-source, symmetrizer, and
   characteristic quantities from the extended primitive state.

Direct interpolation of the physical matrices is forbidden.

The construction must remain admissible and hyperbolic everywhere, retain
separated real characteristic clusters, and ledger all manufactured
background-gradient and responsive-height work. Uniform and embedded tests
must use the identical coefficient field.

Uniform subtraction may remove physical scattering created by the
manufactured coefficient extension. Residual subtraction, fitted
coefficients, or tuning against a propagated result are forbidden.

## Authorized next package

```text
WP10c9d6c7c2a2
manufactured scattering energy preflight
```

This remains a method package, not a propagation package.

It must:

- build and certify the C4 extended primitive state;
- verify primitive admissibility and real separated characteristic clusters;
- derive the complete symmetrized DAE energy identity and signs;
- implement normalization-invariant real-Schur or generalized-QZ projectors;
- implement incident, reflected, transmitted, leakage, dissipation, stored
  energy, and background-work ledgers;
- verify exact physical-core/interface parity;
- pass constant- and variable-coefficient manufactured balance tests;
- pass null-channel, sign, and amplitude-scaling tests;
- freeze exact packets, measurement surfaces, travel windows, and hashes.

Binding method gates include:

```text
interface-core parity defect       <= 1e-12
projector idempotence defect       <= 1e-12
energy-ledger relative defect      <= 1e-10
constant-state residual            <= 1e-12
signal/uncertainty ratio           >= 5
reference uncertainty/fine error  <= 0.10
```

Only if every c2a2 method gate passes may the separate uniform c2b
propagation package begin.

## Scientific limits

The selected patch tests the interface method while avoiding an unjustified
physical-domain extension. It does not certify:

- the extended physical radial background;
- physical scattering across the actual full embedded domain;
- a nonlinear embedded trajectory;
- or reduced slow-time evolution.

A successful manufactured c2b/c2c sequence would certify the coupling
method under the declared variable-coefficient class. A later physical bridge
would still be required before nonlinear physical promotion.

## Verification

```text
9 focused tests passed
```

The repository-wide suite reports:

```text
1032 passed
4 subtests passed
2 pre-existing policy failures
```

The policy failures are the legacy `PROSPECTIVE MANIFEST ONLY` provenance
vocabulary in two older canonical cases and the tracked-file ceiling
(`1170 >= 850`). No scientific or numerical regression test failed.

Canonical evidence is stored in:

```text
results/canonical/
causal_inner_scattering_geometry_wp10c9d6c7c2a1/
```
