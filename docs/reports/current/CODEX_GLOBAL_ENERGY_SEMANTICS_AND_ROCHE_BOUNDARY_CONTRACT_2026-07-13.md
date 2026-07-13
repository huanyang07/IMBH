# Global Energy Semantics and Roche-Boundary Contract

**Date:** 2026-07-13, Asia/Shanghai
**Starting commit:** `1cf430f`
**Scope:** WP0 implementation and WP1 decision only; no long evolution, tide,
wind, or nozzle production solve.

## Result

The cross-consistency issue between the fixed mechanical quadrature offset and
the physical face fluxes is corrected.

The implementation now distinguishes:

```text
stored cell-average total energy
physical center total energy
physical face Bernoulli energy
fixed cell mechanical quadrature offset
stored state used by numerical dissipation
```

The offset remains in conservative cell storage and primitive recovery. It is
removed from physical Bernoulli energy in the Rusanov physical flux,
conserved-donor outer face, and both sides of the inner characteristic
correction. The Rusanov dissipative jump continues to use the stored
conservative energy.

This is not a new physical energy source or sink. It prevents a mesh
quadrature correction from being exported as boundary energy.

## Characteristic Tests

A nonzero-offset amplitude sequence verifies that the inner characteristic
energy correction tends continuously to zero. The exact reference state still
returns the unmodified flux.

The numerical eigensystem audit differentiates the actual vertically
integrated physical flux in scaled conserved variables. In the manufactured
radiation-pressure test state:

```text
maximum analytic eigenvalue defect / c_eff   2.40e-5
finite-difference refinement defect          5.45e-7
incoming acoustic left-vector alignment      1.0
maximum biorthogonality defect                2.19e-11
maximum eigenpair residual                    3.51e-21
```

These results qualify the analytic acoustic projection for its current
small-perturbation reference-state role. They do not yet certify a permanent
inner boundary for nonlinear long evolution.

## Restart Contract

The fixed mechanical reference now has a dedicated restart artifact containing:

```text
schema version
full grid edges
full specific-offset array
generating-state SHA-256
offset-and-grid SHA-256
JSON provenance
```

Loading rejects altered checksums, incompatible meshes, incompatible
generating states, missing fields, and pickle/object data. The offset is never
silently regenerated.

## Outer-Boundary Review

The checked-in Layer-1 Hill-flow package contains capture, angular-momentum,
geometry, and stress diagnostics. It does not contain an ambient pressure,
temperature, entropy, or Bernoulli/Jacobi invariant at `335 rg`.

Therefore the selected physical path is:

```text
one adiabatic Hill/Roche overflow side channel
ending in regular sonic passage at a real L1/L2 saddle.
```

Only a boundary-provider protocol and the nozzle provider should be built.
There should be no placeholder Layer-1 exterior provider until genuine
exterior thermodynamic data exist.

The first model may expose one effective throat-area or azimuthal-filling
parameter, but it must report sensitivity and stop if that parameter controls
the answer. It must not be fitted to recover the old hybrid overflow rate.

## Numerical Verification

Targeted global-evolution tests after the WP0 implementation:

```text
47 passed
```

Full repository verification:

```text
340 passed, 4 subtests passed
repository hygiene passed for 512 tracked files
```

The production simulations and canonical JSON were not regenerated in this
work package because the outer physical boundary is still absent.

## Status

```text
WP2 column work:                         accepted in current one-zone scope
WP3 inner absorber:                      accepted reference-state preflight
WP4 mechanical conditioning:            accepted mapped-state conditioning
WP0 joint energy semantics/restart:      passed targeted tests
physical outer characteristic contract: selected, not implemented
long no-distributed-tide evolution:      blocked on nozzle certification
distributed tide and wind:               deferred
```

## Next Work Package

Implement the standalone adiabatic Hill/Roche nozzle before coupling it to the
disk. Required outputs are one shared overflow flux state:

```text
F_M, F_PR, F_J, F_E
sonic residual
Jacobi residual
frame and energy-zero metadata
throat geometry and filling-factor provenance
```

The standalone solver must pass manufactured nozzle convergence, sonic
regularity, rotating/inertial energy conversion, and throat-area sensitivity
before it may replace the current donor outer face.
