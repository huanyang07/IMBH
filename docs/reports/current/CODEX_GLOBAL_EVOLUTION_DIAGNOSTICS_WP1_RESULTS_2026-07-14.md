# Global Evolution Diagnostics WP1 Results

**Date:** 2026-07-14, Asia/Shanghai
**Starting commit:** `35dbd2f`
**Scope:** diagnostics, provenance, and immutable checkpoints only

## Decision

WP1 passes.

No conservation equation, boundary closure, nonlinear residual tolerance,
ledger tolerance, or 2% adaptive physical-change gate was changed. Physical
distributed-tide continuation and wind remain blocked.

## Implemented Diagnostics

The adaptive controller now identifies the physical-change gate with the
largest fraction of its allowed limit. Every converged nonlinear candidate
records:

```text
variable and change metric
cell index and physical radius
old and candidate values
fraction of the unchanged limit
local radial Mach number
four radial characteristic speeds
whether all characteristics point inward
```

Every campaign milestone now reports the complete outward-oriented inner and
outer conserved-flux triplets:

\[
F_M,\qquad F_J,\qquad F_E.
\]

Mesh-comparable diagnostics are evaluated at exactly:

```text
4.65 rg
4.75 rg
5.00 rg
```

Positive primitives use logarithmic interpolation in logarithmic radius;
signed primitives and face fluxes use linear interpolation in logarithmic
radius. Boundary-adjacent queries use the corresponding two-point
extrapolation rather than silently clamping to the nearest cell center.

The sonic audit locates the innermost \(|\mathcal M_R|=1\) crossing and counts
the cells between that crossing and the inner face. Its velocity-gradient
length is

\[
L_v=\left|\frac{d\ln|v_R|}{dR}\right|^{-1}.
\]

The reported minima of \(L_v/\Delta R\) and \(L_v/H\) are restricted to the
inner face through the first sonic transition plus two neighboring cells.
This prevents a source- or outer-edge gradient from being mislabeled as
plunge resolution.

## Roche Classification

The raw available specific energy is now normalized by the physical
edge-to-saddle Hill/Roche barrier:

\[
\widehat{\Delta B}
=
\frac{B_J-\Phi_{\rm eff}(R_L)}
{\Phi_{\rm eff}(R_{L})-\Phi_{\rm eff}(R_{\rm edge})}.
\]

Closed channels report no nozzle residual. Instead they report an active-set
residual combining positive energetic violation and any nonzero applied mass
flux. Choked channels additionally report the actual sonic nozzle residual.

## Time And Checkpoint Semantics

Reports now distinguish:

```text
physical elapsed time in seconds
mesh-specific initial-mass loading time
one shared campaign reference loading time
fractions using both loading-time definitions
```

The current shared reference is the conservatively mapped `N=128` initial
mass divided by the fixed stream supply. It is a common campaign convention,
not a claim that the finite-resolution initial masses are identical.

Rolling restart files remain available for continuation. A separate immutable
milestone writer creates filenames containing:

```text
case
mesh
physical time
Git SHA
conservative-state SHA
```

Every milestone manifest records the checkpoint, state, reference-state, and
mechanical-offset SHA-256 values together with controller counters, fluxes,
sonic/Roche diagnostics, Git provenance, and the available accepted/rejected
attempt history. Rewriting an existing milestone with different state or
provenance is rejected.

## Production-State Audit

The persisted `N=96`, `1.001e-6` mesh-loading-time state was loaded and
diagnosed without advancing or rewriting the rolling trajectory.

```text
physical elapsed time                         1.5215356193 s
mesh loading time                             1.5200156037e6 s
shared N128 reference loading time            1.5200886034e6 s
mesh/reference loading-time difference       -4.8023e-5

inner mass flux                              -1.5469893e22 g/s
inner angular-momentum flux                  -2.5410292e42 cgs
inner total-energy flux                       7.9529134e41 erg/s
outer mass/angular/energy flux                exactly zero

Mach at 4.65 rg                              -16.3636
Mach at 4.75 rg                               -7.6910
Mach at 5.00 rg                               -2.1588
sonic radius                                  5.23284 rg
cells inside sonic crossing                   3
minimum plunge Lv / cell width                1.69796
minimum plunge Lv / H                         2.02011

raw Roche available energy                   -8.74259e16 erg/g
Roche barrier                                 1.01181e18 erg/g
normalized Roche margin                      -0.0864058
Roche state                                   closed
Roche active-set residual                     0
```

The first-cell Mach number is retained only as a boundary audit. Mesh
certification will use the fixed-radius and emergent-sonic diagnostics above.

## Verification

```text
targeted diagnostics/adaptive/global tests    59 passed
full repository test suite                    367 passed
documented subtests                           4 passed
production N96 zero-step diagnostic           passed
immutable save/load/checksum audit             passed
```

## Locked Next Step

WP2 is one bounded solver-efficiency certification:

1. Profile the same accepted `N=64` and `N=128` candidate steps.
2. Test deterministic gate-aware termination under unchanged residual and
   ledger gates.
3. Attempt selected local/block derivative columns only if profiling identifies
   a dominant local cost.
4. Keep sparse-forward as the reference and do not revive coloring.
5. Adopt a candidate only if it preserves the accept/reject decision,
   controller cell, next timestep, ledgers, and state within `0.1` of measured
   temporal error while giving a material speedup.

If the bounded candidates fail, continue the physics controls with the
certified sparse-forward backend. No third optimization architecture is
authorized.
