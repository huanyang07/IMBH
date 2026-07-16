# Global Exact-Common-Time WP3 Results

**Date:** 2026-07-14
**Scope:** Regenerated N64/N96/N128 supersonic-plunge/Roche-edge evolution
from the canonical initial state at one exactly shared physical time. No
interpolation between evolved states was used.

## Result

All three meshes reached the identical timestamp

```text
t = 0.15200886034168773 s
  = 1.0e-7 of the shared N128 reference loading time.
```

The mesh loading times remain separately reported. Their target loading
fractions differ slightly because the conserved initial mass is mesh
dependent; the physical time is exactly equal as a floating-point value.

The accepted states and their reference/mechanical-offset payloads were saved
as immutable, checksummed milestones under
`outputs/checkpoints/milestones/global_exact_common_time/`. The combined
machine-readable report is
`outputs/tables/global_exact_common_time_snapshot.json`.

## Global comparison

| Quantity | N64 | N96 | N128 |
|---|---:|---:|---:|
| Accepted steps | 15 | 8 | 4 |
| Rejected attempts | 4 | 3 | 2 |
| Inner mass flux / supply | -0.172077 | -0.172078 | -0.170993 |
| Maximum H/R | 0.141105 | 0.141164 | 0.141183 |
| Disk mass relative change | 8.3619e-8 | 8.3165e-8 | 8.3064e-8 |
| Sonic radius (rg) | 5.23827 | 5.22171 | 5.21480 |
| Cells inside sonic radius | 2 | 3 | 4 |
| Minimum Lv/dRcell | 1.487 | 2.212 | 3.097 |
| Minimum Lv/H | 2.591 | 3.010 | 3.703 |
| Normalized Roche margin | -0.08479 | -0.08641 | -0.08827 |

Relative to N128, N96 differs by:

```text
inner mass flux / supply       -1.086e-3
inner angular flux              0.635%
inner total-energy flux         0.716%
maximum H/R                    -0.0131%
disk mass relative change       1.01e-10
sonic radius                    0.00691 rg
```

N64 remains below one percent in all three conserved inner-flux comparisons,
but its total-energy difference is close to that bound at `0.981%`.

All outer mass, angular-momentum, and energy fluxes are exactly zero. The
Roche channel is safely on the closed active set, with zero active-set
residual and no nozzle residual to evaluate. Full-history accepted storage
ledger defects are below `3.5e-16` on every mesh.

## Fixed-radius plunge audit

The exact-time comparison exposes slower pointwise convergence close to the
inner face. N96 minus N128 gives:

| Radius | Delta Mach | Delta ln Sigma | Delta ln T | Delta H/R | Delta mass flux / supply |
|---:|---:|---:|---:|---:|---:|
| 4.65 rg | -0.3810 | 0.01940 | -0.00816 | -5.44e-4 | -2.26e-3 |
| 4.75 rg | -0.2132 | 0.01789 | -0.00473 | -4.83e-4 | -3.13e-3 |
| 5.00 rg | 0.0270 | 0.00520 | 0.00404 | 2.10e-4 | -2.74e-3 |

The fixed-radius Mach values are therefore not yet pointwise mesh converged at
`4.65-4.75 rg`, even though the global conserved fluxes, mass loading, and
maximum thickness pass the preliminary mesh gate. The sonic region becomes
better resolved monotonically with mesh refinement.

## Interpretation

This snapshot supports three scoped conclusions:

1. The earlier shared-time comparison was not an artifact of slightly
   different physical timestamps.
2. The causally outgoing plunge and closed Roche edge remain conservative at
   exact common time.
3. Global evolution diagnostics converge faster than local plunge primitives;
   local sonic-gradient claims require the separate WP4 audit.

It does not establish a long-time no-tide state, a hot branch, a front, or a
limit cycle. The duration remains only `1e-7` of one loading time.

## Next work package

Proceed to WP4, the bounded stationary sonic-gradient/source-resolution audit.
The audit must explain the local `4.65-4.75 rg` convergence pattern before the
N64 source-on/source-off and long-duration controls are interpreted.
