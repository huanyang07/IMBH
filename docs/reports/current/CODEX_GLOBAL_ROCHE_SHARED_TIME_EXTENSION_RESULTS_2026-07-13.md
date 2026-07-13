# Global Roche Shared-Time and Extension Results

**Date:** 2026-07-13, Asia/Shanghai
**Starting commit:** `c78ca60`

## Shared physical time

Fresh adaptive runs were advanced to the same `1e-7 t_load` target:

| Cells | Accepted steps | Rejected attempts | Inner accretion / supply | Overflow / supply | Disk mass change | Max `H/R` |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 0 | 0.17171 | 0 | `8.2829e-8` | 0.141115 |
| 96 | 1 | 0 | 0.17349 | 0 | `8.2651e-8` | 0.141158 |
| 128 | 2 | 1 | 0.17431 | 0 | `8.2685e-8` | 0.141184 |

Relative to N128:

```text
N64 inner-flux difference / supply     2.603e-3
N96 inner-flux difference / supply     8.205e-4
N64 max(H/R) relative difference      -4.948e-4
N96 max(H/R) relative difference      -1.898e-4
```

The N128 full `1e-7 t_load` attempt reached its function-evaluation limit with
residual `5.07e-6`. The adaptive controller rejected it and completed two
`5e-8 t_load` steps with accepted residuals below `9.6e-13`. This is the
intended mesh-aware behavior, not a relaxed gate.

Every mesh remains energetically below the Roche saddle and exports zero mass.
The mesh variation of the Jacobi deficit is the same few-percent edge-state
variation already recorded by the static boundary preflight; it does not
change the closed classification.

## N64 restart extension

The accepted N64 checkpoint at `5e-7 t_load` was loaded and continued to
`1e-6 t_load`:

```text
target reached                       yes
cumulative accepted steps            16
new accepted steps                   8
cumulative rejected attempts         2
new rejected attempts                0
relative disk-mass increase          8.1570e-7
inner accretion / supply              0.19181
outer overflow / supply               0
max(H/R)                              0.141117
B_J-Phi_s at final edge              -8.573001e16 erg/g
```

The increasing inner fraction reduces the accumulated fraction modestly, but
the physical conclusion is unchanged: the disk is loading with a closed Roche
edge and remains geometrically moderate over this short interval.

## Decision

The `1e-7 t_load` shared-time mesh gate passes, and N64 continuation to
`1e-6 t_load` passes. The next bounded campaign should advance N96 and N128 to
the same `1e-6 t_load` checkpoint before extending N64 another decade. Tide
and wind remain blocked; the elapsed duration is still far below loading,
thermal, and viscous times.
