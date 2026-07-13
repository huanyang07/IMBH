# Global Roche Shared `1e-6 t_load` Results

**Date:** 2026-07-13, Asia/Shanghai
**Starting commit:** `4e2d595`

## Result

The accepted N64/N96/N128 checkpoints were resumed to the same
`1e-6 t_load` physical time:

| Cells | Cumulative accepted steps | New accepted steps | Cumulative retries | Inner accretion / supply | Overflow / supply | Disk mass change | Max `H/R` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 16 | 0 | 2 | 0.19181 | 0 | `8.1570e-7` | 0.141117 |
| 96 | 13 | 12 | 2 | 0.18799 | 0 | `8.1751e-7` | 0.141159 |
| 128 | 10 | 8 | 2 | 0.18625 | 0 | `8.1808e-7` | 0.141185 |

N64 was already at the target and was loaded only to evaluate the same final
diagnostics. N96 and N128 checkpoint after every accepted step. All accepted
nonlinear residuals are below `1.1e-11`; no rejected candidate replaces an
accepted state.

Relative to N128:

```text
N64 inner-flux difference / supply      -5.561e-3
N96 inner-flux difference / supply      -1.736e-3
N64 max(H/R) relative difference        -4.838e-4
N96 max(H/R) relative difference        -1.866e-4
N64 disk-mass-change difference         -2.376e-9
N96 disk-mass-change difference         -5.714e-10
```

The mass-change differences are below 0.3% of the accumulated change. The
inner-flux and thickness sequences converge monotonically over these meshes.

## Physical status

Every final edge remains closed:

```text
N64  B_J-Phi_s = -8.573001e16 erg/g
N96  B_J-Phi_s = -8.761776e16 erg/g
N128 B_J-Phi_s = -8.928342e16 erg/g
```

The disk therefore continues to accumulate most of the stream while the
inward fraction rises gradually. No hot transition is visible at this very
short time, and maximum `H/R` remains approximately 0.1412.

## Decision

The shared `1e-6 t_load` mesh gate passes for the declared preliminary targets:

```text
inner-flux mesh spread / supply < 1%
max(H/R) mesh spread             < 0.1%
all accepted residuals           < 1e-8
all outer mass fluxes            = 0
```

This still does not approach a loading, thermal, or viscous time. The next
bounded step is an N64 extension to `1e-5 t_load`, followed by N96/N128 only if
the single-mesh state remains regular and the adaptive cost remains tractable.
Tide and wind remain blocked.
