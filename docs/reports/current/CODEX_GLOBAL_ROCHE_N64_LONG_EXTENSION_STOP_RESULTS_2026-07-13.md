# Global Roche N64 Long-Extension Stop Results

**Date:** 2026-07-13, Asia/Shanghai
**Starting commit:** `6c4bcbb`

## Scope

The physical no-tide, no-wind Roche-loading state was resumed from the shared
N64 `1e-6 t_load` checkpoint. The declared next target was `1e-5 t_load`, but
the campaign was required to stop if adaptive cost or model validity
deteriorated.

No equation, physical closure, tolerance, residual scale, or 2% accepted-step
gate was changed.

## Clean `2e-6 t_load` Gate

The first bounded extension reached `2e-6 t_load` in 17 new accepted steps:

```text
new rejected attempts                         0
largest accepted residual                     4.44e-12
largest accepted Delta ln Sigma               0.01222
largest accepted Delta ln T                   0.00179
largest accepted relative Delta(H/R)          0.01437
inner accretion / supply                      0.22015
outer Roche overflow / supply                 0
disk-mass relative increase                   1.6104e-6
maximum H/R                                   0.141124
Jacobi availability at the Roche edge        -8.57298e16 erg/g
```

The edge remains closed and the controller is regular at this stage.

## Bounded `5e-6` Attempt

The next call was capped at 60 new accepted states. It did not reach the
requested target:

```text
final elapsed time                            3.9166016e-6 t_load
cumulative accepted steps                     93
new accepted steps                            60
new physical-change rejections                4
final proposed step                           5.27344e-9 t_load
largest accepted residual                     5.47e-12
disk-mass relative increase                   3.1540e-6
inner accretion / supply                      0.11644
outer Roche overflow / supply                 0
maximum H/R                                   0.141137
Jacobi availability at the Roche edge        -8.57294e16 erg/g
```

The controller reduced the accepted step from `8.4375e-8` through
`4.21875e-8`, `2.10938e-8`, and `1.05469e-8` to `5.27344e-9 t_load` as the
fixed 2% thickness-change gate was approached. All rejected candidates were
discarded and every retained nonlinear residual remains well below `1e-8`.

The inward fraction peaks near `0.22224` at `2.2344e-6 t_load`, then falls
monotonically to `0.11644` while the step size collapses. This is not a Roche
opening or a conservation failure.

## Limiting Region

A read-only trial from the final checkpoint locates the largest changes at
the inner edge:

| Change | Cell | Radius | Accepted-step value |
|---|---:|---:|---:|
| `abs(Delta ln Sigma)` | 0 | `5.3825 rg` | 0.00524 |
| `abs(Delta ln T)` | 2 | `6.1304 rg` | 0.00241 |
| relative `Delta(H/R)` | 2 | `6.1304 rg` | 0.01151 |

The outer source and Roche regions are not controlling. The first cell has
departed substantially from the fixed transonic reference:

```text
initial inner Mach number                     -0.6543
final inner Mach number                       -0.1483
max abs ln(Sigma/Sigma_ref)                   0.7817
incoming acoustic amplitude before projection 9.7591e8 cm/s
reference effective sound speed               7.0297e8 cm/s
amplitude / reference sound speed             1.388
```

The reference-characteristic boundary was certified only as a linear,
small-perturbation absorber. An incoming correction larger than the reference
sound speed is outside that claim. The checkpoint remains a useful numerical
boundary-breakdown witness, but it is not a physically certified long-time
state.

## Decision

Do not continue this checkpoint to `1e-5 t_load` by smaller steps. The
estimated cost would be dominated by the invalid fixed-reference inner
projection, and the resulting trajectory could not support a physical
loading, hot-state, or cycle conclusion.

The next work package is one causally outgoing inner plunge extension:

1. Continue the accepted stationary transonic solution inward from its
   `5.21024 rg` sonic node on the supersonic branch using the same potential,
   stress, vertical closure, angular ledger, and total-energy convention.
2. Place the time-dependent inner face only where all four radial
   characteristics leave the domain.
3. Rebuild the conservative finite-volume mapping and mechanical-energy
   reference on the extended grid; do not extrapolate primitives ad hoc.
4. Demonstrate exact reference preservation, no incoming characteristics,
   mass/angular/energy ledger closure, and N64/N96 mapping convergence.
5. Repeat the tiny-step, adaptive/restart, shared `1e-6`, and bounded
   long-extension gates once.

If a regular causally outgoing plunge cannot be constructed under the current
one-zone closure, stop the global 1D long-time campaign and escalate the inner
flow model. Do not replace this stop with a nonlinear reference reset, a
relaxed physical-change gate, or an artificial pressure target.

Distributed tide and wind remain blocked.
