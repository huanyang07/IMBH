# Global Supersonic Plunge Results

**Date:** 2026-07-13, Asia/Shanghai
**Starting commit:** `b8c7983`

## Scope

This work package replaces the fixed-reference subsonic inner absorber with a
causally outgoing inner face. The accepted stationary transonic solution is
continued inward from its `5.210237 rg` sonic node with the same production
potential, stress, vertical closure, angular ledger, and local energy
equation. No ballistic, adiabatic, free-fall, or fitted plunge law is added.

The finite-volume inner edge is fixed at `4.5 rg`. At that face the physical
one-sided flux of the first cell is used without an exterior state or
characteristic projection. This is admissible only while all radial
characteristics point toward decreasing radius and leave the modeled domain.

No equation, nonlinear tolerance, ledger tolerance, or 2% adaptive
physical-change gate was relaxed.

## Stationary Plunge

The regular sonic derivative branch is selected by proximity to the resolved
first outer transonic interval. Starting one logarithmic offset inside the
sonic point, the production local ODE integrates smoothly to `4.5 rg`:

```text
sonic radius                                  5.210237256 rg
inner radius                                  4.5 rg
inner radial velocity                         -0.0751214 c
inner Euler acoustic Mach number              -9.45218
incoming radial characteristics               0
sonic-gradient mismatch                       0.189813
maximum scaled local differential residual    6.66e-16
```

The gradient mismatch measures the difference between the selected regular
sonic derivative and the finite first outer interval. It is retained as a
resolution diagnostic; the continued branch itself closes the checked local
equations at roundoff.

## Conservative Mapping And Tiny Steps

The plunge profile is prepended to the accepted inner transonic profile before
the existing conservative annular quadrature is applied. The sonic point is
included once. The global mesh now begins at the physical `4.5 rg` face.

The 32/64-point mapping comparison gives:

| Cells | Maximum conserved-state difference | Mechanical-offset difference |
|---:|---:|---:|
| 64 | `5.29e-4` | `9.76e-6` |
| 96 | `2.07e-4` | `1.55e-5` |
| 128 | `1.36e-4` | `9.28e-6` |

Every mapped state has positive internal energy. The first-cell characteristic
audit and the one-full-step versus two-half-step comparison are:

| Cells | First center | Initial Mach | Incoming | Full/half state difference |
|---:|---:|---:|---:|---:|
| 64 | `4.6541 rg` | `-5.1420` | 0 | `1.85e-6` |
| 96 | `4.6022 rg` | `-6.3387` | 0 | `2.40e-6` |
| 128 | `4.5764 rg` | `-7.0207` | 0 | `2.63e-6` |

All full and half steps are accepted. Their largest normalized residual is
`2.85e-12`; the largest storage-scaled mass/angular/energy ledger defect is
`1.93e-16`. The outer Roche flux remains exactly zero, and the inner
characteristic projection is absent.

## Shared `1e-7 t_load` Gate

A fresh adaptive run with checksum-verified restart after every accepted step
reaches the common target on all meshes:

| Cells | Accepted | Recovered retries | Inner flux / supply | Inner Mach | `max(H/R)` |
|---:|---:|---:|---:|---:|---:|
| 64 | 15 | 4 | `-0.172074` | `-6.797` | `0.141105` |
| 96 | 8 | 3 | `-0.172078` | `-7.391` | `0.141164` |
| 128 | 4 | 2 | `-0.170993` | `-7.652` | `0.141183` |

All retries are converged nonlinear roots rejected only by the declared
physical-change controller. Accepted residuals are at most `1.03e-11`; the
production step accepts only storage-scaled ledger defects below `1e-8`.
There are zero incoming characteristics and zero outer Roche flux on every
mesh.

Relative to N128, N96 differs by `0.1086%` of the stream supply in inner flux
and by `0.0131%` in maximum `H/R`. The shared-time preliminary mesh gate
passes.

## Refined Duration And Target Landing

The accepted N128 checkpoint was resumed without rerunning N64 or N96. The
certified sparse-forward Jacobian becomes expensive late in the sequence. One
step stopped at the original 300-evaluation budget with a residual of
`1.047e-8`, just above the unchanged `1e-8` gate. Raising only the iteration
budget to 600 allowed the same residual gate to be met. No physical-change,
ledger, or characteristic threshold was changed.

The final landing exposed a controller defect. A machine-epsilon positive
remainder was clipped upward to the ordinary `1e-9 t_load` minimum step, so the
persisted N128 checkpoint landed at `1.001e-6 t_load`. The runner now:

1. snaps roundoff-only remainders to the requested target without evolving;
2. permits an exact final step below the ordinary controller minimum when the
   remainder is physically finite; and
3. reports the snap explicitly.

The already accepted N128 state was not rewritten. N64 and N96 were instead
advanced by one short accepted step to the same persisted `1.001e-6 t_load`
time. The resulting zero-step checkpoint audit is:

| Cells | Accepted | Retries | Inner flux / supply | Inner Mach | `max(H/R)` |
|---:|---:|---:|---:|---:|---:|
| 64 | 85 | 4 | `-0.195526` | `-12.0151` | `0.1411074` |
| 96 | 100 | 4 | `-0.189096` | `-23.3424` | `0.1411658` |
| 128 | 151 | 6 | `-0.188953` | `-50.8869` | `0.1411838` |

All three states have zero incoming characteristics and zero outer Roche flux.
Relative to N128, N96 differs by `1.43e-4` of the stream supply in inner flux
and by `1.27e-4` in maximum `H/R`. The N64 differences are `6.57e-3` of supply
and `5.41e-4`, respectively. The refined shared-time gate passes.

In the final N128 completion segment, accepted residuals are at most
`7.44e-9` and accepted storage-scaled ledger defects are at most `5.06e-16`.

## Bounded N64 Extension

The N64 checkpoint was continued toward `2.1e-6 t_load`. It reaches a final
checksum-verified accepted state at `1.430993e-6 t_load`:

```text
accepted steps, total             225
inner flux / stream supply       -0.206515
inner radial Mach                -52.0655
incoming radial characteristics   0
outer Roche flux / supply         0
maximum H/R                       0.141110
```

There is no recurrence of the old fixed-reference absorber collapse. The
inner face becomes more supersonic, the outer edge remains closed, and every
persisted state remains conservative.

The requested `2.1e-6` duration was not reached. At `1.418993e-6`, the next
`1.98e-9 t_load` step hit 600 evaluations with residual `2.40e-8` while its
storage-scaled ledger defect remained `5.77e-16`. A smaller
`1e-9 t_load` continuation advances regularly to the final accepted state but
is too costly for a brute-force duration march. Bounded 100- and
200-evaluation probes are rejected safely; even their half-step residuals are
`3.36e-8` and `2.27e-8`, respectively. The accepted checkpoint is unchanged by
those probes.

This is a nonlinear-solver efficiency boundary, not a physical inner or Roche
boundary failure. The long N64 duration gate remains incomplete.

## Decision

The causally outgoing plunge implementation passes its stationary, mapping,
quadrature, tiny-step, shared-time, conservation, characteristic, and restart
preflights. It removes the physical failure of the old fixed-reference
absorber: the inner flow becomes more supersonic during loading, and no
incoming mode is manufactured or projected away.

Classification:

```text
stationary plunge and boundary contract       supported numerically
shared 1e-7 adaptive mesh gate                passed
shared N64/N96/N128 1.001e-6 gate             passed
N64 bounded extension                         supported to 1.430993e-6
N64 2.1e-6 duration gate                      incomplete: solver cost
long no-tide physical evolution               not certified
distributed tide and wind                     still blocked
```

## Locked Next Step

1. Improve the certified sparse/block Jacobian or nonlinear stopping strategy.
   It must reproduce accepted one-step states from the current N64 and N128
   checkpoints under the same residual, ledger, and 2% physical-change gates.
2. Resume N64 to `2.1e-6 t_load`, then perform one bounded comparison against
   the former `3.9166e-6` fixed-absorber failure time.
3. Begin the selected no-wind distributed-tide continuation only after the
   long no-tide gate is computationally practical and passes.
4. Do not reset the reference state, restore a subsonic projection, use the
   rejected colored Jacobian, or relax a physical or numerical gate.

Wind remains last.
