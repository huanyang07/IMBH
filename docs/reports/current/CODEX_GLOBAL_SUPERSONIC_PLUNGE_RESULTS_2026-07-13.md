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

## Longer Adaptive Result

The same checkpoints were advanced under unchanged gates. N64 and N96 reach
the requested `1e-6 t_load` target. The refined N128 run was stopped after its
last accepted state at `8.875e-7 t_load` because completing the strict target
would require a long sequence of increasingly expensive finite-difference
Jacobian solves. The stopped trial was discarded; only checksum-verified
accepted checkpoints remain.

```text
mesh                 N64           N96           N128
elapsed/t_load       1.00022e-6    1.00000e-6    8.87500e-7
accepted steps       84            99            95
recovered retries    4             4             5
inner flux/supply   -0.19552      -0.18907      -0.18573
inner Mach          -12.006       -23.273       -26.687
outer Roche flux     0             0             0
incoming modes       0             0             0
max(H/R)             0.141107      0.141166      0.141184
```

The accepted N128 timestep was reduced by the fixed 2% relative thickness
change gate, not by a failed nonlinear root, conservation defect, Roche
opening, or incoming characteristic. In the preceding bounded segment, all
accepted storage-scaled ledger defects were below `6.4e-16` and residuals
below `1.33e-11`.

The three final states are not a shared-time mesh comparison because N128 is
younger by `11.25%` of the requested interval. The report JSON now labels such
comparisons explicitly as non-shared-time diagnostics.

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
N64/N96 1e-6 duration gate                    passed
N128 1e-6 duration gate                       incomplete at 8.875e-7
long no-tide physical evolution               not certified
distributed tide and wind                     still blocked
```

## Locked Next Step

1. Resume only the accepted N128 checkpoint to `1e-6 t_load` with the same
   residual, ledger, and 2% physical-change gates. Do not rerun N64/N96.
2. Emit one zero-step shared-time snapshot and apply the existing flux and
   `H/R` mesh gates.
3. Run one bounded N64 plunge-boundary extension beyond `2e-6 t_load` to test
   whether the former `3.9166e-6` inner-boundary collapse has disappeared.
4. If those gates pass, begin the already selected no-wind distributed-tide
   continuation. If the refined march develops a true residual, ledger, or
   characteristic failure, stop and diagnose that block without resetting a
   reference state or relaxing the controller.

Wind remains last.
