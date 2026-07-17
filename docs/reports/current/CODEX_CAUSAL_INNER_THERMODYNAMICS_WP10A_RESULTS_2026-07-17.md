# Causal Inner Thermodynamics WP10a Results

**Date:** 2026-07-17
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Scope:** thermodynamic derivative, local relativistic characteristics, and
one bounded audit of the existing WP9 low-throughput stationary profile. No
global evolution, tide, wind, or production boundary replacement was run.

## Verdict

WP10a repairs the local gas+radiation causality defect without imposing a
numerical sound-speed cap.

The causal thermodynamic prototype passes: all audited sound speeds are below
`c`, and the radiation-dominated values approach `c/sqrt(3)`. The result does
not yet unlock production evolution. The first audited zero-incoming point is
only at `2.0001 rg`, while the stationary plunge and global conservative flux
still use the older Newtonian system. More decisively, the old PW profile's
azimuthal speed exceeds `c` by `3 rg`, so the radial-only crossing is not a
physical full-state excision.

## Thermodynamic Derivative

For the shared gas+radiation EOS, let

```text
c_s,N^2 = (dP/d rho)_s
h_th    = e + P/rho.
```

Including rest-mass energy gives the causal local sound speed

```text
a^2 = c^2 c_s,N^2 / (c^2 + h_th).
```

This is a derived change in thermodynamic variables, not

```text
a = min(c_s,N, c_limit).
```

The implementation reuses the exact gas+radiation entropy derivative and
specific enthalpy already used by the Roche-boundary EOS.

## Low-Rate Profile Audit

The one-dimensional local radial characteristic speeds use
special-relativistic velocity addition in an outward-oriented orthonormal
frame:

```text
(v_R-a)/(1-v_R a/c^2), v_R, v_R, (v_R+a)/(1+v_R a/c^2).
```

| Radius (`rg`) | `v_R/c` | `v_phi/c` | Newtonian `c_s/c` | Causal `a/c` | Incoming modes |
|---:|---:|---:|---:|---:|---:|
| 4.5 | `-8.5359e-6` | `0.8484` | `4.9459e-3` | `4.9457e-3` | 1 |
| 3.0 | `-3.4763e-4` | `1.7096` | `1.2242e-1` | `1.1976e-1` | 1 |
| 2.1 | `-1.2450e-2` | `9.6183` | `2.9510` | `0.56661` | 1 |
| 2.01 | `-7.0869e-2` | `37.266` | `14.941` | `0.57692` | 1 |
| 2.001 | `-0.26276` | `117.12` | `51.890` | `0.57731` | 1 |
| 2.0001 | `-0.86150` | `357.06` | `164.93` | `0.57735` | 0 |

The Newtonian sound speed diverges near the pseudo-potential singularity. The
causal derivative instead saturates at the radiation limit. The accepted WP9
stationary trajectory becomes radially acoustic-supersonic only in the final
audited interval, after its transverse velocity has already become
nonphysical.

## What This Does And Does Not Establish

Established:

1. The shared gas+radiation EOS admits a subluminal relativistic acoustic
   derivative with the correct cold and radiation limits.
2. Local relativistic characteristic counting can identify a zero-incoming
   region on the existing low-rate profile.
3. No arbitrary sound-speed clipping is needed.

Not established:

1. The current stationary critical point is a critical point of a causal
   conservative time-dependent system.
2. The current Newtonian finite-volume flux has the relativistic
   characteristics used by this audit.
3. Relativistic transverse rotation and spacetime lapse/shift effects are
   negligible in the final inner characteristic system.
4. An excision at `2.0001 rg` is numerically or physically acceptable in the
   Paczynski-Wiita model.
5. Fresh-loading evolution is ready to resume.

## Locked Next Plan

The next work package is an equations and rank design for a conservative
causal inner core. It must select the complete physical system before code is
promoted to production:

1. define the conserved rest-mass, radial-momentum, angular-momentum, and
   energy variables;
2. derive one face flux and its characteristic speeds from those variables;
3. specify gravity and rotation consistently in the causal formulation;
4. derive the stationary critical conditions from the same equations;
5. state the inner excision and outer matching row counts;
6. recover the cold Newtonian slim branch in its valid regime;
7. obtain a low-throughput stationary inflow with subluminal primitives and
   zero incoming modes at a finite, resolved excision;
8. pass N64/N96 conservative mapping and one tiny implicit step before any
   physical loading trajectory.

Tide and wind remain blocked. The existing high-throughput causal-plunge
evidence remains valid for its original reference-state scope.

## Verification

```text
focused causal/EOS tests       14 passed
complete repository suite      386 passed, 4 subtests passed
```

Machine-readable evidence:

```text
outputs/tables/causal_inner_thermodynamics_wp10a.json
```

Reproduction:

```bash
PYTHONPATH=src python3 scripts/run_global_inner_boundary_architecture_gate.py
PYTHONPATH=src python3 scripts/run_causal_inner_thermodynamics_wp10a.py
```
