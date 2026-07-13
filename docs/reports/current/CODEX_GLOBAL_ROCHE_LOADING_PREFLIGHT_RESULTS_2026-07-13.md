# Global Roche Loading Preflight Results

**Date:** 2026-07-13, Asia/Shanghai
**Starting commit:** `34ee153`

## Scope

This is the first bounded WP3 evolution using the exact-EOS physical Roche
edge. It tests the initial no-tide loading derivative and timestep/mesh
behavior. It is not a long-time solution or a limit-cycle claim.

The run uses:

```text
stream supply                  5 Mdot_Edd
distributed tidal torque      off
wind                           off
outer edge                     closed-to-choked Roche provider
outer viscous torque           zero
inner edge                     reference characteristic absorber
radiative cooling              on
radial and temporal column work on
```

## Mesh and timestep gate

One full `1e-9 t_load` step and two half steps pass at every requested mesh:

| Cells | Inner accretion / supply | Outer overflow / supply | Disk mass change | Max residual | Max ledger defect |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.16723 | 0 | `8.328e-10` | `2.42e-12` | `1.78e-16` |
| 96 | 0.16876 | 0 | `8.313e-10` | `9.86e-13` | `2.54e-16` |
| 128 | 0.16889 | 0 | `8.311e-10` | `4.94e-12` | `1.08e-16` |

The accumulated fraction equals `1-Mdot_in/Mdot_stream` to the global ledger
accuracy. Full-step versus two-half-step differences are below `2.4e-14` in
relative disk mass and below `1e-11` in maximum `H/R`. The Roche channel stays
closed on every accepted step.

At the end of the bounded preflight:

```text
max(H/R) = 0.14111, 0.14116, 0.14118  (N64,N96,N128)
```

so the initial mapped state remains geometrically moderate.

## Step-size bracket

Additional N64 one-step probes give:

| Total step | Accepted | Max residual | Roche state |
|---:|---:|---:|---|
| `1e-8 t_load` | yes | `7.47e-13` | closed |
| `1e-7 t_load` | yes | `2.12e-12` | closed |
| `1e-6 t_load` | no | `1.37e-4` | closed candidate, state reverted |

The rejected large jump has a storage-scaled ledger defect `3.52e-11`, but it
fails the nonlinear residual gate and is not accepted. This is a timestep/Newton
reach limitation, not a Roche-threshold failure.

## Scientific interpretation

The former open steady control is now serving only as an initial profile. Once
the physical edge replaces donor overflow, approximately 83% of the supplied
mass initially accumulates while about 17% accretes inward. This is the
expected qualitative change and is the first direct realization of the closed
loading-reservoir picture.

No claim about a hot state, steady state, or cycle is allowed from
`1e-9 t_load`. The next solver layer needs adaptive backward-Euler stepping,
accepted-state restart checkpoints, and geometric timestep growth. A rejected
step must halve and retry; accepted steps may grow only under residual,
iteration-count, thermal-change, and Roche-distance controls.

Distributed tide and wind remain blocked until evolution covers physically
relevant loading, thermal, and viscous times with mesh and restart support.
