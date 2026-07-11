# Coupled Open-Overflow Eigenvalue Results

## Scope

This work package implements ADR 0011. It adds one positive inner accretion-rate
eigenvalue and one outer boundary row to the fully coupled finite `335 rg`
minidisk. Stream supply remains an absolute `5 Mdot_Edd`; wind and tidal
heating are off.

The boundary homotopy is

```text
R_edge(chi) = (1-chi) Mdot_out/Mdot_stream
            + chi G_out/G_scale.
```

Thus `chi=0` is the certified mass wall and `chi=1` is an open zero-torque
edge. No stress, thermodynamic, energy, interface, or sonic equation changes.

## Augmented Rank Gate

Adding `log(Mdot_inner/Mdot_stream)` produces `389` unknowns and equations on
the `96/64` mesh. At the wall endpoint it reproduces the original maximum
residual `1.68e-8` and exact zero outer mass flux.

| Boundary | Jacobian rank | Interface response | Sonic rank | Condition |
|---|---:|---:|---:|---:|
| mass wall | 389/389 | 2 | 2 | `2.47e5` |
| open edge | 389/389 | 2 | 2 | `1.35e5` |

## Boundary Continuation

All six coarse stages are accepted without tolerance changes:

| chi | Mdot_inner/stream | overflow | Lrad/LEdd | max outer H/R | stagnation rg |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.0000 | 0.0000 | 1.3482 | 0.3102 | 335.0 |
| 0.10 | 0.9232 | 0.0768 | 1.3092 | 0.3024 | 269.4 |
| 0.25 | 0.8051 | 0.1949 | 1.2416 | 0.2886 | 257.0 |
| 0.50 | 0.6010 | 0.3990 | 1.0940 | 0.2577 | 245.0 |
| 0.75 | 0.3893 | 0.6107 | 0.8714 | 0.2100 | 234.6 |
| 1.00 | 0.1690 | 0.8310 | 0.4803 | 0.1251 | 222.1 |

At the open edge, normalized outer torque is `1.6e-12`. The Hill tidal band
is thin, with maximum `H/R=0.0391`. The solution therefore recovers a regular
finite-density accretion/decretion split in the fully coupled transonic and
total-energy system.

## Mesh Gate

The open root prolongates from `96/64` to a full-rank `144/96` root:

```text
Mdot_inner/Mdot_stream = 0.168937
overflow                = 0.831063
Lrad/LEdd               = 0.479702
max outer H/R           = 0.124599
stagnation radius       = 222.180 rg
maximum residual        = 2.05e-11.
```

The declared refinement to `168/112` fails. Its residual is localized to the
outer endpoint:

```text
outer stress residual = 0.481
outer energy residual = 0.383
inner core residual   = 8.15e-5
interface residual    = 7.90e-13.
```

A direct `192/128` attempt and one adaptive intermediate attempt showed the
same endpoint behavior. No tolerance, projection, clipping, or residual
scaling was changed.

## Verdict

```text
numerical_status = SUPPORTED BUT NOT FULLY CERTIFIED
physical_status  = DIAGNOSTIC ONLY
```

The project now has a fully coupled, full-rank open-overflow control on two
meshes. It is substantially cooler and thinner than the ideal-wall state and
processes only about 17% of the supplied mass inward.

The steady open branch is not mesh certified. Under ADR 0011, this activates
the conservative coupled mass-energy time-evolution fallback. The next solver
must permit accumulation and moving fronts rather than introduce another
steady endpoint closure. Wind remains deferred.
