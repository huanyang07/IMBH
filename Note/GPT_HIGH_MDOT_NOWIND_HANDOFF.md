# GPT High-Mdot No-Wind Handoff

Date: 2026-07-01

This note is a compact, self-contained handoff for GPT.  It summarizes the
standard no-wind slim-disk high-accretion-rate results, which are separate from
the latest finite-minidisk stream source/sink sign-fix work.

## Executive Summary

We recovered the standard no-wind slim-disk advective branch up to
`Mdot/Edd = 5`.

This is not yet the full stream-fed/wind finite-minidisk hot branch, but it is
the necessary high-`Mdot` backbone showing that the solver can find a physical
advective slim branch when the inner supplied accretion rate is actually raised
above Eddington.

The current picture is:

```text
Track A: standard no-wind slim disk
  recovered to Mdot/Edd = 5
  advective fraction grows smoothly
  luminosity grows sublinearly
  sonic point moves inward smoothly
  disk remains height-integrated/slim, max H/R ~ 0.32 at Mdot/Edd = 5

Track B: corrected finite-minidisk stream source/sink model
  latest source-sign run was at Mdot_inner/Edd = 1
  source bookkeeping is now correct and robust
  but this fixed-inner-Mdot stream-fed case is still weakly advective
```

## High-Mdot No-Wind Result

The standard no-wind slim branch was continued to `Mdot/Edd = 5`.  The first
high-rate scout showed the expected monotonic advective trend, and the final
outer-interval residual bottleneck was solved with targeted adaptive outer
remeshing.

Representative certified/accepted sequence:

| `Mdot/Edd` | N | residual | anchor? | `f_adv_global` | `f_adv_inner(R<20rg)` | `Lrad/LEdd` | max `H/R` | `Rson/rg` |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| `1` | `512` | `5.175e-6` | no | `0.0347` | `-0.1151` | `0.5934` | `0.1572` | `5.084` |
| `2` | `640` | `6.450e-6` | no | `0.1865` | `0.0943` | `0.9660` | `0.2269` | `4.661` |
| `3` | `640` | `7.262e-6` | no | `0.3018` | `0.2562` | `1.213` | `0.2677` | `4.502` |
| `4` | `640` | `2.507e-6` | yes | `0.3894` | `0.3830` | `1.396` | `0.2956` | `4.415` |
| `5` | `768` | `2.293e-6` | yes | `0.4534` | `0.4666` | `1.541` | `0.3164` | `4.360` |

Interpretation:

- The no-wind slim solution becomes increasingly advective as `Mdot` rises.
- The inner advective diagnostic crosses positive near `Mdot/Edd ~ 1.5--1.6`.
- Radiative luminosity grows sublinearly: by `Mdot/Edd = 5`, `Lrad` is only
  about `1.54 L_Edd`.
- The sonic point moves inward smoothly.
- The disk thickens but remains in a height-integrated slim-disk regime, with
  `max H/R ~ 0.32` at `Mdot/Edd = 5`.

## Numerical Bottleneck That Was Solved

Before adaptive outer remeshing, the dominant residual above `Mdot/Edd ~ 3` was
`interval_E`, localized almost entirely in the final outer intervals near
`Rout = 10000 rg`.

Residual localization showed:

```text
Mdot/Edd = 4:
  pre-remesh residual ~1.176e-4
  peak interval_E at R ~ 9964 rg

Mdot/Edd = 5:
  pre-remesh residual ~1.349e-4
  peak interval_E at R ~ 9964 rg
```

The median and 90th-percentile interval-energy residuals were tiny, so this was
not a sonic failure or an inner advective-branch failure.  Targeted adaptive
outer remeshing fixed the problem:

```text
Mdot/Edd = 4:
  N640 adaptive outer mesh -> residual 2.507e-6, anchor quality

Mdot/Edd = 5:
  N640 adaptive outer mesh -> residual ~3.8e-6 to 5e-6, strict but not anchor
  N768 adaptive outer mesh spot check -> residual 2.293e-6, anchor quality
```

## Relation To Latest Stream Source/Sink Work

The latest pushed update fixed the finite-minidisk stream source/sink sign:

```text
dMdot/dlnR = wind_sink_prime - stream_source_prime
```

With inward-positive accretion rate:

```text
positive stream source -> Mdot_outer < Mdot_inner
positive wind sink     -> Mdot_outer > Mdot_inner
```

That corrected source/sink model was stress-tested at:

```text
Mdot_inner/Edd = 1
Rout = 300 rg
N = 640
stream source annulus centered at 0.8 Rout
```

Best corrected source-loading checkpoint:

```text
source fraction = 0.49
Mdot_outer/Mdot_inner = 0.511844
source integral / inner Mdot = 0.4881
full residual = 2.438e-6
dominant residual = interval_E
max H/R = 0.1571
integrated advective fraction = 0.03404
```

This stream-fed fixed-`Mdot_inner=1` case is not the hot branch.  It verifies
the corrected source bookkeeping.  The standard no-wind high-`Mdot` track shows
that true advection appears when the inner supplied accretion rate is raised.

## Current Scientific Interpretation

We should distinguish three statements:

1. **Recovered standard advective slim branch:** yes, for no-wind disks up to
   `Mdot/Edd = 5`.
2. **Recovered corrected stream-source bookkeeping:** yes, up to large source
   fractions at fixed `Mdot_inner/Edd = 1`.
3. **Recovered full stream-fed/wind hot branch for the IMRI finite-minidisk
   model:** not yet.

The most plausible path is to connect Track A and Track B: start from the
standard high-`Mdot` no-wind advective branch and continue into finite-boundary,
stream-fed, torque/heating, and eventually wind/source-sink physics.

## Questions For GPT

Please advise the next plan given both tracks:

1. Should the next continuation path start from the no-wind `Mdot/Edd > 1`
   advective branch and then turn on finite-boundary/stream/wind physics?
2. Or should we first push the corrected stream source/sink model at
   `Mdot_inner/Edd = 1` through the high-source `interval_E` stiffness near
   source fraction `0.49`?
3. What is the most physical stream-fed continuation parameter: raise
   `Mdot_inner`, raise stream source fraction, add stream torque/heating, or
   introduce wind mass loss?
4. What diagnostics should be required before calling the stream-fed solution a
   true advective/hot branch rather than a numerical continuation of the cool
   branch?
5. What minimal experiment best connects the recovered no-wind advective branch
   to the finite-minidisk IMRI stream model?

