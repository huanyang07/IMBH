# Codex Stream Source/Sink Sign Fix Results

Date: 2026-07-01

This update starts the corrected finite-minidisk stream-source work after the
standard no-wind slim benchmark was certified to `Mdot/Edd=5`.

## Problem fixed

The old stream-mass helper used:

```text
Mdot_local = Mdot_inner * (1 + stream_mass_fraction * shape)
```

so positive stream loading made:

```text
Mdot_outer > Mdot_inner
```

That is the sign of a wind/recycling sink, not a conservative no-wind stream
source.

With inward-positive accretion rate, the intended bookkeeping is:

```text
dMdot/dlnR = wind_sink_prime - stream_source_prime
```

where both `stream_source_prime` and `wind_sink_prime` are positive rates per
`dlnR`.

## Implementation

Core changes:

- Added explicit helpers in `src/imri_qpe/layer3_minidisk_1d/transonic_local.py`:
  - `stream_source_prime(logR, params)`
  - `wind_sink_prime(logR, params)`
  - `mdot_profile_from_source_sink(logR, params)`
- Kept `stream_mass_rate_and_derivative` as a compatibility wrapper around the
  new signed source/sink profile.
- Changed stream heating to use positive `stream_source_prime`, not signed
  `dMdot/dlnR`.
- Added explicit `TransonicSlimParams` fields:
  - `stream_source_fraction`, `stream_source_center_fraction`,
    `stream_source_log_width`
  - `wind_sink_fraction`, `wind_sink_center_fraction`,
    `wind_sink_log_width`
- Kept `stream_mass_fraction` as a deprecated source alias for old scripts and
  checkpoints.

Runner updates:

- `scripts/run_standard_slim_stream_mass_annulus_scan.py`
- `scripts/run_standard_slim_stream_heating_annulus_scan.py`
- `scripts/run_standard_slim_stream_heating_geometry_scan.py`
- `scripts/run_standard_slim_stream_source_strength_scan.py`
- `scripts/run_standard_slim_stream_heating_profile_diagnostics.py`

New audit script:

- `scripts/run_stream_source_sink_bookkeeping_audit.py`

## Audit result

Output:

- `outputs/tables/stream_source_sink_bookkeeping_audit.md`

Key rows:

| case | source frac | wind frac | `Mdot_outer/Mdot_inner` | `Delta Mdot/param` | source integral | wind integral | relative budget error |
|---|---:|---:|---:|---:|---:|---:|---:|
| source explicit | `0.03` | `0` | `0.9701` | `-0.02989` | `0.02989` | `0` | `3.614e-10` |
| source legacy alias | `0.03` | `0` | `0.9701` | `-0.02989` | `0.02989` | `0` | `3.614e-10` |
| wind only | `0` | `0.02` | `1.019` | `+0.01942` | `0` | `0.01942` | `3.050e-9` |
| mixed | `0.05` | `0.02` | `0.9696` | `-0.03039` | `0.04981` | `0.01942` | `2.448e-9` |

This confirms the corrected sign:

```text
positive stream source -> Mdot_outer < Mdot_inner
positive wind sink     -> Mdot_outer > Mdot_inner
```

## Smoke test

I ran a small corrected stream-mass scan:

- `outputs/tables/slim_benchmark_stream_mass_annulus_corrected_sign_smoke.md`

At source fraction `1e-3`:

```text
Mdot_outer/Mdot_inner = 0.999004
relative mass budget error = 9.509e-08
final residual = 1.019e-07
```

So the corrected source sign is compatible with the existing square Newton
polish for small stream loading.

## Stress tests

I then pushed the no-wind finite-minidisk source annulus at the certified
`Mdot/Edd=1`, `Rout=300 rg`, `N=640` benchmark.

First pass:

- `outputs/tables/slim_benchmark_stream_mass_annulus_corrected_sign_stress.md`

This accepted source fractions through `0.1`; the `0.1` row had
`Mdot_outer/Mdot_inner = 0.90038` and full residual `3.262e-6`, just above the
strict-anchor threshold used in the runner.

With refreshed outer-slope repolish:

- `outputs/tables/slim_benchmark_stream_mass_annulus_corrected_sign_refresh_to0p3.md`

The same ladder became strict through source fraction `0.2`:

```text
source fraction 0.2
Mdot_outer/Mdot_inner = 0.800753
full residual = 1.686e-6
dominant = outer_omega
```

A direct jump to `0.3` failed at full residual `3.417e-2`, dominated by
`outer_omega`. This was a continuation-step failure rather than a physical
source-sign problem: staged source-fraction steps recovered it cleanly.

Useful staged ladders:

- `outputs/tables/slim_benchmark_stream_mass_annulus_corrected_sign_bracket_0p2_0p3.md`
- `outputs/tables/slim_benchmark_stream_mass_annulus_corrected_sign_bracket_0p3_0p35_smallstep.md`
- `outputs/tables/slim_benchmark_stream_mass_annulus_corrected_sign_bracket_0p41_0p42_tinystep.md`
- `outputs/tables/slim_benchmark_stream_mass_annulus_corrected_sign_bracket_0p42_0p45_midstep.md`
- `outputs/tables/slim_benchmark_stream_mass_annulus_corrected_sign_bracket_0p45_0p5_midstep.md`

Best accepted checkpoint from this pass:

```text
source fraction = 0.49
Mdot_outer/Mdot_inner = 0.511844
source integral over inner Mdot = 0.4881
relative mass budget error = 4.659e-5
full residual = 2.438e-6
dominant = interval_E
Rson = 5.07 rg
max H/R = 0.1571
integrated advective fraction = 0.03404
```

The attempted `0.495` continuation became very expensive during Jacobian
construction and was interrupted. The important change near the top of this
scan is that the bottleneck shifts from harmless outer angular closure
residuals to `interval_E`; that is a numerical/collocation stiffness issue to
diagnose before claiming robust source fractions above about `0.49`.

## Tests

Full test suite:

```text
140 passed in 2.50s
```

## Next step

The corrected source/sink sign is fixed and stress-tested. Next, diagnose the
high-source `interval_E` bottleneck near source fraction `0.49` with residual
localization, then decide whether to improve the source-fraction continuation
predictor or add local/adaptive mesh refinement around the stream annulus.
