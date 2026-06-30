# GPT Review Prompt: Standard Slim-Disk Mdot Continuation Bottleneck

Please review the latest repository state:

```text
https://github.com/huanyang07/IMBH
```

Start with these files:

```text
Note/CODEX_5EEBBEE_STANDARD_SLIM_BENCHMARK_FIX_PLAN.md
Note/GPT_REVIEW_PROMPT_STANDARD_SLIM_BENCHMARK.md
scripts/run_standard_slim_analytic_seed_audit.py
scripts/run_standard_slim_sonic_compatibility_probe.py
scripts/run_standard_slim_sonic_root_injection.py
scripts/run_standard_slim_rout_injection_ladder.py
scripts/run_standard_slim_mdot_injection_ladder.py
outputs/tables/slim_benchmark_rout_injection_ladder.md
outputs/tables/slim_benchmark_mdot_injection_rout10000_onepercent.md
outputs/tables/slim_benchmark_mdot_injection_rout10000_twopercent.md
outputs/tables/slim_benchmark_mdot_injection_rout10000_2pct_ladder.md
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
src/imri_qpe/layer3_minidisk_1d/transonic_continuation.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
```

## Context

We are trying to validate the transonic slim-disk solver on a standard
single-BH, no-wind, no-stream, no-tide benchmark before returning to the IMRI
minidisk/wind model.

The earlier low-Mdot benchmark failed. Codex then added analytic seed audits,
local sonic compatibility probes, sonic-root injection, and an outer-radius
ladder.

The important new result is that the low-Mdot benchmark now passes at
`Mdot/Edd = 1e-3`.

## Recovered R_out benchmark

Using `scripts/run_standard_slim_rout_injection_ladder.py`, the solution was
continued through:

```text
R_out/rg = 1000, 1500, 2000, 3000, 5000, 7000, 10000
```

Configuration:

```text
Paczynski-Wiita potential
Mdot/Edd = 1e-3
alpha = 0.01
stress_factor = 1.0
no wind
outer_closure = thin_value
interval_residual_form = differential
integrated_residual_weighting = none
N = 80 at R_out=1000, up to N=128 at R_out=10000
```

Key accepted rows:

```text
R_out=1000,  N=80,  full residual=1.982e-6, Rson=5.911 rg
R_out=3000,  N=96,  full residual=1.305e-6, Rson=5.910 rg
R_out=10000, N=128, full residual=1.428e-6, Rson=5.920 rg
```

This appears to recover the low-Mdot standard no-wind slim/thin benchmark.

## New Mdot continuation result

Codex added:

```text
scripts/run_standard_slim_mdot_injection_ladder.py
```

The script starts from the accepted `R_out=10000 rg`, `Mdot/Edd=1e-3`
checkpoint, remaps the profile to a new Mdot, performs a local free-lambda
sonic-root reinjection, and square-polishes the global system using a
block-Jacobian Newton/LSQ route.

The result is that Mdot continuation works, but only with small steps near
`Mdot/Edd=1e-3`.

One-percent full-radius test:

```text
R_out=10000, N=128
Mdot/Edd=9.9e-4:  full residual=1.654e-6, accepted
Mdot/Edd=1.01e-3: full residual=2.519e-6, accepted
```

Two-percent one-step full-radius test:

```text
R_out=10000, N=128
Mdot/Edd=9.8e-4:  full residual=3.455e-6, accepted
Mdot/Edd=1.02e-3: full residual=4.379e-6, accepted
```

Two-percent multi-step ladder:

```text
anchor: Mdot/Edd=0.001,      full=1.428e-6
down:   Mdot/Edd=0.00098,    full=3.455e-6, accepted
down:   Mdot/Edd=0.0009604,  full=7.578e-6, accepted
down:   Mdot/Edd=0.0009412,  full=2.764e-5, failed, dominant interval_R

up:     Mdot/Edd=0.00102,    full=4.379e-6, accepted
up:     Mdot/Edd=0.0010404,  full=8.046e-6, accepted
up:     Mdot/Edd=0.0010612,  full=9.807e-6, accepted
up:     Mdot/Edd=0.0010824,  full=8.987e-6, accepted
up:     Mdot/Edd=0.0011041,  full=1.100e-5, failed, dominant interval_R
```

Five-percent and ten-percent jumps fail:

```text
R_out=10000, 5 percent down to 9.5e-4: full=5.116e-5, dominant interval_R
R_out=10000, 5 percent up to 1.05e-3:  full=1.444e-5, dominant interval_R
R_out=10000, 10 percent-ish jumps fail more clearly
```

The same behavior also appears at `R_out=1000`: 1 percent steps pass, while
10 percent jumps fail. Pivot scans over `C1`, `C2`, and `K` do not remove the
problem. The failure is mostly a global radial interval residual after remap
and polish, not a local sonic algebra inconsistency.

## Current interpretation

The low-Mdot standard benchmark is now recovered at fixed Mdot and large
outer radius, but Mdot continuation remains very step-size sensitive. The
current practical safe step near `Mdot/Edd=1e-3` is about 1-2 percent.

Accepted rows often have `optimizer_success=False` because the LSQ fallback
hits the configured evaluation cap after the full physical residual has already
fallen below the acceptance tolerance. This is bookkeeping, not necessarily a
physical rejection.

## Questions for review

1. Is the remaining Mdot-continuation bottleneck mainly due to the remap
   formula, the fixed square sonic-pivot polish, finite-difference collocation,
   insufficient Newton globalization, or true branch stiffness?
2. Should the next implementation be an adaptive Mdot step controller using
   accepted/rejected residual feedback, or should we first improve the
   continuation variable/tangent, e.g. pseudo-arclength in `log Mdot` with the
   recovered standard branch?
3. How should the initial remap scale `logu`, `logT`, `Rson`, and `lambda0`
   with Mdot in the thin/slim limit? The current helper scales temperature as
   `T -> T * mdot_factor**0.25` and interpolates the rest.
4. Since failures are dominated by interval_R, what diagnostic should be run
   next to distinguish:
   - poor radial momentum remapping,
   - inadequate inner sonic/global matching,
   - collocation discretization error,
   - Newton/LSQ optimization floor,
   - a need for integrated or higher-order collocation residuals?
5. What is the best next sprint if the goal is to continue from
   `Mdot/Edd=1e-3` toward `1e-2`, and eventually toward the advective/wind
   branch?

Please propose a prioritized fix/test plan. The immediate goal is not yet the
IMRI wind branch; it is a robust standard no-wind slim-disk Mdot ladder from
`1e-3` toward `1e-2` and `1e-4` with mesh/outer-radius validation.
