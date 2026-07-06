# Codex Mdot=5 Source-Band Split-Collocation Results

Date: 2026-07-06

This note records the first attempt to turn the radial representation caveat
into an active solver residual instead of a post-processing audit only.

## Implementation

Added two opt-in interval residual forms in
`src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py`:

```text
split_differential
split_rms_differential
```

Both forms sub-sample each interval at its midpoint and at internal stream
source, wind sink, and outer-buffer transition radii when those transitions
fall inside the interval.

Definitions:

```text
split_differential:
    return the signed worst sub-interval differential residual per equation.

split_rms_differential:
    return sqrt(mean(sub-interval residual^2)) per equation.
```

The split-RMS form is smoother for least-squares polishing; the signed-worst
form is sharper as a diagnostic.

Regression coverage was added in `tests/test_transonic_collocation.py`.

## Test Status

Focused tests:

```text
PYTHONPATH=src python -m pytest \
  tests/test_transonic_collocation.py \
  tests/test_winds.py \
  tests/test_transonic_local.py

86 passed
```

## Why This Was Needed

The eta_E=90 N168 checkpoint was strict under the original midpoint
differential residual:

```text
final_full = 6.531e-06
local_R = 6.531e-06
local_E = 4.432e-06
mass_residual_max = 6.858e-07
```

But the radial representation audit showed a large source-transition mismatch
near R~250 rg:

```text
representation_tau ~ 1.476e-02
source_prime/Mdot ~ 14.138
```

The goal here was to expose that hidden subcell residual directly to the
optimizer.

## Split-Residual Seed Audits

Starting checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta90_N168_localjac_innerweight20/stage_00_etaE_90_N168.npz
```

Seed-only outputs:

```text
outputs/tables/m5_local_mdot_eta90_N168_splitdiff_seed_audit.md
outputs/tables/m5_local_mdot_eta90_N168_splitrms_seed_audit.md
```

Results:

| form | N | final_full | local_R | local_E | mass max | peak R row | peak E row |
|---|---:|---:|---:|---:|---:|---:|---:|
| split_differential | 168 | 1.477e-01 | 1.961e-02 | 1.477e-01 | 6.858e-07 | 235.369 rg | 245.324 rg |
| split_rms_differential | 168 | 1.048e-01 | 1.776e-02 | 1.048e-01 | 6.858e-07 | 235.369 rg | 245.324 rg |

Interpretation:

```text
The midpoint-strict checkpoint is not split-collocation strict.  The hidden
defect is a real source-annulus subcell residual, mostly in the energy row
between R~235 and 255 rg.
```

## Global Split Polishes

Commands used local finite-difference Jacobian and inner-Mdot weighting:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_USE_LOCAL_JACOBIAN=1
IMBH_MDOT5_LOCAL_MDOT_ETA_LOCAL_JACOBIAN_STEP=1e-6
IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_MDOT_WEIGHT=20
```

Outputs:

```text
outputs/tables/m5_local_mdot_eta90_N168_splitdiff_global_localjac_probe.md
outputs/tables/m5_local_mdot_eta90_N168_splitrms_global_localjac_probe.md
outputs/tables/m5_local_mdot_eta90_N168_splitrms_global_localjac_resume.md
```

Results:

| run | N | final_full | local_R | local_E | mass max | nfev | accepted |
|---|---:|---:|---:|---:|---:|---:|---|
| signed-worst split | 168 | 3.701e-02 | 3.701e-02 | 2.152e-02 | 1.732e-03 | 14 | no |
| split-RMS probe | 168 | 2.673e-02 | 2.673e-02 | 1.575e-02 | 6.231e-04 | 100 | no |
| split-RMS resume | 168 | 2.629e-02 | 2.629e-02 | 1.569e-02 | 5.198e-04 | 160 | no |

The split-RMS form is clearly the better optimizer target, but N168 stalls
near `final_full ~ 2.6e-2`.

## Resolution/Remap Checks

Direct nested remap from the improved N168 split-RMS state to N200:

```text
outputs/tables/m5_local_mdot_eta90_N200_nested_splitrms_from_splitrms_seed.md
```

Result:

```text
final_full = 2.138e-02
local_R = 2.138e-02
local_E = 2.128e-02
mass max = 4.027e-04
```

N200 local-Jacobian polish:

```text
outputs/tables/m5_local_mdot_eta90_N200_splitrms_global_localjac_probe.md
```

Result:

```text
final_full = 2.112e-02
local_R = 1.625e-02
local_E = 2.112e-02
mass max = 7.230e-04
```

N240 direct nested remap from N200:

```text
outputs/tables/m5_local_mdot_eta90_N240_nested_splitrms_from_N200_seed.md
```

Result:

```text
final_full = 2.530e-02
local_R = 1.237e-02
local_E = 2.530e-02
mass max = 7.916e-04
```

Forced transition-aligned remaps were worse:

```text
N200 transition-aligned seed from midpoint state: final_full ~ 1.65
N200 transition-aligned seed from split-RMS state: final_full ~ 1.71
```

The transition nodes land exactly on the compact source inner edge, peak, and
outer edge, but the current interpolation/remap produces a rough state.

## Current Interpretation

Accepted:

```text
The previous representation caveat has been converted into an active numerical
test.  The source-annulus subcell defect is real and large under split
collocation.
```

Not accepted:

```text
The eta_E=90 branch is not yet split-collocation strict.  Best split-RMS result
so far is final_full ~ 2.1e-2, not <= 1e-5.
```

Most likely bottleneck:

```text
The compact source annulus is under-resolved by the current piecewise-linear
state representation.  Compressing multiple subcell residuals back into two
interval rows helps expose the problem but does not provide enough local
degrees of freedom to satisfy both the mass/source transition and the radial/
energy balances.
```

## Recommended Next Step

The next structural change should be one of:

```text
1. Add an overdetermined source-band collocation mode with explicit subcell
   residual rows in the source annulus.

2. Or add a true multi-domain/source-annulus grid with extra state nodes inside
   the compact source support, plus a remapper that preserves the source-band
   differential defect.
```

The first option is probably faster to test.  It requires updating the local
Jacobian path because the current local-Jacobian implementation assumes exactly
`3N+2` residual rows.  A practical implementation would:

```text
- keep the existing square residual as the base rows;
- append extra split-RMS or explicit subcell rows only for intervals in the
  source annulus;
- build a rectangular local finite-difference Jacobian;
- use least_squares with the rectangular Jacobian;
- audit both base midpoint rows and extra subcell rows separately.
```

Until this is done, lowering eta_E further would only continue a midpoint-
strict branch and would not solve the real source-band representation problem.
