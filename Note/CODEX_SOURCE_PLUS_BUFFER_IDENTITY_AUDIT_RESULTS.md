# Source-Plus-Buffer Identity Audit Results

Date: 2026-07-06

## Goal

Implement the first part of the GPT next plan:

1. freeze the current source-plus-buffer eta_E=100 anchors as regression references;
2. add an audit that compares old source-band differential rows, endpoint-compatible ODE-integral rows, source-interface finite-volume rows, and source-element polynomial/Simpson rows interval by interval.

The original identity audit is diagnostic only. The later mass-row homotopy
addendum below adds an optional production-residual replacement for the mass
rows only.

## Code Changes

Primary file:

- `scripts/run_mdot5_local_mdot_eta_continuation.py`

New flag:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_IDENTITY_AUDIT=1`

New diagnostics:

- `source_plus_buffer_identity_audit` in the profile JSON.
- Compact summary columns in the table JSON/Markdown:
  - old midpoint maxima: radial, energy, mass;
  - ODE-integral maxima: radial, energy, mass;
  - source-interface FV maxima: mass, energy;
  - source-element FV maxima: mass, energy;
  - interface/element energy numerator compatibility;
  - cumulative mass/energy compatibility;
  - ODE matrix condition number maxima;
  - row-local ODE conditioning diagnostics.

Per source-plus-buffer interval, the profile rows now include:

- `R_left_rg`, `R_mid_rg`, `R_right_rg`;
- old midpoint radial/energy/mass residuals;
- linear midpoint residuals;
- endpoint-compatible ODE-integral radial/energy/mass residuals;
- source-interface FV mass and energy terms;
- source-element polynomial FV mass and energy terms;
- cumulative mass/energy increment compatibility;
- physical energy numerator and denominator values;
- `Qvisc`, `Qstream`, `Qrad`, `Qadv`, `Qwind` integrals;
- ODE slopes and ODE matrix condition numbers at left/mid/right points.
- singular values and simple row/column-equilibrated condition numbers;
- `g_direct = -A^{-1} c`, SVD/pseudoinverse slope, old interval slope
  `g_old = (z_R-z_L)/h`;
- local equation defects `A g_old + c`, `A g_direct + c`, and `A g_svd + c`.

## Run

Input checkpoint:

- `outputs/checkpoints/m5_source_plus_buffer_production_eta100_N164_bandonly_nfev8/stage_00_etaE_100_N164.npz`

Command stem:

- `outputs/tables/m5_source_plus_buffer_identity_eta100_N164_seed.*`

Physical setup:

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact C2 stream source
- `torque_delta_l_fraction = +0.005`
- `eta_E = 100`
- `N = 164`

## Compact Results

The checkpoint remains strict:

- `final_full = 9.354290283e-6`
- `accepted_exploratory = true`

Identity audit:

- source-plus-buffer intervals audited: `18`
- old midpoint max: `9.354e-6`
  - old radial max: `9.354e-6`
  - old energy max: `6.672e-6`
  - old mass max: `1.709e-7`
- ODE-integral max: `14.484`
  - ODE radial integral max: `14.484`
  - ODE energy integral max: `1.613`
  - ODE mass/interface max: `1.208e-2`
- source-interface FV max: `1.208e-2`
- source-element FV max: `2.178e-2`
- energy numerator compatibility max: `7.124e-3`
- mass cumulative compatibility max: `1.718e-16`
- energy cumulative compatibility max: `1.265e-11`
- ODE matrix condition max: `1.814e5`

Peak locations:

- max ODE condition: `R ~= 250.431 rg`
- max ODE radial/energy integral defect: `R ~= 255.626 rg`
- max source-element energy FV defect: `R ~= 255.626 rg`
- max energy numerator compatibility defect: `R ~= 250.431 rg`
- max source-interface energy defect: `R ~= 219.903 rg`

Row-local ODE conditioning:

- smallest singular value min: `2.375e-5` at `R ~= 250.431 rg`
- raw condition max: `1.814e5`
- row-scaled condition max: `1.074e3`
- row+column-scaled condition max: `2.377e2`
- max `|g_direct - g_old|_inf`: `1.025e3`
- max `|g_svd - g_old|_inf`: `1.025e3`
- max `|A g_old + c|_inf`: `10.044`
- max `|A g_direct + c|_inf`: `2.49e-14`
- max `|A g_svd + c|_inf`: `2.38e-13`

This says the direct solve and SVD/pseudoinverse solve agree very well; the
catastrophic ODE-integral row is not mainly a noisy inversion artifact.  It is
showing that the old midpoint-strict interval slope is not the local ODE-flow
slope in the compact source band.

## Interpretation

The audit confirms GPT's structural diagnosis, but with an extra nuance:

- The old midpoint production rows are strict in the source band.
- The source-interface cumulative variables are internally compatible with their own FV integrals:
  - mass cumulative compatibility is near machine precision;
  - energy cumulative compatibility is very small.
- The source-element polynomial/FV energy view still sees a significant defect.
- The endpoint-compatible ODE-integral view is not merely mildly different from the old midpoint rows; it is catastrophically different in the same band.
- The large ODE-integral defects coincide with a large local ODE matrix condition number.

So the immediate conclusion is:

> Before using endpoint-compatible ODE-integral rows as production replacements, we need to understand whether the ODE inversion `g = -A^{-1}c` is trustworthy in this source band. The audit says the old midpoint residual can be strict while the local ODE flow map is badly conditioned.

This does not invalidate the band-replacement plan, but it changes the safest order:

1. Keep the identity audit as a regression gate.
2. Add a row-local derivative/conditioning audit for source-band ODE-integral rows.
3. Build the homotopy with a guarded option:
   - first replace mass rows only;
   - then radial/energy ODE-integral rows only after conditioning is understood;
   - keep physical energy numerator compatibility as an audit until equivalence is established.

## Recommended Next Step

Implement the band-local replacement homotopy in stages:

1. Add `SOURCE_PLUS_BUFFER_REPLACE_BAND=1` and `SOURCE_PLUS_BUFFER_CHI`.
2. Start with mass-row replacement only, because mass/FV/cumulative compatibility is already well behaved.
3. Add radial and energy ODE-integral replacement behind separate flags.
4. Include condition-number gates for the ODE-integral rows.
5. Run `chi = 0, 0.05, 0.1, 0.2` first before attempting the full ladder.

Do not lower `eta_E` until eta_E=100 is strict under the replacement formulation and the identity audit no longer shows unresolved contradictions.

## Addendum: Mass-Only Replacement Homotopy

New flags:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_REPLACE_MASS`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_REPLACE_RADIAL`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_REPLACE_ENERGY`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_REPLACE_CHI`

Only the mass replacement is implemented as a production residual.  Radial and
energy replacement flags are parsed and reported, but are intentionally not
used yet.

Mass replacement is a source-plus-buffer interval blend:

```text
mass_row = (1 - chi) old_mass_row + chi finite_volume_mass_row
```

with radial and energy midpoint rows unchanged.

Runs from the strict eta_E=100 compact-source checkpoint:

| stem | chi | final | old-base full | production mass | intR | intE | max FV-old mass gap | ODE-integral max | max `A g_old + c` | nfev | accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `m5_source_plus_buffer_massreplace_chi000_eta100_N164` | 0.00 | `9.354e-6` | `9.354e-6` | `3.874e-7` | `9.354e-6` | `6.672e-6` | `1.208e-2` | `14.484` | `10.044` | seed | yes |
| `m5_source_plus_buffer_massreplace_chi005_eta100_N164` | 0.05 | `6.342e-6` | `6.350e-4` | `5.010e-8` | `6.342e-6` | `4.228e-6` | `1.270e-2` | `14.460` | `10.043` | 33 | yes |
| `m5_source_plus_buffer_massreplace_chi010_eta100_N164` | 0.10 | `6.092e-6` | `1.339e-3` | `7.621e-7` | `6.092e-6` | `3.974e-6` | `1.339e-2` | `14.457` | `10.042` | 8 | yes |
| `m5_source_plus_buffer_massreplace_chi015_eta100_N164` | 0.15 | `7.764e-6` | `2.124e-3` | `1.756e-6` | `6.061e-6` | `3.872e-6` | `1.416e-2` | `14.456` | `10.042` | 5 | yes |
| `m5_source_plus_buffer_massreplace_chi020_eta100_N164` | 0.20 | `1.354e-5` | `3.004e-3` | `3.085e-6` | `6.201e-6` | `3.893e-6` | `1.502e-2` | `14.456` | `10.041` | 5 | no |
| `m5_source_plus_buffer_massreplace_chi020_from015_eta100_N164` | 0.20 | `1.256e-5` | `3.004e-3` | `2.867e-6` | `6.075e-6` | `3.802e-6` | `1.502e-2` | `14.456` | `10.041` | 5 | no |
| `m5_source_plus_buffer_massreplace_chi020_localjac_eta100_N164` | 0.20 | `2.587e-5` | `2.972e-3` | `2.587e-5` | `6.079e-6` | `3.826e-6` | `1.499e-2` | `14.456` | `10.041` | 3 | no |

Interpretation:

- Mass-only replacement is a smooth deformation through `chi = 0.15`.
- `chi = 0.20` is close but not strict with the current global corrector.
- The legacy old-base full residual grows because it still measures the old
  pilot mass rows; the active production residual is the blended mass row.
- The ODE-integral defect and `A g_old + c` defect remain essentially unchanged
  along the mass-only ladder, as expected.
- This supports GPT's proposed ordering: mass replacement can be developed
  first, but it does not solve the radial/energy source-band representation
  defect.

## Sanity Check Caveat

I also tried a no-source/no-wind source-plus-buffer identity audit on the
existing `m5_source_element_identity_nowind_nosource_R220_260_N640` checkpoint.
It is not a useful gate for this runner because the same state is not strict
under the current local-Mdot production residual:

- `final_full = 0.708`
- source-plus-buffer old midpoint max: `0.691`
- ODE-integral max: `0.720`

So this run should be treated as a failed/invalid sanity input, not as a
physics conclusion.  A clean no-source/no-wind local-Mdot checkpoint needs to
be rebuilt if we want this exact source-plus-buffer identity sanity test.

## Updated Next Step

Do not lower `eta_E` yet.  The next production change should be an
implicit-slope source-band collocation block:

```text
unknowns: g_q = dz/dlnR at source-band quadrature points
rows:     A_q g_q + c_q = 0
rows:     z_{i+1} - z_i - integral(g_q dlnR) = 0
```

This avoids explicit `g = -A^{-1} c` as a production row while still forcing
the endpoint/poly view and differential equation view to agree.

## Addendum: Implicit-Slope Source-Band Prototype

Implemented a guarded local implicit-slope corrector in
`scripts/run_mdot5_local_mdot_eta_continuation.py`.

New flags:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_IMPLICIT_SLOPE_CORRECT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_IMPLICIT_MAX_NFEV`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_IMPLICIT_NODE_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_IMPLICIT_MID_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_IMPLICIT_INTEGRAL_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_IMPLICIT_MASS_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_IMPLICIT_PRODUCTION_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_IMPLICIT_SLOPE_BOUND`

Formulation:

- unknown source-plus-buffer states: `logu_i`, `logT_i`, `logMdot_i`;
- unknown node slopes: `g_i = d(logu,logT)/dlnR`;
- unknown midpoint slopes: `g_mid,i`;
- node rows: `A_i g_i + c_i = 0`;
- midpoint rows: `A_mid g_mid + c_mid = 0`;
- Simpson endpoint rows:

```text
z_{i+1} - z_i - h/6 (g_i + 4 g_mid,i + g_{i+1}) = 0
```

- finite-volume mass rows and optional production rows are included as guards;
- strict accepted states are protected by the existing source-plus-buffer
  full-residual guard unless explicitly disabled.

The corrector now uses a sparse finite-difference Jacobian pattern.  It is
still noticeably expensive, but usable for small source-band pilots.

### Strict-Guard Pilot

Run:

- `outputs/tables/m5_source_plus_buffer_implicit_pilot12_eta100_N164.*`
- `max_nfev = 12`
- production weight = `1`
- strict accepted-state preservation enabled

Result:

- initial implicit selected residual: `5.060`
- candidate implicit score: `0.1067`
- candidate global full residual: `2.181e-2`
- final applied: `false`
- final global residual remains strict: `9.354e-6`

Line-search diagnosis:

- `alpha = 1` gives good implicit rows:
  - implicit ODE max `6.87e-2`
  - implicit integral max `9.99e-3`
  - FV mass `1.21e-2`
  - but production full `2.18e-2`
- shrinking `alpha` restores production residual, but the implicit ODE defect
  rises back toward `5`.
- no accepted line-search point both preserves strict production residual and
  meaningfully improves the implicit block.

### Production-Weighted Strict Pilot

Run:

- `outputs/tables/m5_source_plus_buffer_implicit_pilot12_prod50_eta100_N164.*`
- production weight = `50`

Result:

- applied: `true`, but only at `alpha = 0.001953125`
- final global residual: `9.549e-6`, still strict
- implicit ODE max: `5.060 -> 5.050`
- implicit integral max: `0.10958 -> 0.10937`

This is a tiny strict move, not a solution of the representation problem.

### Relaxed Scout and Global Polish

Relaxed scout:

- `outputs/tables/m5_source_plus_buffer_implicit_relaxed_scout2_eta100_N164.*`
- strict accepted-state preservation disabled;
- full guard relaxed to allow the implicit-compatible state.

Result:

- final global residual: `2.181e-2`
- not accepted
- it confirms a nearby local implicit-slope state exists, but it leaves the
  production manifold.

Global polish from that relaxed scout:

- `outputs/tables/m5_source_plus_buffer_implicit_relaxed_global_polish_eta100_N164.*`
- implicit corrector off;
- `max_nfev = 120`.

Result:

- global residual improves from `2.181e-2` to `8.311e-5`;
- still not accepted;
- dominant residual: `interval_R` near `R ~= 250.45 rg`;
- identity ODE-integral max worsens slightly to `15.124`;
- `A g_old + c` remains `~10.04`;
- solver reports `success = false` at `nfev = 120`.

Interpretation:

- The implicit-slope rows are not failing from explicit inversion.  They can be
  satisfied much better locally.
- But the local implicit-compatible state is not currently compatible with the
  old midpoint production residual at strict tolerance.
- A simple local corrector plus line search is therefore insufficient.
- The next real production formulation should solve the global production rows
  and implicit-slope rows in one coupled augmented system, or introduce a
  genuine source-band micro-domain whose state polynomial is defined by the
  implicit slopes from the start.
