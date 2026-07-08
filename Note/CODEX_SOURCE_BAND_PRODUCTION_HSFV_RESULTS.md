# Source-Band Production HS/FV Row Replacement Results

Date: 2026-07-07

## Context

The previous source-band identity audits showed that the eta_E=100,
Mdot_inner/Edd=5, Rout=335 rg, Rinj=240 rg, f_s=0.80 checkpoint could be
strict under the legacy midpoint residual while remaining inconsistent under
endpoint-compatible ODE-integral views inside the compact source band.

The goal of this sprint was to move from audits toward a production source-band
formulation:

- finite-volume mass rows in the source band;
- implicit Hermite-Simpson/Lobatto radial and energy rows using auxiliary
  slopes instead of explicit `g = -A^{-1} c`;
- midpoint state variables inside the source band;
- checkpoint persistence for the auxiliary source-band representation.

## Implementation Changes

Updated `scripts/run_mdot5_local_mdot_eta_continuation.py`.

Added/extended:

- `SOURCE_BAND_REPLACEMENT_MIDPOINT_STATES`
- `SOURCE_BAND_REPLACEMENT_MIDPOINT_TRUST`
- `SOURCE_BAND_REPLACEMENT_MIDPOINT_WEIGHT`
- `SOURCE_BAND_REPLACEMENT_IMPLICIT_SEED=checkpoint_hs`
- `SOURCE_BAND_REPLACEMENT_IMPLICIT_SEED=checkpoint_replacement`
- source-band replacement dependency caching;
- source-band replacement midpoint-state trial layout;
- checkpoint loading from saved HS aux arrays;
- checkpoint loading from saved production replacement aux arrays;
- checkpoint writing of production replacement aux arrays:
  - `source_band_replacement_aux_interval_indices`
  - `source_band_replacement_aux_node_indices`
  - `source_band_replacement_aux_midpoint_y`
  - `source_band_replacement_aux_g_node`
  - `source_band_replacement_aux_g_mid`
- Markdown/JSON metadata field:
  - `source_band_replacement_checkpoint_aux_source`

## Key Diagnostic Lessons

### 1. Explicit Hermite rows are not usable here

A direct deterministic Hermite-Simpson evaluation using explicit
`g = -A^{-1} c` was catastrophically unstable in the source band:

- state rows reached order `1e54`;
- mass rows reached order `1e88`.

This confirms the GPT diagnosis: the source-band production method should not
invert the local ODE matrix explicitly in badly conditioned cells.

### 2. HS aux seeding is essential

Using the old profile/regularized slope seed gave poor replacement residuals.

Evaluate-only from the saved HS endpoint-release checkpoint with midpoint
states and matching halo:

```text
output:
outputs/tables/m5_source_band_production_replace_eval_hsseed_midpoint_eta100_N164.json

N = 164
source_band_replacement_n_intervals = 54
source_band_replacement_n_variables = 491
source_band_replacement_n_rows = 826
source_band_replacement_initial_score = 9.430931e-04
source_band_replacement_initial_fv_mass = 7.462366e-06
source_band_replacement_initial_implicit_ode = 8.485612e-05
source_band_replacement_initial_midpoint = 9.430931e-04
source_band_replacement_initial_simpson = 5.593352e-04
source_band_replacement_initial_outside_old = 5.779814e-06
```

This exactly reproduces the endpoint-release HS score, meaning the production
replacement residual now agrees with the HS endpoint-release representation.

### 3. Halo/grid compatibility matters

An initial seed-only polish accidentally used the default halo width
`SOURCE_PLUS_BUFFER_HALO_INTERVALS=4` instead of the HS aux halo width 32.
That changed the replacement band from 54 intervals to 18 intervals, so the
checkpoint aux arrays could not be used and the run fell back to a bad seed:

```text
output:
outputs/tables/m5_source_band_production_replace_midpoint_seedonly20_eta100_N164.json

source_band_replacement_n_intervals = 18
source_band_replacement_initial_score = 1.457818
source_band_replacement_final_score = 1.667224e-02
source_band_replacement_initial_midpoint = 0
```

This is not a physics result. It is a useful guardrail: source-band replacement
checkpoints are only meaningful when the interval/node set matches.

### 4. Production HS/FV local replacement succeeds with matching halo

With `N=164`, `eta_E=100`, `SOURCE_PLUS_BUFFER_HALO_INTERVALS=32`,
`SOURCE_BAND_REPLACEMENT_MIDPOINT_STATES=1`, and HS aux seeding:

```text
output:
outputs/tables/m5_source_band_production_replace_midpoint_persist_eta100_N164.json
checkpoint:
outputs/checkpoints/m5_source_band_production_replace_midpoint_persist_eta100_N164/stage_00_etaE_100_N164.npz

source_band_replacement_checkpoint_aux_source = source_band_hs
source_band_replacement_n_intervals = 54
source_band_replacement_n_variables = 491
source_band_replacement_n_rows = 826
source_band_replacement_initial_score = 9.430931e-04
source_band_replacement_final_score = 7.678043e-06
source_band_replacement_final_fv_mass = 7.678043e-06
source_band_replacement_final_implicit_ode = 1.048639e-08
source_band_replacement_final_midpoint = 6.978203e-08
source_band_replacement_final_simpson = 3.515901e-08
source_band_replacement_final_interface = 1.653431e-07
source_band_replacement_nfev = 3
```

The strict residual floor is now the FV mass row at `7.68e-6`; the implicit ODE,
midpoint, Simpson, and interface pieces are all much smaller.

### 5. The result is reproducible from checkpoint

Reloading the persisted production checkpoint with
`SOURCE_BAND_REPLACEMENT_IMPLICIT_SEED=checkpoint_replacement` gives:

```text
output:
outputs/tables/m5_source_band_production_replace_reload_eval_eta100_N164.json

source_band_replacement_checkpoint_aux_source = source_band_replacement
source_band_replacement_initial_score = 7.678043e-06
source_band_replacement_initial_fv_mass = 7.678043e-06
source_band_replacement_initial_implicit_ode = 1.048639e-08
source_band_replacement_initial_midpoint = 6.978203e-08
source_band_replacement_initial_simpson = 3.515901e-08
source_band_replacement_initial_outside_old = 5.779814e-06
```

So the strict production-row state is no longer a transient optimizer artifact.

### 6. Augmented global HS/FV source-band residual path

Added an opt-in augmented global source-band replacement corrector:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_GLOBAL_REPLACEMENT=1
```

The augmented unknown vector is:

```text
full disk x
+ source-band midpoint states
+ source-band node slopes
+ source-band midpoint slopes
```

The residual contains old/legacy rows outside the source band and HS/FV rows
inside the source band. This is not yet the default global continuation solver,
but it is now a real solver-facing square/near-square residual path rather than
only a local source-band audit.

Evaluate-only from the persisted production checkpoint:

```text
output:
outputs/tables/m5_source_band_global_replace_eval_eta100_N164.json

source_band_global_replacement_checkpoint_aux_source = source_band_replacement
source_band_global_replacement_n_variables = 820
source_band_global_replacement_n_rows = 826
source_band_global_replacement_initial_score = 7.678043e-06
source_band_global_replacement_initial_outside_old = 5.779814e-06
source_band_global_replacement_initial_fv_mass = 7.678043e-06
source_band_global_replacement_initial_implicit_ode = 1.048639e-08
source_band_global_replacement_initial_midpoint = 6.978203e-08
source_band_global_replacement_initial_simpson = 3.515901e-08
```

This confirms that the strict local source-band production checkpoint is also
strict under the augmented global HS/FV residual view.

### 7. Practical augmented-solve behavior

A first forced augmented polish with the energy audit still enabled was
interrupted because finite-difference Jacobian evaluation spent most of its time
in expensive source-element FV-energy audit calls, even though FV-energy rows
were not active. The code now has:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_ENERGY_AUDIT=0
```

With energy audit disabled and accepted-seed skipping enabled:

```text
output:
outputs/tables/m5_source_band_global_replace_skipaccepted_eta100_N164.json

source_band_global_replacement_initial_score = 7.678043e-06
source_band_global_replacement_final_score = 7.678043e-06
source_band_global_replacement_nfev = 0
source_band_global_replacement_message = initial_score_below_accept_tol
```

Forcing a small augmented polish with `MAX_NFEV=3` and skip-accepted disabled:

```text
output:
outputs/tables/m5_source_band_global_replace_forced3_eta100_N164.json

source_band_global_replacement_initial_score = 7.678043e-06
source_band_global_replacement_candidate_score = 7.634029e-06
source_band_global_replacement_final_score = 7.634029e-06
source_band_global_replacement_final_outside_old = 5.769236e-06
source_band_global_replacement_final_fv_mass = 7.634029e-06
source_band_global_replacement_final_implicit_ode = 8.733219e-09
source_band_global_replacement_final_midpoint = 3.204246e-08
source_band_global_replacement_final_simpson = 1.634622e-08
source_band_global_replacement_nfev = 3
```

The augmented solver is functional, but the improvement is small and still
limited by FV mass. This points to the next numerical bottleneck: analytic/local
Jacobian support for the FV mass row or a better mass-increment representation,
not more generic finite-difference Newton iterations.

### 8. Targeted FV-mass analytic corrector

Added an opt-in source-band FV-mass corrector:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_FV_MASS_CORRECT=1
```

It adjusts only the source-band `logMdot` nodes and uses the endpoint-exact
analytic derivative of

```text
(Mdot_R - Mdot_L - Delta Mdot_wind + Delta Mdot_stream) / sqrt(Mdot_L Mdot_R)
```

with respect to the two interval endpoint `logMdot` values. Wind/source
derivatives are not included in this first analytic Jacobian; the line search
and augmented HS/FV audit decide whether a mass-only step is compatible.

Unanchored mass solve with shallow line search:

```text
output:
outputs/tables/m5_source_band_fv_mass_correct_eta100_N164.json

candidate_fv_mass = 2.751693e-08
candidate_score = 2.310030e-02
applied = False
```

Interpretation: the analytic mass step can almost eliminate the FV mass row by
itself, but it strongly violates outside-old and implicit-ODE compatibility.
The guard correctly rejects the full step.

With deeper line search:

```text
output:
outputs/tables/m5_source_band_fv_mass_correct_linesearch16_eta100_N164.json

initial_score = 7.678043e-06
final_score = 7.676175e-06
final_fv_mass = 7.676175e-06
final_outside_old = 5.779814e-06
final_implicit_ode = 2.039823e-07
alpha = 2.441406e-04
nfev = 2
```

Anchored variants confirm the same result:

```text
all-anchor 1e-2:
outputs/tables/m5_source_band_fv_mass_correct_allanchor1em2_eta100_N164.json
final_score = 7.676285e-06

all-anchor 1e-1:
outputs/tables/m5_source_band_fv_mass_correct_allanchor1em1_eta100_N164.json
final_score = 7.677432e-06
```

Starting a short augmented global polish from the line-search mass-corrected
checkpoint improves a little more:

```text
output:
outputs/tables/m5_source_band_global_after_masscorrect_forced3_eta100_N164.json

initial_score = 7.676175e-06
candidate_score = 7.599103e-06
final_score = 7.599103e-06
final_fv_mass = 7.599103e-06
final_outside_old = 5.903776e-06
final_implicit_ode = 8.706348e-08
nfev = 3
```

This is a real but small improvement. The main lesson is that the FV mass floor
is coupled to the outside source-band interface and ODE compatibility. A
standalone `logMdot` correction is too constrained to remove the floor.

## Important Caveat

The old global residual is still large:

```text
legacy/source old residual ~ 1.126
mass_residual_max under legacy profile ~ 0.605
```

This is expected for this experiment because the old source-band midpoint rows
are being replaced. The old residual should now be treated as an audit of the
legacy formulation, not as the acceptance metric for the new source-band
production formulation.

The augmented global residual path now exists and certifies eta_E=100 at the
few-`1e-6` level, but it is not yet the default continuation method. The
remaining practical issue is not the old source-band residual itself; it is the
cost and conditioning of the augmented finite-difference Jacobian, with the
residual floor dominated by FV mass. The FV-mass analytic-corrector experiment
shows this floor is interface-coupled, not merely a missing derivative.

## Verification

```text
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache \
PYTHONPATH=src \
/Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest

160 passed in 2.93s
```

## Next Recommended Step

The next numerical infrastructure step is to make the augmented global solve
efficient enough for continuation:

1. Promote the mass equation to a source-band increment/interface formulation:
   add per-interval or cumulative mass-increment variables with endpoint
   compatibility, so the source-band mass budget can be satisfied without
   forcing incompatible pointwise `logMdot` shifts.
2. Keep the analytic endpoint derivative support for the increment
   compatibility rows.
3. Keep the old midpoint source-band rows as audit-only diagnostics.
4. Re-certify eta_E=100 with the efficient augmented path, then lower eta_E to
   90, 80, and 70.

## 2026-07-07 Update: Mass-Increment Interface Rows

Implemented interval mass-increment auxiliary variables for the source-band
HS/FV production replacement path.

New controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_MASS_INCREMENT=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_MASS_INCREMENT_MODE=interval
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_MASS_INCREMENT_INIT=endpoint|integral|zero
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_MASS_INCREMENT_SCALE=mdot_inner
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_MASS_INCREMENT_INT_WEIGHT=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_MASS_INCREMENT_LINK_WEIGHT=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_MASS_INCREMENT_BOUND=10
```

The active source-band mass row is now split into two compatible rows when the
flag is enabled:

```text
r_M_int  = DeltaM_i - integral_i(Mdot_wind_prime - Mdot_stream_prime) / Mdot_inner
r_M_link = (Mdot_{i+1} - Mdot_i) / Mdot_inner - DeltaM_i
```

The old finite-volume mass residual is retained as an audit. Checkpoints now
persist:

```text
source_band_mass_increment_aux_delta
```

### Seed-Split Sanity Test

Endpoint-initialized DeltaM:

```text
output:
outputs/tables/m5_source_band_mass_increment_eval_endpoint_eta100_N164.json

source_band_global_replacement_initial_score = 7.911037e-06
source_band_global_replacement_initial_mass_increment_int = 7.911037e-06
source_band_global_replacement_initial_mass_increment_link = 0
source_band_global_replacement_initial_implicit_ode = 1.048639e-08
source_band_global_replacement_initial_midpoint = 6.978203e-08
source_band_global_replacement_initial_simpson = 3.515901e-08
```

Integral-initialized DeltaM:

```text
output:
outputs/tables/m5_source_band_mass_increment_eval_integral_eta100_N164.json

source_band_global_replacement_initial_score = 7.911037e-06
source_band_global_replacement_initial_mass_increment_int = 0
source_band_global_replacement_initial_mass_increment_link = 7.911037e-06
source_band_global_replacement_initial_implicit_ode = 1.048639e-08
source_band_global_replacement_initial_midpoint = 6.978203e-08
source_band_global_replacement_initial_simpson = 3.515901e-08
```

This confirms that the new DeltaM split is representing the same old FV mass
budget mismatch, not introducing a new source-band inconsistency.

### Coupled Augmented Polish

A short forced augmented global polish from the production HS/FV checkpoint:

```text
output:
outputs/tables/m5_source_band_mass_increment_global_forced5_eta100_N164.json
checkpoint:
outputs/checkpoints/m5_source_band_mass_increment_global_forced5_eta100_N164/stage_00_etaE_100_N164.npz

initial_score = 7.911037e-06
candidate_score = 5.716155e-06
final_score = 5.716155e-06
final_mass_increment_int = 3.802756e-06
final_mass_increment_link = 3.881698e-06
final_implicit_ode = 1.793610e-08
final_midpoint = 1.672623e-09
final_simpson = 1.028920e-09
final_outside_old = 5.716155e-06
nfev = 5
alpha = 1
```

The source-band mass defect is now shared consistently between the integral and
endpoint-link rows. The implicit HS/FV rows remain tiny.

### Local Follow-Up

A targeted local source-band replacement restart from the `forced5` checkpoint
did not improve the solution:

```text
output:
outputs/tables/m5_source_band_mass_increment_local_forced20_eta100_N164.json

initial_score = 5.716155e-06
final_score = 5.716155e-06
alpha = 0
nfev = 20
```

The line search accepted no update. This is not a new source-band failure; the
floor had moved to the active outside-old rows.

### Outside-Floor Localization

Added audit-only outside-old row classification by legacy row type. This does
not change the residual vector.

Evaluate-only localization from the `forced5` checkpoint:

```text
output:
outputs/tables/m5_source_band_mass_increment_floor_audit_eta100_N164.json

source_band_global_replacement_initial_score = 5.716155e-06
source_band_global_replacement_initial_outside_old = 5.716155e-06
source_band_global_replacement_initial_outside_old_peak_R_rg = 30.853298
source_band_global_replacement_initial_outside_old_radial = 1.830434e-06
source_band_global_replacement_initial_outside_old_energy = 5.716155e-06
source_band_global_replacement_initial_outside_old_mass = 1.585137e-07
source_band_global_replacement_initial_mass_increment_int = 3.802756e-06
source_band_global_replacement_initial_mass_increment_link = 3.881698e-06
source_band_global_replacement_initial_implicit_ode = 1.793610e-08
source_band_global_replacement_initial_midpoint = 1.672623e-09
source_band_global_replacement_initial_simpson = 1.028920e-09
```

Interpretation: the remaining few-`1e-6` production residual is no longer a
compact source-band representation defect. It is an outside-domain legacy
energy row at `R ~= 30.85 rg`, while the source-band HS/FV rows are already
strict at or below the few-`1e-6` level.

## Revised Next Step

The eta_E=100 source-band representation is now usable as an exploratory
identity-aware production formulation. Before lowering eta_E, the best next
step is not more source-band mass surgery; it is to make the augmented global
polish efficient and to decide whether the outside-domain energy floor should
be globally polished or accepted as below the current `1e-5` exploratory
tolerance.

Recommended sequence:

1. Add local/analytic Jacobian support for the DeltaM split rows and source-band
   HS/FV rows to avoid slow global finite-difference Jacobians.
2. Run one longer augmented global polish with the local Jacobian enabled.
3. If the outside energy floor remains near `5e-6`, treat eta_E=100 as certified
   at the exploratory tolerance and continue to eta_E=90.
4. If the outside floor grows during eta continuation, add a small global polish
   or dedicated outside-energy correction around `R ~= 30.85 rg`.

## Verification After Mass-Increment Update

```text
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache \
PYTHONPATH=src \
/Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest

160 passed in 2.95s
```

## 2026-07-07 Update: Freeze-Aux Global Polish

Implemented two follow-up numerical infrastructure pieces:

1. `SOURCE_BAND_REPLACEMENT_HYBRID_JAC`
   - emits exact auxiliary-Jacobian entries for:
     - implicit ODE rows with respect to auxiliary slopes;
     - midpoint-state compatibility rows;
     - Simpson integral rows;
     - DeltaM integral/link rows;
     - optional slope-interface rows.
   - the current wrapper still finite-differences the physical source-band
     state block, so it is not yet a production speedup by itself.

2. `SOURCE_BAND_GLOBAL_REPLACEMENT_FREEZE_AUX`
   - keeps the certified source-band midpoint/slope/DeltaM auxiliary
     representation fixed;
   - optimizes only the global physical disk state;
   - reduces the eta_E=100 optimizer from 874 augmented variables to 494 global
     physical variables.

Freeze-aux evaluate-only smoke test:

```text
output:
outputs/tables/m5_source_band_freezeaux_eval_eta100_N164.json

initial_score = 5.716155e-06
initial_outside_old = 5.716155e-06
initial_outside_old_energy = 5.716155e-06
optimizer_n_variables = 494
freeze_aux = True
```

First freeze-aux polish:

```text
output:
outputs/tables/m5_source_band_freezeaux_polish8_eta100_N164.json
checkpoint:
outputs/checkpoints/m5_source_band_freezeaux_polish8_eta100_N164/stage_00_etaE_100_N164.npz

initial_score = 5.716155e-06
candidate_score = 4.259085e-06
final_score = 4.259085e-06
final_outside_old = 4.259085e-06
final_outside_old_energy = 4.259085e-06
final_outside_old_peak_R_rg = 31.934463
final_mass_increment_int = 3.802756e-06
final_mass_increment_link = 3.881102e-06
final_implicit_ode = 2.546671e-07
final_midpoint = 4.297066e-07
final_simpson = 4.266397e-07
alpha = 1
nfev = 8
```

Second freeze-aux polish:

```text
output:
outputs/tables/m5_source_band_freezeaux_polish16_eta100_N164.json
checkpoint:
outputs/checkpoints/m5_source_band_freezeaux_polish16_eta100_N164/stage_00_etaE_100_N164.npz

initial_score = 4.259085e-06
candidate_score = 4.202018e-06
final_score = 4.202018e-06
final_outside_old = 4.202018e-06
final_outside_old_energy = 4.202018e-06
final_outside_old_peak_R_rg = 31.934461
final_mass_increment_int = 3.802756e-06
final_mass_increment_link = 3.881196e-06
final_implicit_ode = 9.537768e-08
final_midpoint = 1.323996e-07
final_simpson = 1.317595e-07
alpha = 1
nfev = 3
termination = xtol
```

Interpretation:

- The previous `5.716e-6` floor was not a fixed physical/source-band
  obstruction. It was polishable by global physical-state motion while holding
  the source-band representation fixed.
- After two freeze-aux polishes, eta_E=100 is comfortably strict under the
  identity-aware production HS/FV residual: `final_score = 4.202e-6`.
- The active floor is still outside-domain energy, now near `R ~= 31.93 rg`.
- The source-band DeltaM split is stable at `~3.8e-6`, and implicit HS/FV rows
  remain below `~1.4e-7`.
- A short local hybrid-Jacobian validation run was stopped because the wrapper
  still finite-differences too much of the physical source-band state block.
  Keep the exact auxiliary entries, but do not use this wrapper as the
  production speed path until it is row-local or block-colored.

## Revised Next Step After Freeze-Aux

The eta_E=100 checkpoint

```text
outputs/checkpoints/m5_source_band_freezeaux_polish16_eta100_N164/stage_00_etaE_100_N164.npz
```

is the best current identity-aware source-band production anchor.

Recommended next sequence:

1. Use freeze-aux global polish as the default eta_E=100 certification step.
2. Continue to eta_E=90 from the `freezeaux_polish16` checkpoint with:
   - source-band mass increment enabled;
   - checkpoint replacement aux seed;
   - freeze-aux first;
   - then, only if needed, one short full augmented polish.
3. Treat the current hybrid-Jacobian wrapper as diagnostic only. A true
   production Jacobian should be row-local/block-colored for the physical
   source-band state block, not naive per-column finite difference.

## Verification After Freeze-Aux Update

```text
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache \
PYTHONPATH=src \
/Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest

160 passed in 3.10s
```

## 2026-07-07 Update: Eta Continuation Attempts Below 100

Tested continuation from the eta_E=100 identity-aware anchor:

```text
outputs/checkpoints/m5_source_band_freezeaux_polish16_eta100_N164/stage_00_etaE_100_N164.npz
```

### Direct eta_E=90

Evaluate-only with frozen aux:

```text
output:
outputs/tables/m5_source_band_eta90_freezeaux_eval_N164.json

initial_score = 1.647210e-03
initial_outside_old = 1.647210e-03
initial_outside_old_mass = 1.647210e-03
initial_outside_old_energy = 4.202018e-06
initial_mass_increment_int = 3.854404e-06
initial_mass_increment_link = 3.881196e-06
initial_midpoint = 1.323996e-07
initial_simpson = 1.317595e-07
```

Short freeze-aux polish:

```text
output:
outputs/tables/m5_source_band_eta90_freezeaux_polish8_N164.json

initial_score = 1.647210e-03
candidate_score = 2.423671e-03
final_score = 1.211213e-03
final_outside_old_mass = 9.354427e-04
final_mass_increment_int = 1.313820e-05
final_mass_increment_link = 6.197504e-06
final_implicit_ode = 7.169563e-05
final_midpoint = 3.200602e-04
final_simpson = 3.190706e-04
alpha = 0.5
nfev = 8
```

Interpretation: eta_E=90 is too large a jump. Freeze-aux reduces the outside
mass row but moves the physical state far enough that the frozen source-band
compatibility rows become non-strict.

### eta_E=95

Evaluate-only:

```text
output:
outputs/tables/m5_source_band_eta95_freezeaux_eval_N164.json

initial_score = 7.802334e-04
initial_outside_old_mass = 7.802334e-04
initial_mass_increment_int = 3.827221e-06
initial_mass_increment_link = 3.881196e-06
initial_midpoint = 1.323996e-07
initial_simpson = 1.317595e-07
```

Freeze-aux polish:

```text
output:
outputs/tables/m5_source_band_eta95_freezeaux_polish8_N164.json

initial_score = 7.802334e-04
final_score = 5.749765e-04
final_outside_old_mass = 4.432975e-04
final_mass_increment_int = 7.476886e-06
final_mass_increment_link = 4.204022e-06
final_implicit_ode = 3.373281e-05
final_midpoint = 1.405337e-04
final_simpson = 1.402587e-04
alpha = 0.5
nfev = 8
```

Full augmented polish at eta_E=95:

```text
output:
outputs/tables/m5_source_band_eta95_fullaug_polish5_N164.json

initial_score = 7.802334e-04
final_score = 6.580702e-04
final_outside_old_mass = 6.226348e-04
final_mass_increment_int = 3.882034e-05
final_mass_increment_link = 3.902874e-05
final_implicit_ode = 1.195033e-05
final_midpoint = 1.737876e-07
final_simpson = 2.086158e-07
alpha = 0.25
nfev = 5
```

Interpretation: releasing auxiliaries keeps midpoint/Simpson strict but does
not solve the outside mass row, and it pushes the DeltaM split rows up to
`~4e-5`.

### Mass-Only Eta Predictor

Implemented an opt-in eta mass-profile predictor:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_MASS_PROFILE_PREDICTOR=1
IMBH_MDOT5_LOCAL_MDOT_ETA_MASS_PROFILE_PREDICTOR_DAMPING=<0..1>
IMBH_MDOT5_LOCAL_MDOT_ETA_MASS_PROFILE_PREDICTOR_SWEEPS=<int>
```

It propagates the finite-volume mass equation outward at the target eta_E,
holding `u,T` fixed.

At eta_E=95 with full damping:

```text
output:
outputs/tables/m5_source_band_eta95_masspredict_eval_N164.json

eta_mass_predictor_before_fv_mass_max = 3.668784e-04
eta_mass_predictor_after_fv_mass_max = 1.027703e-05
eta_mass_predictor_delta_logMdot_max = 8.457219e-02
source_band_global_replacement_initial_score = 1.196530
source_band_global_replacement_initial_implicit_ode = 1.196530
```

At eta_E=95 with damping 0.1:

```text
output:
outputs/tables/m5_source_band_eta95_masspredict_damp01_eval_N164.json

eta_mass_predictor_before_fv_mass_max = 3.668784e-04
eta_mass_predictor_after_fv_mass_max = 3.304165e-04
eta_mass_predictor_delta_logMdot_max = 8.133084e-03
source_band_global_replacement_initial_score = 1.052936e-01
source_band_global_replacement_initial_implicit_ode = 1.052936e-01
```

Interpretation: mass-only prediction is not usable. It can improve a standalone
FV mass audit, but changing `logMdot(R)` without a coupled `u,T` predictor badly
violates the implicit ODE rows.

### eta_E=99

Evaluate-only:

```text
output:
outputs/tables/m5_source_band_eta99_freezeaux_eval_N164.json

initial_score = 1.497050e-04
initial_outside_old_mass = 1.497050e-04
initial_mass_increment_int = 3.807451e-06
initial_mass_increment_link = 3.881196e-06
initial_midpoint = 1.323996e-07
initial_simpson = 1.317595e-07
```

First freeze-aux polish:

```text
output:
outputs/tables/m5_source_band_eta99_freezeaux_polish8_N164.json
checkpoint:
outputs/checkpoints/m5_source_band_eta99_freezeaux_polish8_N164/stage_00_etaE_99_N164.npz

initial_score = 1.497050e-04
final_score = 1.129653e-04
final_outside_old_mass = 1.129653e-04
final_mass_increment_int = 3.807451e-06
final_mass_increment_link = 3.888818e-06
final_implicit_ode = 1.449031e-06
final_midpoint = 3.089960e-06
final_simpson = 3.227868e-06
alpha = 0.25
nfev = 8
```

Second freeze-aux polish:

```text
output:
outputs/tables/m5_source_band_eta99_freezeaux_polish16_N164.json
checkpoint:
outputs/checkpoints/m5_source_band_eta99_freezeaux_polish16_N164/stage_00_etaE_99_N164.npz

initial_score = 1.129653e-04
final_score = 9.945201e-05
final_outside_old_mass = 9.945201e-05
final_mass_increment_int = 3.807451e-06
final_mass_increment_link = 3.891518e-06
final_implicit_ode = 1.873432e-06
final_midpoint = 4.158303e-06
final_simpson = 4.332151e-06
alpha = 0.125
nfev = 8
```

Interpretation: even a 1% eta step is not robustly strict with the current
predictor/corrector. It makes progress, but line search shrinks and the outside
mass row remains at `~1e-4`.

## Current Diagnosis

The eta_E=100 source-band HS/FV representation is certified at the
few-`1e-6` level. The new bottleneck for eta continuation is not source-band
representation. It is the global response of the physical disk state to the
changed wind mass loading, especially the outside-domain mass profile.

Freeze-aux is useful for same-eta polishing, but not as a predictor for eta
changes. Mass-only prediction is worse because it violates ODE compatibility.

## Revised Plan

Before continuing to eta_E=90, implement a coupled eta tangent predictor:

1. Build a bordered/linearized predictor for the physical global state:
   solve approximately
   `J_x dx/deta = -F_eta`
   using the existing square/global residual and the source-band replacement
   residual view.
2. Include at least `logu`, `logT`, and `logMdot`; do not predict only
   `logMdot`.
3. Keep source-band aux either:
   - frozen during the predictor and released in a short corrector, or
   - included with exact aux-Jacobian entries.
4. Retry eta ladder with small steps:
   `100 -> 99.5 -> 99 -> 98 -> ...`
   and only increase step size after the outside mass row stays below
   `~1e-5`.

## Verification After Eta-Continuation Update

```text
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache \
PYTHONPATH=src \
/Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest

160 passed in 3.08s
```

## 2026-07-07 Update: Coupled Eta Tangent Predictor

Implemented the next eta-continuation layer in
`scripts/run_mdot5_local_mdot_eta_continuation.py`. Details and run outputs are
summarized in:

```text
Note/CODEX_MDOT5_ETA_TANGENT_CONTINUATION_RESULTS.md
```

Main result: in the compatible source-band mass-increment/global-replacement
view, direct eta changes are dominated by outside mass residuals, while the new
coupled `mu = 1 / eta_E` tangent predictor removes that broad mass mismatch.

Best strict micro-step ladder:

```text
outputs/tables/m5_eta_microstep_tangent_eval_massview_reg1em5_N164.json
```

The ladder remains strict through eta_E=99.85 with `dmu ~= 5e-6`. Larger steps
are not strict yet because the tangent exposes a near-inner/global row around
R ~= 5.30 rg. A full freeze-aux nonlinear corrector is still too slow with the
current finite-difference wrapper.

Important caveat: forcing full-weight source-band implicit HS/FV rows with
`SOURCE_BAND_CHI_IMPL=1` is not compatible with the current eta_E=100 checkpoint
and gives order-unity active residuals. The tangent result therefore certifies
the compatible current production residual view, not a full unit-weight
implicit source-band collocation view.
