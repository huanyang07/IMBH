# Codex Review: Outer-Buffer Source-Fraction Continuation Next Step

Date: 2026-07-03

Repository reviewed: `huanyang07/IMBH`, current public `main`.

Primary files/results reviewed:

```text
Note/CODEX_COMPACT_SOURCE_OUTER_BUFFER_RESULTS.md
Note/GPT_PROMPT_OUTER_BUFFER_SOURCE_FRACTION_NEXT.md
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
scripts/run_standard_slim_stream_mass_annulus_scan.py
outputs/tables/high_mdot_stream_compact_outer_buffer_ladder_validation_profile.md
outputs/tables/high_mdot_stream_compact_outer_buffer_fs08625_to090_N896.md
```

## 1. Executive diagnosis

The current `f_s ~ 0.876` bottleneck should be treated as a **predictor / remesh / source-interface cost wall**, not as evidence for a physical source-fraction endpoint.

The outer-buffer formulation has done what it was supposed to do:

```text
Mdot_inner/Edd = 2
Rout = 335 rg
Rinj = 240 rg
R_buffer = 300 rg
source = compact_c2
torque_delta_l_fraction = +0.005
no wind
no stream heating
buffer weights (R,E,B) = (1e-3, 1e-3, 1e-4)
```

The `f_s=0.8625` reservoir branch passes first-pass robustness checks across
`N=768/896/1024`, `R_buffer=295/300/305`, and stricter buffer weights. Source-fraction continuation then reaches the strict clean anchor:

```text
f_s = 0.8759639587
final weighted full residual = 1.242e-07
physical raw energy residual max = 1.473e-05
buffer raw energy residual max = 1.669e-01
f_adv_global = 0.20422
f_adv_inner = 0.09443
Lrad/LEdd = 0.86717
Rson = 4.65992 rg
```

The global physics diagnostics are smooth. The raw buffer residual remains large by construction and should not be used as the physical convergence metric. The relevant audit is the split physical/source-domain residual.

## 2. Important code-level observation

`run_standard_slim_stream_mass_annulus_scan.py` already contains an opt-in source-fraction tangent predictor infrastructure:

```text
IMBH_STANDARD_SLIM_STREAM_MASS_USE_TANGENT_PREDICTOR
IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_DAMPINGS
IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_FD_STEP
IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_SOLVER
IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_LINEAR_DAMPING
IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_MAXITER
```

and functions equivalent to:

```python
finite_difference_source_column(...)
equilibrated_tangent_solve(...)
source_fraction_tangent(...)
source_fraction_seed(...)
```

So the next step is **not** to implement `J_z dz/df_s = -F_{f_s}` from scratch. The next step is to enable it, harden the diagnostics around it, and run a controlled A/B pilot against the current predictor.

The latest table shows the continuation was still using `predictor=current` for the expensive final stretch. That is exactly the regime where the tangent predictor should help.

## 3. Recommended priority order

### Priority 0 — Fix reproducibility metadata before more expensive runs

The latest output table has confusing header metadata: the header reports fallback/default values such as `source shape tanh` and `torque fraction 0`, while the row data and current status indicate `compact_c2` and `torque fraction 0.005`.

Before running another long continuation, update the table header and JSON metadata to report the **effective parameters after checkpoint/default inheritance**, not only the environment-variable defaults.

Add header/JSON fields:

```text
effective_source_shape
effective_source_shape_blend
effective_torque_fraction
effective_Rinj_rg
effective_torque_Rinj_rg
effective_outer_buffer_inner_rg
effective_buffer_weights
effective_outer_closure
anchor_checkpoint
anchor_source_fraction
anchor_Rout_rg
```

This is not a physics blocker, but it prevents future handoff confusion.

### Priority 1 — Run a tangent-predictor A/B pilot from `f_s=0.8759639587`

Use the latest strict anchor in:

```text
outputs/checkpoints/high_mdot_stream_compact_outer_buffer_fs08625_to090_N896/
```

Run three short pilots to `f_s=0.878` or `0.880`, not all the way to `0.90` yet:

```text
A. baseline current predictor
B. secant predictor enabled
C. tangent + secant predictors enabled
```

Suggested environment for the tangent pilot:

```bash
export IMBH_STANDARD_SLIM_STREAM_MASS_USE_SECANT_PREDICTOR=1
export IMBH_STANDARD_SLIM_STREAM_MASS_USE_TANGENT_PREDICTOR=1
export IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_DAMPINGS=1,0.5,0.25,0.1,0.05
export IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_FD_STEP=1e-5
export IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_SOLVER=equilibrated_lsmr
export IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_LINEAR_DAMPING=0
export IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_INITIAL_STEP=5e-4
export IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_MIN_STEP=1.25e-4
export IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_EVERY_STEP=1
export IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_ON_REJECT=1
export IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_INNER_RG=300
export IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_RADIAL_WEIGHT=1e-3
export IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_ENERGY_WEIGHT=1e-3
export IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_BOUNDARY_WEIGHT=1e-4
export IMBH_STANDARD_SLIM_STREAM_MASS_INTERVAL_FORM=integrated
```

Success criterion for the tangent pilot:

```text
median nfev per accepted step reduced by >= 2x relative to predictor=current,
OR accepted step size grows above 2.5e-4 without degrading split-audit residuals.
```

Do not judge by final weighted residual alone. Judge by:

```text
predictor initial residual
final weighted residual
physical/source-domain raw E max
physical/source-domain E L2
mass/source budget error
nfev total
whether remesh was adopted
f_adv_global, f_adv_inner, Lrad/LEdd, Rson
```

### Priority 2 — Add predictor diagnostics, not just the chosen predictor label

At the moment, the table records the chosen predictor and one `predictor_initial_full`. Add a candidate-predictor audit so every attempted step reports:

```text
initial_full_current
initial_full_secant_best
initial_full_tangent_best
chosen_predictor
tangent_damping_chosen
tangent_fd_step
tangent_solver
tangent_linear_damping
tangent_norm_inf
tangent_norm_l2
tangent_linear_residual_norm = ||J dz_df + F_f||
tangent_secant_cosine, if a secant exists
step_df
predicted_state_clip_count
```

This turns the next run into a real numerical diagnosis. If tangent does not help, Codex will know whether the problem is a bad finite-difference source column, an ill-conditioned Jacobian solve, clipping/bounds, or a genuine fold-like loss of parameter continuation.

### Priority 3 — If tangent helps but physical residual spikes persist, do a local source/interface remesh

The remaining intermittent physical-domain spikes occur around the compact source / inner-buffer transition rather than the terminal outer boundary. Do **not** globally chase the raw buffer residual. Instead, add a physical-domain remesh monitor that emphasizes:

```text
R < R_buffer only for physical convergence;
|physical interval_E|;
source_prime and d(source_prime)/dlnR;
|dMdot/dlnR|;
|dQstream/dlnR|;
a narrow window around Rinj = 240 rg;
a narrow window around R_buffer = 300 rg.
```

Operational rule:

```text
If partition_peak_physical_E_rg lies within the compact source window
or within ~5-10 rg of R_buffer, trigger source/interface remeshing.
Otherwise do not spend grid nodes chasing buffer raw E.
```

Preserve:

```text
exact Rout;
exact R_buffer location if possible;
source normalization;
mass budget;
Rinj and torque center in physical rg;
smooth remap of logu/logT;
pre/post split-audit table.
```

### Priority 4 — Use pseudo-arclength only as a wall classifier, not the first production method

There is not yet evidence for a physical fold. Global diagnostics remain smooth and the old `f_s~0.86` wall was crossed by better continuation/remeshing.

Implement pseudo-arclength only if the tangent pilot shows one of these:

```text
1. tangent norm grows rapidly over consecutive accepted steps;
2. tangent and secant directions become nearly orthogonal or flip sign;
3. the Jacobian/tangent linear solve becomes ill-conditioned;
4. accepted steps remain stuck at df_s <= 1.25e-4 despite tangent prediction;
5. failures persist after local source/interface remesh.
```

Minimal pseudo-arclength formulation:

```text
Unknowns: (z, f_s)
Residuals: square_collocation_residual(z; f_s) plus arclength constraint
Constraint: <W_z (z-z0), W_z dz_tan> + w_f (f_s-f0) - ds = 0
```

Use it for a short branch audit over `f_s ~ 0.875-0.885`, not as the default long-run method unless it clearly outperforms the tangent predictor.

### Priority 5 — Defer a more explicit two-domain reservoir

A two-domain reservoir is scientifically cleaner, but it is not the cost-optimal next step. The current reservoir formulation is acceptable as a controlled finite-reservoir boundary **provided the split audit is always reported** and the physical/source-domain residual remains stable.

Move to a more explicit two-domain reservoir only if:

```text
R_buffer sensitivity reappears near f_s -> 0.90;
source/interface remeshing cannot stabilize physical residuals;
publication-quality boundary interpretation becomes the blocker;
or Mdot_inner/Edd = 3/5 runs reveal reservoir-coupling artifacts.
```

## 4. Acceptance criteria

Use two acceptance tiers.

### 4.1 Computational continuation anchor

Allowed for carrying the branch forward:

```text
final weighted full <= 1e-5
mass/source relative budget error <= 3e-4
positive state and valid sonic solution
no terminal weighted residual domination
physical diagnostics remain smooth
```

### 4.2 Physics/reporting anchor

Required for claims in notes/papers:

```text
final weighted full <= 3e-6 preferred
physical/source-domain raw E max <= 3e-5 preferred
physical/source-domain raw E max <= 1e-4 acceptable only with N/R_buffer/weight validation
mass/source relative budget error <= 3e-4
f_adv_global stable to <1%
f_adv_inner stable to <1-2%
Lrad/LEdd stable to <1%
Rson stable to <1e-2 rg
N = 768/896/1024 check for selected anchors
R_buffer = 295/300/305 check for selected anchors
buffer weights baseline and stricter-weight check for selected anchors
```

The raw buffer residual is recorded but not used as the physical convergence criterion.

## 5. Criteria for declaring a physical source-fraction wall

Do **not** call a wall physical merely because Newton/remeshing becomes expensive.

A physical wall requires all of the following:

```text
1. tangent predictor, secant predictor, and current-state predictor all fail;
2. local source/interface remeshing fails at multiple N;
3. pseudo-arclength cannot pass the point or reveals a reproducible fold;
4. failure localizes in the physical/source domain, not only in the softened buffer;
5. the peak physical residual location converges with N and does not collapse into one unresolved cell;
6. the result is insensitive to R_buffer = 295/300/305 and stricter buffer weights;
7. global diagnostics show a coherent physical trend, not random residual spikes;
8. the mass/source budget remains closed up to the wall.
```

If the failure remains dominated by one physical/source-interface cell or by raw buffer residual, it is still numerical.

## 6. Concrete first 3 tasks for Codex

### Task 1 — Metadata + candidate-predictor audit

Patch `scripts/run_standard_slim_stream_mass_annulus_scan.py` to:

```text
- report effective inherited source shape and torque in table/JSON headers;
- record all candidate predictor initial residuals;
- record tangent norm, tangent linear residual, damping chosen, and clip count;
- keep current fallback behavior if tangent fails.
```

### Task 2 — Tangent A/B pilot

Run short branches from the `f_s=0.8759639587` checkpoint:

```text
baseline current predictor: f_s -> 0.878 or 0.880
secant predictor:          f_s -> 0.878 or 0.880
tangent+secant predictor:  f_s -> 0.878 or 0.880
```

Compare cost and split-audit residuals. If tangent reduces cost, continue to `f_s=0.90` using tangent+secant with residual remesh.

### Task 3 — Source/interface remesh trigger

Add a targeted remesh mode that activates when `partition_peak_physical_E_rg` lies near the compact source annulus or the inner buffer boundary. The remesh should focus on physical/source-domain residuals, not raw buffer residuals.

## 7. Proposed execution plan

This plan is designed to push the branch from

```text
f_s = 0.8759639587
```

toward

```text
f_s = 0.90
```

while minimizing wasted Newton/remesh cost and preserving a clean distinction between numerical and physical failure.

### Phase A — Freeze the accepted anchor and make the run reproducible

Goal: make sure any new failure can be compared against the same trusted starting point.

Codex should first create a small anchor manifest for the accepted checkpoint:

```text
anchor_name = outer_buffer_compact_c2_Rout335_Rbuffer300_fs08759639587
Mdot_inner/Edd = 2
Rout = 335 rg
Rinj = 240 rg
R_buffer = 300 rg
N = 896
source = compact_c2
torque_delta_l_fraction = +0.005
buffer weights = (1e-3, 1e-3, 1e-4)
final weighted full residual = 1.242e-07
physical raw energy residual max = 1.473e-05
f_adv_global = 0.20422
f_adv_inner = 0.09443
Lrad/LEdd = 0.86717
Rson = 4.65992 rg
```

Implementation notes:

```text
- Do not overwrite the accepted checkpoint.
- Copy or symlink it into a clearly named pilot directory.
- Record the git commit hash, environment variables, Python version, and solver tolerances.
- Add effective inherited parameters to both markdown and machine-readable JSON outputs.
```

Exit criteria:

```text
- Re-running a zero-step or one-step restart from the anchor reproduces the split audit.
- Output metadata reports compact_c2 and torque_delta_l_fraction=+0.005 correctly.
```

### Phase B — Patch diagnostics before changing the numerical method

Goal: make every attempted continuation step explain why it succeeded or failed.

Add candidate-predictor diagnostics to the scan output. At minimum, every attempted step should report:

```text
f_s_start
f_s_trial
step_df
chosen_predictor
initial_full_current
initial_full_secant_best
initial_full_tangent_best
tangent_damping_chosen
tangent_norm_inf
tangent_norm_l2
tangent_linear_residual_norm
predicted_state_clip_count
newton_success
newton_nfev
newton_final_weighted_full
physical_raw_E_max
physical_raw_E_L2
partition_peak_physical_E_rg
partition_peak_buffer_E_rg
mass_source_relative_budget_error
remesh_attempted
remesh_adopted
```

Do not block the run if tangent diagnostics fail. Instead, report the failure reason and fall back to the current predictor.

Exit criteria:

```text
- A baseline current-predictor pilot produces the new columns.
- Existing behavior is unchanged when tangent predictor flags are disabled.
```

### Phase C — Run a short A/B predictor pilot

Goal: determine whether the tangent predictor reduces cost before trying to reach `f_s=0.90`.

Start all pilots from the same accepted anchor at `f_s=0.8759639587`. Use the same grid, buffer, weights, residual-remesh settings, and solver tolerances.

Run three branches only to `f_s=0.878` or `f_s=0.880`:

```text
Pilot A: current predictor only
Pilot B: secant predictor enabled
Pilot C: tangent + secant predictors enabled
```

Suggested tangent pilot environment:

```bash
export IMBH_STANDARD_SLIM_STREAM_MASS_USE_SECANT_PREDICTOR=1
export IMBH_STANDARD_SLIM_STREAM_MASS_USE_TANGENT_PREDICTOR=1
export IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_DAMPINGS=1,0.5,0.25,0.1,0.05
export IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_FD_STEP=1e-5
export IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_SOLVER=equilibrated_lsmr
export IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_LINEAR_DAMPING=0
export IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_INITIAL_STEP=5e-4
export IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_MIN_STEP=1.25e-4
export IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_EVERY_STEP=1
export IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_ON_REJECT=1
export IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_INNER_RG=300
export IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_RADIAL_WEIGHT=1e-3
export IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_ENERGY_WEIGHT=1e-3
export IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_BOUNDARY_WEIGHT=1e-4
export IMBH_STANDARD_SLIM_STREAM_MASS_INTERVAL_FORM=integrated
```

Compare the pilots using:

```text
accepted steps per branch
rejected steps per branch
median nfev per accepted step
total nfev to reach same f_s
median predictor_initial_full
median final weighted full residual
physical/source-domain raw E max
physical/source-domain E L2
mass/source budget error
smoothness of f_adv_global, f_adv_inner, Lrad/LEdd, Rson
number of remesh attempts and adoptions
```

Decision gate:

```text
Use tangent+secant for production if it reduces median nfev per accepted step by >= 2x,
or if it supports accepted df_s >= 2.5e-4 without degrading the split audit.

Use secant-only if tangent gives no cost improvement but secant is stable.

Stay with current predictor only if both secant and tangent create larger initial residuals,
more clipping, or worse physical/source-domain residuals.
```

### Phase D — Production continuation toward `f_s=0.90`

Goal: advance only with the cheapest predictor that passed Phase C.

Recommended stepping strategy:

```text
start: f_s = 0.8759639587
initial df_s = 5e-4 if predictor pilot passed cleanly
minimum df_s = 1.25e-4
maximum df_s = 1e-3 only after two consecutive clean accepted steps
checkpoint every accepted step
write split audit every accepted and rejected step
```

Use the following hard stop conditions:

```text
physical/source-domain raw E max > 1e-4 for two consecutive accepted/recovered attempts
mass/source relative budget error > 3e-4
Rson jumps by > 1e-2 rg between adjacent accepted anchors
f_adv_global or Lrad/LEdd changes nonsmoothly by > 1% without a corresponding smooth trend
accepted df_s remains at the minimum for >= 5 consecutive attempts
partition_peak_physical_E_rg remains locked to one unresolved cell after remesh
```

When a hard stop is triggered, do not immediately call it physical. Go to Phase E.

### Phase E — Local source/interface remeshing if residuals localize physically

Goal: spend grid resolution only where the split audit says the physical problem lives.

Trigger local remeshing if:

```text
partition_peak_physical_E_rg lies near Rinj = 240 rg,
or near the compact source edges,
or within ~5-10 rg of R_buffer = 300 rg,
and physical/source-domain raw E max does not improve after a rejected Newton retry.
```

Remeshing should prioritize:

```text
compact source support
source derivative region
mass-injection gradient
stream torque region
inner side of R_buffer
sonic region only if Rson diagnostics become unstable
```

Do not refine solely because of raw buffer residual peaks.

After remeshing, repeat the same trial `df_s` once. If it fails again, halve `df_s`. If it still fails at the minimum step, go to Phase F.

### Phase F — Pseudo-arclength wall classifier

Goal: decide whether the source-fraction limit is a real branch feature or still a parameter-continuation artifact.

Use pseudo-arclength only after:

```text
- current, secant, and tangent predictors have been tried;
- local source/interface remeshing has been tried;
- split-audit residuals still fail in the physical/source domain;
- the failure repeats at N=768/896/1024 or at least N=896/1024.
```

Minimal test:

```text
Start from the last two clean accepted anchors.
Construct a secant/tangent direction in (z, f_s).
Solve the augmented collocation residual plus arclength constraint.
Try to continue across the failed f_s point by a small arclength step.
```

Interpretation:

```text
If pseudo-arclength crosses the point cleanly, the old wall was numerical.
If pseudo-arclength reveals a reproducible fold with stable physical residual localization,
then the wall may be physical.
If pseudo-arclength also fails but failure remains buffer-dominated,
the reservoir formulation rather than the physical source fraction is the likely bottleneck.
```

### Phase G — Validation anchors for publication-quality claims

Goal: avoid over-interpreting a single successful continuation track.

For selected anchors near:

```text
f_s = 0.88
f_s = 0.89
f_s = 0.90, if reached
```

run the validation grid:

```text
N = 768, 896, 1024
R_buffer = 295, 300, 305 rg
baseline buffer weights = (1e-3, 1e-3, 1e-4)
stricter buffer weights = recommended stricter set from current notes
```

Report each anchor with:

```text
weighted full residual
split physical/source-domain residuals
mass/source budget closure
f_adv_global
f_adv_inner
Lrad/LEdd
Rson
R_buffer sensitivity
N sensitivity
buffer-weight sensitivity
```

A branch point is suitable for scientific discussion only if the physical/source-domain diagnostics are stable under this validation matrix.

### Phase H — Escalation to explicit two-domain reservoir

Goal: reserve the more invasive formulation for cases where the current reservoir closure becomes the limiting ambiguity.

Move to an explicit two-domain reservoir only if at least one of the following is true:

```text
R_buffer sensitivity reappears as f_s approaches 0.90;
raw buffer behavior contaminates physical/source-domain residuals;
pseudo-arclength suggests the reservoir closure is shaping the branch;
publication review requires a cleaner boundary interpretation;
higher Mdot_inner/Edd runs show reservoir-coupling artifacts.
```

Until then, the current outer-buffer reservoir is adequate if every claimed anchor includes the split residual audit.

### Deliverables Codex should produce

At the end of this numerical step, produce:

```text
1. patched runner with effective metadata and predictor diagnostics;
2. three short A/B pilot tables;
3. one comparison markdown summarizing predictor cost and split-audit quality;
4. production continuation table toward f_s=0.90 if Phase C passes;
5. validation table for any new accepted anchors near 0.88, 0.89, or 0.90;
6. a wall-classification note if continuation stalls before 0.90.
```

## 8. Bottom line

The best next numerical step is:

```text
Enable and harden the existing true source-fraction tangent predictor,
then run a short A/B pilot before spending more Newton/remesh cost.
```

Pseudo-arclength is the fallback wall classifier. Local source/interface remeshing is the next improvement if tangent reduces predictor cost but physical residual spikes remain. A fully explicit two-domain reservoir should wait until the tangent/remesh path either reaches `f_s=0.90` or demonstrably fails under split-audit validation.

Do not add wind or stream heating yet.
