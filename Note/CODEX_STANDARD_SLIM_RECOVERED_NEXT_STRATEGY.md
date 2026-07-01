# Codex Next-Step Brief: Standard Slim Disk Is Recovered to 0.2 and Scouted to 0.916; Now Certify the Near-Eddington Benchmark Efficiently

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
latest main branch
```

Start from:

```text
Note/CODEX_STANDARD_SLIM_MDOT_EQUILIBRATED_CONTINUATION_RESULTS.md
```

Key outputs referenced in that note:

```text
outputs/tables/slim_benchmark_adaptive_mdot_ladder_pressure_1e4_1e2.md
outputs/tables/slim_benchmark_adaptive_mdot_ladder_pressure_1e2_3e2.md
outputs/tables/slim_benchmark_adaptive_mdot_ladder_pressure_3e2_1e1.md
outputs/tables/slim_benchmark_adaptive_mdot_ladder_pressure_adaptiveN_1p55e1_2e1.md
outputs/tables/slim_benchmark_mesh_closure_validation_pressure_adaptiveN_2e1.md
outputs/tables/slim_benchmark_adaptive_mdot_ladder_pressure_N512_scout_2e1_5e1.md
outputs/tables/slim_benchmark_adaptive_mdot_ladder_pressure_N512_scout_5e1_1.md
```

## Executive assessment

The standard single-BH no-wind slim-disk benchmark is now largely recovered.

The critical breakthrough was replacing the finite-radius exact Keplerian outer
condition with the pressure-supported outer angular-velocity condition. With
that closure, the standard benchmark is a strict-anchor ladder from

```text
Mdot/Edd = 1e-4
to
Mdot/Edd = 1e-2
```

at:

```text
R_out = 10000 rg
N = 128
```

and it extends robustly to:

```text
Mdot/Edd = 0.2
```

using adaptive \(N\).

The direct high-resolution \(N=512\) scout reaches:

```text
Mdot/Edd = 0.5
```

comfortably under a relaxed \(1e-5\) criterion, and reaches at least:

```text
Mdot/Edd ~= 0.916
```

as a near-Eddington scout. The remaining difficulty is not a sonic barrier or
outer-boundary failure. It is now mostly:

```text
runtime + interval_E residual control + high-N Jacobian/collocation efficiency.
```

This is a major change in interpretation.

The solver can now recover a standard no-wind slim disk over a broad range.
Therefore, if the IMRI/minidisk no-wind branch continues to fail after the same
numerical improvements, that failure becomes much more likely to be a genuine
finite-boundary/minidisk physics issue rather than a generic slim-disk solver
failure.

---

# 1. What is now established

## 1.1 Pressure-supported boundary fixed the old 1e-2 floor

With the old `thin_value` boundary, the high-side residual floor at
\(Mdot/Edd = 1e-2\) was dominated by the outermost interval and stayed above
the strict-anchor threshold.

With the pressure-supported closure:

```text
1e-4 -> 1e-2:
    every branch row is a strict anchor;
    endpoint 1e-2 residual = 8.819e-7;
    strict anchors = 34/34 down and 34/34 up.
```

This means the finite-radius outer-boundary issue is solved for the standard
benchmark.

## 1.2 The benchmark reaches 0.03 cleanly

The pressure-supported ladder from \(1e-2\) to \(3e-2\) has:

```text
endpoint Mdot/Edd = 0.03
endpoint residual = 1.528e-6
strict anchors = 22/22
dominant = interval_R
```

Mesh refinement improves it:

```text
N=128 residual = 1.528e-6
N=160 residual = 9.416e-7
```

This is a robust standard-slim checkpoint.

## 1.3 The benchmark reaches 0.2 with adaptive N

The adaptive-N run reaches:

```text
Mdot/Edd = 0.2
selected N = 288
residual = 2.964e-6
dominant = interval_E
pressure-target mismatch = 5.922e-7
Rson/rg = 5.953
lambda0/lK_ISCO = 1.00002
max H/R = 0.0416
```

The N=320 spot check improves the residual:

```text
N=288 residual = 2.964e-6
N=320 residual = 2.621e-6
```

So 0.2 is a strict, mesh-supported standard no-wind benchmark point.

## 1.4 The N512 scout reaches 0.5 and nearly 1

The direct N512 scout reaches:

```text
Mdot/Edd = 0.5
residual = 4.663e-6
max H/R = 0.0957
integrated advective fraction = -1.49e-2
```

and reaches:

```text
Mdot/Edd = 0.915786
residual = 9.896e-6
max H/R = 0.1486
integrated advective fraction = 1.927e-2
```

This is not yet a strict benchmark, but it is strong evidence that the standard
no-wind branch continues toward Eddington.

The failed/unfinished step toward 0.972 appears to be a practical numerical
cost issue, not a clear physical fold.

---

# 2. Physical interpretation

The standard no-wind slim disk has essentially been recovered.

This means:

```text
The core transonic equations, sonic eigenvalue structure, and continuation
machinery can find a global no-wind solution when the boundary-value problem
is the classical single-BH slim disk.
```

This does **not** prove that the IMRI/minidisk no-wind branch must exist.
The IMRI problem has a finite stream-fed/tidally truncated outer reservoir,
not an asymptotic standard disk.

But it does remove the biggest concern:

```text
The earlier IMRI failure was not simply because the solver is incapable of
finding any no-wind slim disk.
```

The next standard-benchmark goal is no longer basic recovery. It is
certification of the near-Eddington branch.

---

# 3. Best strategy now

Do not add wind yet.

The best strategy is:

```text
A. certify the standard benchmark to Mdot/Edd ~= 1;
B. then run a finite-boundary deformation test;
C. only then return to the IMRI no-wind/wind decision.
```

This separates three questions:

```text
1. Can the code solve standard no-wind slim disks?       Mostly yes.
2. Can the code solve finite-boundary no-wind disks?     Not yet tested cleanly.
3. Does the IMRI no-wind branch physically exist?        Still open.
```

---

# 4. Immediate Sprint A: certify 0.5 and 1.0 efficiently

## 4.1 Treat N512 scout as a map, not final certification

The N512 scout is useful, but it used:

```text
acceptance = anchor = current = 1e-5
```

Therefore points above 0.2 are scouts, not strict anchors.

Do not use the N512 run alone as final proof.

## 4.2 Certify selected checkpoints

Pick:

```text
Mdot/Edd = 0.3
0.5
0.7
0.9
1.0
```

For each checkpoint, run:

```text
N = 384, 512, 640
```

or, if 640 is too expensive:

```text
N = 384, 512
plus one N = 768 final spot at 1.0
```

Acceptance:

```text
residual <= 1e-5 for scout acceptance
residual <= 3e-6 for strict anchor
Rson stable to < 0.02 rg
lambda0/lK stable to < 5e-4
max H/R stable to a few percent
integrated advective fraction stable to < 0.01
dominant residual geography understood
```

At this stage, a strict residual below \(3e-6\) at \(Mdot/Edd=1\) may be
expensive. If the sequence is mesh-converged at the \(1e-5\) level and no
sonic/fold pathology appears, it is still a scientifically useful benchmark.

## 4.3 Run residual localization at high rate

For:

```text
0.5
0.9
1.0
```

create:

```text
outputs/tables/slim_benchmark_high_rate_residual_profile.md
outputs/figures/slim_benchmark_high_rate_residual_profile.png
```

Report per interval:

```text
R/rg
interval_R
interval_E
Qadv/Qvisc
H/R
Omega/OmegaK
condition(A)
smin/smax(A)
```

The key question is whether `interval_E` is:

```text
localized near the sonic/inner advective region,
localized near the outer boundary,
or broad over the whole disk.
```

This determines the next numerical improvement.

---

# 5. Immediate Sprint B: reduce high-N cost

The current bottleneck near Eddington is runtime. The N512 steps above 0.5
take hundreds of seconds, and the attempted 0.972 correction did not complete
quickly.

The main cost likely comes from:

```text
finite-difference Jacobian work,
outer-slope refresh,
large N collocation,
regularized LSMR iterations.
```

## 5.1 Freeze outer slopes during Newton iterations

Current pressure-supported closure refreshes slopes once per step, which is
fine. Do not refresh slopes inside Newton iterations.

For high-N runs:

```text
measure slopes at source;
use fixed slopes for prediction and polish;
refresh once after convergence;
repolish only if pressure mismatch worsens.
```

This prevents the boundary condition from changing during linearization.

## 5.2 Reduce finite-difference Jacobian cost

Add a high-N mode:

```text
reuse Jacobian for 2--3 Newton iterations unless residual reduction stalls.
```

or:

```text
update only the blocks near regions with large residual.
```

The standard branch is smooth; full Jacobian recomputation every iteration may
not be necessary.

## 5.3 Implement analytic/local block derivatives for interval_E first

The dominant high-rate residual is `interval_E`.

Do not derive the entire Jacobian at once. Start by replacing finite-difference
derivatives for the energy residual block with analytic or automatic-diff
local derivatives.

Target:

```text
reduce per-step runtime by factor 2--5.
```

## 5.4 Use residual-based adaptive mesh instead of uniform N

The selected N grows rapidly with Mdot. Uniform N is wasteful.

Implement a mesh adaptation based on interval residuals:

```text
refine where |interval_E| or |interval_R| is large;
coarsen where both are tiny.
```

Maintain smoothness with PCHIP/Hermite remapping.

This may let the benchmark reach \(Mdot/Edd=1\) with far fewer effective nodes
than uniform N512.

---

# 6. Immediate Sprint C: continue from 0.916 to 1.0

Use the existing N512 checkpoint at:

```text
Mdot/Edd = 0.915786
```

Restart with:

```text
MAX_STEP_MU = 0.015 or 0.02
N = 512
acceptance = 1e-5
anchor = 1e-5 initially
```

Target:

```text
0.95
0.98
1.0
```

Do not jump directly to 0.972 with a large step.

If the first attempt is expensive:

```text
save intermediate states every Newton iteration or every accepted residual
decrease;
lower max step;
try Jacobian reuse;
try freeze slopes fully.
```

Then spot-check \(Mdot=1\) at:

```text
N=640 or N=768
```

if computationally possible.

---

# 7. After standard benchmark reaches 1: finite-boundary deformation

Before adding wind to the IMRI model, run one more controlled experiment.

Take the standard no-wind solution and gradually deform the outer boundary from
a classical far boundary to a finite minidisk-like boundary.

Homotopy parameters:

```text
R_out: 10000 rg -> 1000 rg -> 300 rg -> minidisk relevant radius
outer closure: pressure-supported far -> finite reservoir target
outer entropy/temperature: standard thin -> prescribed finite reservoir
outer angular momentum: Keplerian -> stream/circularization value
```

This answers:

```text
Does merely making the outer boundary finite produce the kind of fold seen in
the IMRI problem?
```

If yes, then the IMRI no-wind obstruction is probably a finite-boundary effect,
not necessarily a wind effect.

If no, then add tidal/stream terms one at a time.

---

# 8. When to return to IMRI

Return to the IMRI no-wind problem only after either:

```text
standard benchmark reaches Mdot/Edd ~= 1 robustly
```

or:

```text
standard benchmark is shown to be robust enough up to Mdot/Edd ~= 0.5 and
the remaining near-Eddington issue is purely runtime/certification.
```

Then do:

```text
standard disk
-> finite boundary
-> finite boundary + minidisk radius
-> stream angular momentum
-> stream heating
-> tidal torque
```

If the branch disappears during this homotopy, that is the physically
meaningful point where no-wind fails.

---

# 9. What this means for wind

Wind is still the most likely physical next ingredient for the IMRI high state,
but it should not be added to fix a standard slim benchmark problem.

The standard benchmark is now healthy enough that wind is no longer needed as
a numerical rescue. Wind should be added only after the finite-boundary/minidisk
homotopy shows that no smooth no-wind branch survives.

---

# 10. Concrete scripts to add or modify

## Certification

```text
scripts/run_standard_slim_high_rate_certification.py
```

Outputs:

```text
outputs/tables/slim_benchmark_high_rate_certification.md
outputs/figures/slim_benchmark_high_rate_certification.png
```

## Residual localization

```text
scripts/run_standard_slim_high_rate_residual_profile.py
```

Outputs:

```text
outputs/tables/slim_benchmark_high_rate_residual_profile.md
outputs/figures/slim_benchmark_high_rate_residual_profile.png
```

## Efficient high-N continuation

```text
scripts/run_standard_slim_highN_efficiency_audit.py
```

Test:

```text
slope freeze
Jacobian reuse
energy-block analytic derivative
adaptive mesh
```

Outputs:

```text
outputs/tables/slim_benchmark_highN_efficiency_audit.md
```

## Finite-boundary deformation

```text
scripts/run_standard_slim_finite_boundary_homotopy.py
```

Outputs:

```text
outputs/tables/slim_benchmark_finite_boundary_homotopy.md
outputs/figures/slim_benchmark_finite_boundary_homotopy.png
```

---

# 11. Acceptance criteria for the next milestone

## Near-Eddington standard benchmark

At \(Mdot/Edd=1\):

```text
residual <= 1e-5 acceptable scout
residual <= 3e-6 strict anchor if feasible
no sonic compatibility failure
no radial projection fold before R_out
H/R and integrated advective fraction reasonable
profiles stable under at least one higher-N spot check
```

## Finite-boundary homotopy

A meaningful no-wind failure requires:

```text
standard branch passes;
finite-boundary homotopy is smooth until a specific parameter value;
branch disappears through a documented fold/critical event;
result stable under grid and closure checks.
```

---

# 12. Compact Codex prompt

```text
The standard no-wind slim benchmark is now recovered much further than before.
Pressure-supported outer closure gives strict anchors from Mdot/Edd=1e-4 to
1e-2 at Rout=10000,N=128. Continuation to 3e-2 is robust and strict.
Adaptive-N reaches 0.2 as a strict anchor with N=288 and N=320 spot-check
support. Direct N512 relaxed scout reaches 0.5 and then 0.916, with residuals
still dominated by interval_E and no sonic failure.

Next:
1. Certify selected checkpoints 0.3,0.5,0.7,0.9,1.0 with N=384/512/640 if
   feasible.
2. Restart from N512 Mdot=0.915786 with smaller steps MAX_STEP_MU=0.015--0.02
   to reach 1.0.
3. Add high-rate residual localization to see where interval_E lives.
4. Reduce high-N cost by freezing outer slopes during Newton, reusing Jacobians,
   and implementing analytic/local derivatives for interval_E first.
5. Add residual-based adaptive mesh rather than uniform N growth.
6. After standard benchmark reaches ~1, run a finite-boundary homotopy before
   adding wind to the IMRI model.
```

---

# 13. Bottom line

The standard no-wind slim disk is no longer "almost" recovered in the low and
moderate range. It is recovered through \(Mdot/Edd=0.2\), and scouted close to
Eddington.

The best next strategy is:

```text
certify near-Eddington standard no-wind behavior efficiently,
then test finite-boundary deformation,
then decide on IMRI wind.
```
