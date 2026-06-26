# Codex Next-Step Brief: Slope-Law / Grid-Validation Bottleneck After Latest Results

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
latest reviewed main commit: 42bbbb8
```

Start from these latest outputs:

```text
Note/CODEX_SLOPE_LAW_GRID_VALIDATION_NEXT_STEPS.md
outputs/tables/transonic_outer_slope_law_audit.md
outputs/tables/transonic_integrated_defect_slope_law_audit.md
outputs/tables/transonic_controlled_slope_rout_continuation.md
outputs/tables/transonic_slope_law_nested_grid_audit.md
outputs/tables/transonic_high_rate_ladder.md
outputs/tables/transonic_coarse_to_fine.md
outputs/tables/transonic_adaptive_homotopy.md
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
src/imri_qpe/layer3_minidisk_1d/transonic_continuation.py
```

The latest results are not a failure of the physical model. They show that the
solver has reached a more delicate numerical issue:

```text
The fixed-Mdot transonic root can be made good locally at N=64, but it is not
yet grid-robust because the outer-slope closure and remapping/refinement
strategy are controlling the solution.
```

Do not add wind, stream feeding, tidal truncation, or time dependence yet.

---

# 1. What the latest run actually achieved

The new full-slope and slope-law audits are a major improvement over the older
finite-radius Keplerian outer boundary.

The best local N64 fixed-\(\dot M\) state is now approximately:

```text
Mdot/Mdot_Edd ~= 0.90277664
R_out         ~= 6500 r_g
R_son         ~= 5.826 r_g
lambda0       ~= 3.675
max H/R       ~= 0.15
outer H/R     ~= 0.01437
integrated Qadv/Qvisc ~= 0.074
physical residual ~= 1.7e-6
```

That is a real local numerical success.

However, the same result is not yet scientifically robust because nested-grid
validation fails:

```text
N=33  residual ~= 3.24e-4, R_son ~= 5.678
N=65  residual ~= 1.43e-5, R_son ~= 5.832
N=129 residual ~= 1.74e-3, R_son ~= 5.118
```

This means:

```text
The N64 1.7e-6 root is a good local branch point, but not yet a
mesh-converged transonic solution.
```

The next step must be a controlled grid/slope validation program, not further
high-\(\dot M\) continuation.

---

# 2. Main diagnosis

The current difficulty has moved through several stages:

```text
old problem:     imposed local xi branch was not global
then:            exact finite-Rout Keplerian outer condition produced a 1e-4 floor
then:            nested matched_outer_state closure was noisy/nonsmooth
current problem: outer slope closure + grid remapping controls the fixed-Mdot root
```

The latest evidence:

## 2.1 Slope tuning matters at the \(10^{-3}\) level

At \(R_{\rm out}=5000r_g\), changing only

```text
g_T -> g_T + 1e-3
```

reduces the physical residual to roughly:

```text
4.3e-6.
```

Changing

```text
g_u -> g_u - 3e-3
```

gives roughly:

```text
7.4e-6.
```

At \(R_{\rm out}=6500r_g\), the baseline local quadratic slope is best among
the tried options, reaching:

```text
1.7e-6.
```

This is telling us that the outer boundary is not just a harmless asymptotic
condition. It is actively selecting the numerical branch.

## 2.2 The optimum slope correction is not a smooth obvious law

The best correction changes with radius:

```text
R=5000: g_T + 1e-3
R=5500: g_T + 1e-3
R=6000: g_u - 6e-3, g_T - 3e-3
R=6500: baseline
```

That is not a clean physical asymptotic sequence. It is more likely the result
of coupling among:

```text
finite-radius slope fitting
collocation truncation error
branch/basin selection
Newton tolerance
```

Therefore do not hard-code the fitted correction law as physics.

## 2.3 Integrated defects help selected residuals but do not solve grid robustness

The integrated-defect audit from the best \(R_{\rm out}=6500r_g\) root gives
selected residuals as low as \(5\times10^{-7}\), but the unweighted
differential physical audit remains around:

```text
2.2e-6.
```

This is still useful. It means integrated defects can be a good preconditioned
Newton residual, but the differential residual should remain the physical
audit.

## 2.4 The N129 result is almost certainly a basin/remap failure, not a physical branch

The N129 run jumps to:

```text
R_son ~= 5.118 r_g
lambda0 ~= 3.712
integrated advective fraction ~= 0.035
physical residual ~= 1.7e-3
```

from an N64/N65 state near:

```text
R_son ~= 5.83 r_g
lambda0 ~= 3.675
integrated advective fraction ~= 0.074.
```

That is not a smooth resolution refinement. It is a branch/basin jump.

Do not use that N129 result as evidence of non-convergence of the physical
solution. It is evidence that the current prolongation and fixed-grid polish
are not controlling the basin.

---

# 3. High-rate continuation should pause

The high-rate ladder is encouraging but not yet authoritative.

The scout N48 branch reaches:

```text
Mdot/Mdot_Edd ~= 0.7
max H/R ~= 0.125
integrated advective fraction ~= 0.031
max residual ~= 2.9e-4
```

but fails around:

```text
Mdot/Mdot_Edd ~= 0.73--1
```

with `outer_Omega` dominating. In addition, the N64 confirmation accepts
\(0.3\), but the N64 confirmation at \(0.5\) fails badly with:

```text
interval_R ~= 2.9e-3
C2 ~= 7.95e-4
```

This tells us the high-rate ladder is currently a scout calculation, not a
validated branch. It is being run before the fixed-\(\dot M\) roots have
mesh-converged.

Action:

```text
Stop high-Mdot continuation until the fixed-Mdot root at Mdot/Edd ~= 0.9028
is stable under N=65 -> 97 -> 129 refinement.
```

---

# 4. Recommended next strategy

The next Codex sprint should have one objective:

```text
Turn the N64 R_out=6500 fixed-Mdot root into a mesh-converged fixed-Mdot root,
or demonstrate cleanly why the present boundary formulation prevents that.
```

Do this in four layers.

---

# 5. Layer A: staged resolution continuation with slope refresh

Do not jump directly from N64 to N129.

Use:

```text
N = 65, 81, 97, 113, 129
```

or:

```text
N = 65, 81, 97, 113, 129, 161
```

at fixed:

```text
Mdot/Edd = 0.90277664
R_out    = 6500 r_g
outer_closure = full_slope_match
sonic rows = symmetric D,C1,C2
interval form = differential for physical audit
```

At each resolution stage:

1. Remap the previous accepted solution with PCHIP or Hermite interpolation.
2. Recompute the local outer slope estimate at that resolution.
3. Polish with a trust-region restriction:
   ```text
   |Delta R_son| < 0.03--0.05 r_g initially
   |Delta lambda0| < 1e-3 initially
   max |Delta logu|, max |Delta logT| bounded initially
   ```
4. Relax the trust region only after residual decreases.
5. Store both the initially fitted slope and the final output slope.

Acceptance criterion for each stage:

```text
physical residual <= 5e-6 for intermediate stages
physical residual <= 2e-6 for final N>=129
R_son changes smoothly
lambda0 changes smoothly
integrated advective fraction changes smoothly
no branch jump in R_son
```

If a stage tries to jump from \(R_{\rm son}\simeq5.83\) to \(5.1\), reject it
as a basin jump and reduce the resolution step.

---

# 6. Layer B: outer slope as an auxiliary unknown

The slope-law experiments show that \(g_u,g_T\) are too important to be treated
as fixed post-hoc numbers. The clean next improvement is to promote them to
auxiliary unknowns in the fixed-\(\dot M\) solve.

## 6.1 Unknowns

Add two global unknowns:

```text
g_u_out
g_T_out
```

## 6.2 Boundary residuals

Keep the direct full-slope boundary residual:

```text
B_outer = F_local(R_out, y_out, g_out, lambda0)
```

which gives two equations.

## 6.3 Add two matching/regularization equations

Instead of prescribing \(g_{\rm out}\), constrain it to the reduced/asymptotic
slope with a soft or hard matching condition:

```math
B_{g_u}
=
\frac{g_u-g_{u,\rm red}}{\sigma_{g_u}},
```

```math
B_{g_T}
=
\frac{g_T-g_{T,\rm red}}{\sigma_{g_T}}.
```

Start with:

```text
sigma_g_u = 3e-3--1e-2
sigma_g_T = 1e-3--3e-3
```

There are two possible modes:

### Mode 1: hard square system

Add \(g_u,g_T\) as unknowns and add \(B_{g_u}=0,B_{g_T}=0\).

This is equivalent to prescribed slopes but makes the algebra cleaner.

### Mode 2: weakly constrained overdetermined system

Add \(g_u,g_T\) as unknowns, keep \(B_outer=0\), and add \(B_g\) as weak
regularization rows.

This lets the solver adjust the slopes by the tiny amount needed for a true
root while penalizing unphysical slope drift.

Recommended first test: Mode 2.

## 6.4 Why this helps

The current slope sensitivity shows that changing \(g_T\) by \(10^{-3}\) can
change the residual by more than an order of magnitude. Treating slopes as
unknowns with a physically motivated prior prevents hand-tuning while keeping
the solution tied to the asymptotic disk.

## 6.5 Audit columns to add

For every fixed-\(\dot M\) root report:

```text
g_u_red
g_T_red
g_u_solved
g_T_solved
Delta g_u
Delta g_T
regularization contribution to chi2
outer residuals
```

If \(\Delta g\) remains small and converges with grid, this is a valid outer
matching method.

If \(\Delta g\) grows with N or \(R_{\rm out}\), the outer asymptotic model is
not adequate.

---

# 7. Layer C: use integrated defects as a solver preconditioner, not as the science metric

Based on the latest audit:

```text
differential residual gives physical ~= 1.72e-6
integrated raw gives selected ~= 5e-7 but differential physical ~= 2.26e-6
integrated l2 gives differential physical ~= 2.19e-6
```

Recommendation:

```text
Use integrated defects to generate a better Newton step, but always polish and
judge the solution using the differential physical residual.
```

Practical implementation:

1. Run integrated-defect solve for 50--100 iterations.
2. Switch to differential residual polish.
3. Require differential physical residual \(<2\times10^{-6}\).
4. Compare the final differential root to a differential-only root.

Do not replace the physical audit with the integrated selected residual.

---

# 8. Layer D: branch tracking and trust-region basin control

The N129 jump shows that the solver can move to the wrong basin.

Add branch-tracking logic.

## 8.1 Branch distance metric

For a candidate solution relative to the previous resolution stage, compute:

```math
d_{\rm branch}^2
=
\left(\frac{\Delta R_{\rm son}}{0.05r_g}\right)^2
+
\left(\frac{\Delta\lambda_0}{10^{-3}}\right)^2
+
\frac{1}{N}\sum_i
\left(\frac{\Delta\ln u_i}{0.02}\right)^2
+
\frac{1}{N}\sum_i
\left(\frac{\Delta\ln T_i}{0.01}\right)^2.
```

Reject or downweight candidates with:

```text
d_branch >> 1
```

unless they have a dramatically smaller residual and can be connected by
continuation.

## 8.2 Two-stage solve

For each new \(N\):

### Stage 1: continuation-locked polish

Minimize:

```math
||F||^2 + \epsilon_{\rm lock} d_{\rm branch}^2
```

with large \(\epsilon_{\rm lock}\).

### Stage 2: release

Reduce \(\epsilon_{\rm lock}\) gradually to zero while monitoring that the
solution remains on the same branch.

This prevents the N129 solve from jumping to \(R_{\rm son}\simeq5.1\).

---

# 9. Higher-order collocation should be prototyped now

The midpoint scheme is low order, and the current residual is dominated by
`interval_R` once the outer slope is tuned.

Implement one higher-order alternative, but use it only as a comparison first.

## 9.1 Trapezoidal collocation

For a local ODE form:

```math
y'=f(x,y),
```

use:

```math
y_{i+1}-y_i
-
\frac{\Delta x_i}{2}
(f_i+f_{i+1})=0.
```

Here \(f\) is obtained from the local differential matrix:

```math
g=f(x,y)=-A^{-1}c.
```

Away from the sonic node this is straightforward.

Near the first sonic interval, either:

```text
- keep midpoint on the first interval, or
- use a regular sonic derivative once implemented.
```

## 9.2 Hermite-Simpson later

Once trapezoidal works, implement Hermite-Simpson:

```math
y_{i+1}-y_i
-
\frac{\Delta x_i}{6}
(f_i+4f_m+f_{i+1})=0.
```

Do not make Hermite-Simpson the default until the lower-order comparison is
understood.

## 9.3 Required test

At the same \(N=65\) and \(N=129\), compare:

```text
midpoint
trapezoidal
eventually Hermite-Simpson
```

If the root location and residual converge with scheme order, the BVP is
becoming trustworthy.

---

# 10. Recompute outer slopes at each grid and radius

Do not use N64 slope fits for N129.

For each resolution and \(R_{\rm out}\):

1. Build the reduced/asymptotic outer profile on that grid.
2. Fit \(g_u,g_T\) using a fixed physical radial window, not just a fixed
   number of cells.
3. Use the same physical window across \(N\).

Suggested window:

```text
R in [0.92 R_out, 1.00 R_out]
```

or, if too narrow:

```text
R in [0.85 R_out, 1.00 R_out].
```

Then fit a quadratic in \(\ln R\) and report sensitivity to:

```text
window width
polynomial degree
skip last cell
```

The latest calibration shows that degree and window choice matter at the
\(10^{-3}\)-slope level, which is already enough to control the residual.

---

# 11. Do not overuse max_nfev status

Many good rows report `success=no` only because `max_nfev` is reached after the
residual is already small. That is acceptable for diagnostics, but the branch
table should separate:

```text
optimizer_success
residual_pass
branch_pass
science_pass
```

For fixed-\(\dot M\) root validation use:

```text
science_pass = residual_pass and branch_pass and grid_pass
```

where:

```text
residual_pass: physical residual < threshold
branch_pass: no jump in Rson/lambda/profile
grid_pass: stable under neighboring N
```

Do not require SciPy's `success=True` if the residual is already below target.
But do require grid and branch stability before using the solution scientifically.

---

# 12. Recommended immediate experiment sequence

## Experiment 1: N65 local root with slope auxiliary unknowns

Configuration:

```text
Mdot/Edd = 0.90277664
R_out = 6500 r_g
N = 65
outer_closure = full_slope_match
sonic rows = symmetric D,C1,C2
interval form = differential
unknowns += [g_u_out, g_T_out]
weak slope priors around local quadratic n_fit=8 slopes
```

Goal:

```text
physical residual < 2e-6
Delta g_T and Delta g_u small
Rson ~= 5.83
lambda0 ~= 3.675
int_adv ~= 0.074
```

## Experiment 2: staged N65 -> N81 -> N97 -> N113 -> N129

Use branch locking and slope refresh.

Goal:

```text
no jump in Rson
residual remains few x 1e-6
profiles converge
```

## Experiment 3: fixed-grid \(R_{\rm out}\) continuation

At N65 or N81, continue:

```text
R_out = 5000 -> 5500 -> 6000 -> 6500 -> 7000 -> 8000
```

with slope auxiliary unknowns.

Goal:

```text
outer slope corrections trend smoothly
solution profiles converge as R_out increases
```

## Experiment 4: trapezoidal collocation comparison

Run midpoint vs trapezoidal at:

```text
N = 65, 97, 129
R_out = 6500
Mdot/Edd = 0.90277664
```

Goal:

```text
scheme dependence decreases with N
```

## Experiment 5: only then resume \(\dot M\)-continuation

Once the fixed-\(\dot M\) root is grid-stable, continue:

```text
0.90 -> 1.0 -> 1.2 -> 1.5 -> 2.0
```

using the same slope-auxiliary outer closure and branch-tracking metric.

---

# 13. Reinterpreting the current high-rate results

The high-rate ladder should now be treated as a scout.

Useful information from it:

```text
N48 can reach Mdot/Edd ~= 0.7 with H/R ~= 0.125.
The branch does not obviously end below that.
Outer residuals and resolution sensitivity dominate failures.
```

Not yet reliable:

```text
any claim that the branch ends near Mdot/Edd ~= 0.7--1.
any N64 confirmation failure seeded by a non-grid-validated N48 profile.
```

The high-rate physics question remains open.

---

# 14. Code changes by file

## `transonic_collocation.py`

Add a new boundary mode:

```python
outer_closure = "full_slope_match_with_slope_unknowns"
```

Support global unknowns:

```text
g_u_out
g_T_out
```

Add residual rows:

```text
B_outer_R
B_outer_E
B_g_u_prior
B_g_T_prior
```

If using a square system, replace two rows appropriately or solve as
least-squares with weak priors. For development, least-squares with priors is
acceptable.

Add output diagnostics:

```text
g_u_prior
g_T_prior
g_u_solved
g_T_solved
delta_g_u
delta_g_T
slope_prior_chi2
```

## `transonic_continuation.py`

Add branch-locking trust-region support:

```python
def branch_distance(profile_new, profile_old, metric):
    ...

def branch_locked_polish(...):
    ...
```

Add a status flag:

```text
branch_continuity_pass
```

## New scripts

Create:

```text
scripts/run_transonic_slope_unknown_root.py
scripts/run_transonic_staged_resolution_continuation.py
scripts/run_transonic_trapezoid_collocation_audit.py
scripts/run_transonic_slope_unknown_rout_continuation.py
```

Outputs:

```text
outputs/tables/transonic_slope_unknown_root.md
outputs/tables/transonic_staged_resolution_continuation.md
outputs/tables/transonic_trapezoid_collocation_audit.md
outputs/tables/transonic_slope_unknown_rout_continuation.md
```

---

# 15. Acceptance criteria before returning to high rates

Do not restart high-\(\dot M\) continuation until all of the following pass:

```text
1. N65, N97, N129 fixed-Mdot roots exist near Rson ~= 5.83.
2. physical residual <= few x 1e-6 on all three.
3. lambda0 stable to <1e-3.
4. integrated advective fraction stable to <1e-3--1e-2.
5. max profile differences decrease with N.
6. slope corrections remain small and smooth with N.
7. midpoint/trapezoid scheme differences decrease with N.
8. no branch jump like Rson=5.1 occurs.
```

Only after this should Codex resume the high-rate ladder.

---

# 16. Compact Codex prompt

```text
The latest 42bbbb8 results show a good local N64 fixed-Mdot root at
Mdot/Edd=0.9028, R_out=6500 rg, with residual ~1.7e-6, Rson~5.826,
lambda0~3.675, H/R~0.15. But nested-grid validation fails: N65 is only
~1.4e-5 and N129 jumps to Rson~5.118 with residual ~1.7e-3. This means the
root is not yet grid-robust.

Next implement slope/grid validation before any high-Mdot continuation:

1. Promote outer slopes g_u_out and g_T_out to auxiliary unknowns with weak
   priors to the reduced/asymptotic slope estimates. Keep full_slope_match
   boundary equations, but stop treating slopes as fixed post-hoc numbers.

2. Recompute slope priors at every resolution using a fixed physical outer
   window, not a fixed number of cells.

3. Run staged resolution continuation:
       N = 65 -> 81 -> 97 -> 113 -> 129
   with branch locking on Rson, lambda0, logu, logT. Reject jumps to
   Rson~5.1 unless connected by smooth continuation.

4. Use integrated defects only as a Newton/preconditioning aid. Always judge
   physical residual with the differential audit.

5. Prototype trapezoidal collocation and compare against midpoint at N65, N97,
   N129.

6. Add branch_distance and branch_continuity_pass to the status.

7. Do not resume high-Mdot continuation until the fixed-Mdot root at 0.9028 is
   stable under N and collocation scheme.
```

---

# 17. Bottom line

The solver is making real progress. The best current state is not bad; it is a
high-quality local root. The problem is that it is **not yet a validated
discrete approximation to a continuum transonic solution**.

The next scientific milestone is therefore:

```text
mesh-converged fixed-Mdot transonic root at Mdot/Edd ~= 0.9
```

not:

```text
pushing the high-rate ladder farther.
```
