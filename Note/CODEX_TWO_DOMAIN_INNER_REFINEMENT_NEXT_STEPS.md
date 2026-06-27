# Codex Next-Step Brief: Two-Domain Solver Now Works at the Outer Boundary; Remaining Blocker Is Sonic-Aware Inner Refinement

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
branch: main, latest accessible state
```

Primary outputs reviewed:

```text
outputs/tables/transonic_two_domain_outer_extension.md
outputs/tables/transonic_two_domain_mesh_validation.md
outputs/tables/transonic_two_domain_inner_refinement.md
scripts/run_transonic_two_domain_inner_refinement.py
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
src/imri_qpe/layer3_minidisk_1d/transonic_continuation.py
```

## Executive update

The diagnosis has changed.

Earlier problems were dominated by the finite-radius outer boundary and outer-slope closure. The new two-domain results show that this is mostly fixed:

```text
- pressure-supported far closure is much better than exact thin/Keplerian far closure;
- outer-domain refinement is stable;
- far-radius variation is stable;
- R_son, lambda0, and integrated Qadv/Qvisc are stable when only the outer domain is changed.
```

The current blocker is now:

```text
sonic-aware inner-domain refinement.
```

The solver has a credible two-domain root at roughly:

```text
Mdot/Mdot_Edd ~= 0.9028
N_inner = 65
N_outer = 54
R_far = 1e5 rg
R_son ~= 5.828 rg
lambda0 ~= 3.675
int Qadv/Qvisc ~= 0.0736
max H/R ~= 0.15
physical residual ~= 1.9e-6
```

This is a meaningful local solution. But it is not yet a mesh-converged continuum solution, because refining the **inner** grid causes the solution to drift and the sonic residual to grow.

Do not resume high-\(\dot M\) continuation yet.

---

# 1. What the new two-domain outer-extension result shows

The two-domain prototype uses:

```text
inner domain: R_son -> 6500 rg
outer domain: 6500 rg -> R_far
interface continuity
far boundary closure
```

The key comparison is:

```text
thin far closure:
    exact Omega = Omega_K at finite R_far
    residual remains ~5e-5 at R_far=5e4 rg

pressure-supported far closure:
    radial-force-balance correction to far rotation
    residual improves to ~3.7e-6 at R_far=5e4 rg
    and ~2e-6 at R_far=1e5 rg
```

Interpretation:

```text
The two-domain pressure-supported outer extension has largely solved the
finite-radius outer-boundary problem.
```

The far closure is now good enough for continued solver development. Freeze it for the next sprint.

Recommended baseline:

```text
R_match = 6500 rg
R_far   = 1e5 rg
N_outer = 54
far closure = pressure_supported
```

Do not keep tuning the far boundary until the inner refinement problem is solved.

---

# 2. What the mesh-validation result shows

The outer-domain tests are stable.

Examples:

```text
N_outer = 54:
    physical ~= 1.92e-6
    R_son ~= 5.828
    lambda0 ~= 3.675
    int_adv ~= 0.07357

N_outer = 72:
    physical ~= 1.92e-6
    R_son ~= 5.828
    lambda0 ~= 3.675
    int_adv ~= 0.07357

R_far = 5e4, 1e5, 2e5:
    physical remains ~1.9e-6
    R_son, lambda0, int_adv remain stable
```

So the outer domain is not the limiting factor.

The failure is specifically the inner grid:

```text
N_inner = 97:
    physical ~= 4.9e-4
    dominant = inner_R
    R_son shifts from 5.828 to 5.908
    lambda0 shifts by ~1e-3
    int_adv shifts by ~0.008
```

This is not smooth mesh convergence.

---

# 3. What the inner-refinement audit shows

The staged inner refinement starts from the good two-domain source:

```text
N_inner = 65
N_outer = 54
physical ~= 1.92e-6
dominant = C1
```

Then:

```text
N66:
    physical ~= 3.36e-6
    dominant = C1
    pass yes

N67:
    physical ~= 4.94e-6
    dominant = C1
    pass yes

N69:
    physical ~= 8.53e-6
    dominant = C1
    pass no

N73:
    physical ~= 1.90e-5
    dominant = C1
    pass no

N77:
    physical ~= 4.27e-5
    dominant = C1
    pass no

N81:
    physical ~= 1.06e-4
    dominant = C1, with inner_R starting to grow
```

The important feature is:

```text
inner interval residuals are initially small after polishing;
the growing failure is first a sonic-compatibility failure, especially C1.
```

The N97 mesh-validation failure later becomes dominated by `inner_R`, but the staged refinement shows the breakdown begins as the sonic block loses regularity under inner-grid insertion.

---

# 4. Revised main diagnosis

The two-domain solution is not failing because of the far boundary. It is failing because the refinement strategy is not preserving the regular sonic critical solution.

The likely causes are:

```text
1. New nodes inserted near R_son are initialized by interpolation, not by the
   regular sonic Taylor expansion.

2. The first few collocation intervals are too close to the singular point for
   ordinary midpoint residuals to behave smoothly.

3. The sonic residual C1 is sensitive to small changes in R_son, lambda0, and
   the first few inner nodes.

4. The current refinement solve frees too many variables at once and lets the
   sonic eigenparameters drift rather than solving a controlled local
   prolongation problem.

5. PCHIP/Hermite interpolation is value-continuous but not defect-preserving.
   The seed residuals show large inner-energy defects immediately after remap.
```

The next milestone is:

```text
A mesh-converged two-domain fixed-Mdot root at Mdot/Mdot_Edd ~= 0.9028,
with N_inner increasing while R_son, lambda0, and int_adv remain stable.
```

---

# 5. Immediate task: localize the residual in the inner domain

Before changing algorithms, output residual profiles.

Create:

```text
outputs/tables/transonic_two_domain_inner_residual_profile_N65.md
outputs/tables/transonic_two_domain_inner_residual_profile_N67.md
outputs/tables/transonic_two_domain_inner_residual_profile_N77.md
outputs/figures/transonic_two_domain_inner_residual_profile.png
```

For each inner interval report:

```text
i
R_left/rg
R_mid/rg
R_right/rg
dx
interval_R
interval_E
condition(A_mid)
smin/smax(A_mid)
Qadv/Qvisc
H/R
xi_eff
```

Also report sonic residuals separately:

```text
D
C1
C2
K
smin/smax at sonic point
null_radial_fraction
```

This will tell us whether the problem is:

```text
- purely sonic residual C1/C2;
- first inner interval;
- a small sonic neighborhood;
- or a distributed inner-domain discretization error.
```

The current tables only give the maximum block; we need radial localization.

---

# 6. Do not refine by generic interpolation near the sonic point

The seed candidates show that generic remapping produces large defects:

```text
sonic-aware PCHIP seed:
    inner_E ~ 0.014 at N66/N67

Hermite variants:
    inner_E can be ~0.05--0.25
```

This means the new-grid seeds are not close to the discrete BVP.

## 6.1 Implement defect-preserving interval splitting

When adding a new node inside an old interval, do not assign it by interpolation only.

Given old endpoints:

```text
y_L, y_R
```

and a new midpoint:

```text
y_M = (logu_M, logT_M)
```

solve a local two-child defect minimization:

```math
\min_{y_M}
\left[
\|F_{L,M}(y_L,y_M)\|^2
+
\|F_{M,R}(y_M,y_R)\|^2
\right].
```

Here \(F_{L,M}\) and \(F_{M,R}\) are the same inner-domain collocation residuals used by the global solver.

Use interpolated \(y_M\) as the initial guess, but solve the local two-equation/two-unknown problem.

For intervals near the sonic point, include a weak prior:

```text
||y_M - y_interp|| / scale
```

to avoid branch jumps.

This produces a seed that approximately satisfies the new-grid interval equations before the global polish.

## 6.2 Apply splitting progressively

Instead of jumping N65 to N97:

```text
split one interval at a time or split all intervals once using local solves;
then global polish.
```

A good diagnostic is the seed residual before global polish. It should drop from:

```text
inner_E ~ 1e-2
```

to something like:

```text
inner_E ~ 1e-4 or smaller
```

before global least-squares.

---

# 7. Treat the sonic neighborhood as a special subdomain

The first few intervals near \(R_{\rm son}\) should not be refined like ordinary intervals.

## 7.1 Keep a finite sonic buffer

Introduce a sonic buffer endpoint:

```text
R_buffer = R_son * exp(Delta_s)
```

with:

```text
Delta_s ~ 0.02--0.05 in ln R
```

Use:

```text
sonic patch: R_son -> R_buffer
regular inner domain: R_buffer -> R_match
outer domain: R_match -> R_far
```

During ordinary inner-grid refinement, keep the sonic patch resolution fixed and refine only the regular inner domain first.

This tests whether the divergence with N is caused by inserting nodes too close to the singular point.

## 7.2 Sonic patch residual

For the sonic patch, replace ordinary midpoint collocation with a Taylor patch:

```math
y(R) =
y_s
+
g_s (x-x_s)
+
\frac{1}{2}h_s (x-x_s)^2
```

where \(x=\ln R\).

At first, implement a first-order patch:

```math
y_{\rm buffer}
=
y_s + g_s \Delta_s.
```

The regular sonic derivative \(g_s\) is not obtained by inverting singular \(A_s\). It must satisfy:

```math
A_s g_s + c_s = 0
```

plus the differentiated regularity condition. For a first implementation, Codex can determine \(g_s\) numerically by solving the collocation residual on a small sonic patch with \(y_s\), \(R_s\), and \(\lambda_0\) fixed.

Practical first version:

```text
Use the existing N65 solution to estimate g_s from the first interval.
Keep g_s fixed during seed generation.
Use global polish to correct.
```

Then implement the full differentiated regularity method later.

## 7.3 Immediate experiment

Run:

```text
same N_inner total, but no new nodes within Delta_s=0.03 of R_son
```

If refinement now succeeds, the failure was sonic-localized.

---

# 8. Refine the regular inner domain with nested grids

The current sequence:

```text
65 -> 66 -> 67 -> 69 -> 73 -> 77 -> 81
```

does not create a clean nested hierarchy and steadily changes the grid.

Use a structured nested hierarchy for the regular inner domain:

```text
N_regular = 32 -> 64 -> 128
```

or:

```text
N_total = 65 -> 129 -> 257
```

but only after adding the sonic buffer.

Use the same physical breakpoints:

```text
R_son
R_buffer
R_match
R_far
```

and keep old nodes where possible.

---

# 9. Strengthen sonic solve/polish before refinement

The source solution is already dominated by C1 at:

```text
physical ~= 1.9e-6
```

If refinement immediately grows C1, the source anchor may not be polished enough in the sonic variables.

Before refining, run a sonic-focused polish at N65:

```text
hold most profile nodes fixed or strongly regularized;
free:
    logR_son
    lambda0
    sonic-node logu/logT
    first 2--4 inner nodes
minimize:
    D,C1,C2,K
    plus first few interval residuals
```

Target:

```text
D,C1,C2,K < 1e-8--1e-7
```

Then repeat the N66/N67 refinement.

If the refinement works only after sonic-focused polish, the issue was under-polished sonic eigenparameters.

---

# 10. Use all sonic compatibility equations as diagnostics, not just C1

The staged refinement is dominated by C1, but we need to know whether C2 and K stay small.

Create a table:

```text
outputs/tables/transonic_two_domain_sonic_residuals_vs_N.md
```

Columns:

```text
N_inner
D
C1
C2
K
smin/smax
Rson
lambda0
first_dx
second_dx
```

Plot:

```text
D,C1,C2,K versus N_inner
D,C1,C2,K versus first_dx
```

If C1 grows as a power of first_dx or N, this is a discretization/sonic-patch issue.

If C1 grows with Rson drift, it is an eigenparameter polish issue.

---

# 11. Branch locking should be local, not global

Current branch comparisons report huge max changes in logu/logT as N grows, e.g.:

```text
max dlogu > 1
max dlogT > 0.6
```

This may be dominated by interpolation near the sonic endpoint or by comparing functions on mismatched grids.

Use a branch-distance metric excluding the sonic patch initially:

```text
R > R_buffer
```

and separately track:

```text
sonic patch distance
regular inner distance
outer distance
```

Reject branch jumps only when the regular inner and outer profiles move discontinuously. Do not let singular sonic-node comparisons dominate the metric.

---

# 12. Improve Newton/polish algorithm for refinement

All release/polish rows report `success=no` due to max function evaluations, even when residual is reasonably small.

For refinement solves, use a staged objective:

## Stage A: defect-seed relaxation

Only new nodes are free; old nodes fixed.

## Stage B: local relaxation

Free new nodes plus nearest old neighbors.

## Stage C: full profile with branch lock

Free all inner nodes but keep \(R_{\rm son}\), \(\lambda_0\), and outer domain fixed.

## Stage D: eigenparameter release

Free \(R_{\rm son}\) and \(\lambda_0\).

## Stage E: full polish

Free everything.

This is much better conditioned than freeing all variables immediately.

---

# 13. Do not resume high-rate continuation yet

The outer extension success is important, but the inner-grid refinement failure means the two-domain solver is not yet validated.

High-rate continuation should remain paused until:

```text
N_inner=65 -> 129 refinement works at fixed Mdot/Edd ~= 0.9028
```

Acceptance target:

```text
physical residual <= few x 1e-6
Rson stable within 0.01--0.02 rg
lambda0 stable within 5e-4
int_adv stable within 1e-3--5e-3
D,C1,C2,K all controlled
```

---

# 14. Recommended immediate Codex sprint

## Task 1: residual localization

Add interval-residual profile tables/figures for N65, N67, N77, N81.

## Task 2: sonic residual versus N

Add a sonic-residual scaling table:

```text
D,C1,C2,K,smin/smax,first_dx,Rson,lambda0 vs N_inner
```

## Task 3: sonic-focused N65 polish

Polish the N65 root focusing on sonic variables and first few intervals.

## Task 4: sonic buffer experiment

Refine only outside:

```text
Delta_s = 0.02, 0.03, 0.05
```

from the sonic point.

## Task 5: defect-preserving interval splitting

Replace pure interpolation with local child-interval residual minimization.

## Task 6: nested regular-domain refinement

After Tasks 1--5, try:

```text
N_regular = 32 -> 64 -> 128
```

with sonic patch held fixed.

## Task 7: only then full N65 -> N129 refinement

Run global refinement after the above diagnostics pass.

---

# 15. Concrete code-level suggestions

## New functions

Add to the two-domain/refinement script or a helper module:

```python
def inner_interval_residual_profile(x, params):
    # return per-interval inner R/E residuals, midpoint radius, condition(A)
    ...

def sonic_residual_audit(x, params):
    # return D,C1,C2,K,smin/smax,null radial fraction, first_dx
    ...

def split_interval_defect_preserving(y_left, y_right, x_left, x_right, params):
    # solve for y_mid minimizing two child interval residuals
    ...

def remap_inner_defect_preserving(old_profile, new_grid, params):
    # use local interval splitting rather than interpolation only
    ...

def sonic_focused_polish(x, params, n_first_intervals=4):
    # free sonic variables and first few nodes, regularize others
    ...

def sonic_buffer_grid(logR_son, logR_match, n_patch, n_regular, delta_s):
    # build grid with a fixed sonic patch and regular-domain nested refinement
    ...
```

## New scripts

```text
scripts/run_transonic_two_domain_residual_localization.py
scripts/run_transonic_two_domain_sonic_scaling.py
scripts/run_transonic_two_domain_sonic_focused_polish.py
scripts/run_transonic_two_domain_sonic_buffer_refinement.py
scripts/run_transonic_two_domain_defect_preserving_refinement.py
```

## New outputs

```text
outputs/tables/transonic_two_domain_inner_residual_profile.md
outputs/tables/transonic_two_domain_sonic_scaling.md
outputs/tables/transonic_two_domain_sonic_focused_polish.md
outputs/tables/transonic_two_domain_sonic_buffer_refinement.md
outputs/tables/transonic_two_domain_defect_preserving_refinement.md
```

---

# 16. Compact Codex prompt

```text
The two-domain pressure-supported outer extension now works. Outer-grid and
far-radius refinements are stable at physical residual ~1.9e-6, with
Rson~5.828, lambda0~3.675, int_adv~0.0736. The remaining failure is inner-grid
refinement: N66/N67 are marginal, but N69+ grow in residual dominated by sonic
C1, and N97 jumps to an inner_R failure. Do not continue high Mdot yet.

Focus next on sonic-aware inner refinement.

Implement:
1. Per-interval residual localization for the inner domain.
2. Sonic residual audit D,C1,C2,K,smin/smax versus N and first_dx.
3. Sonic-focused polish of the N65 root, freeing Rson, lambda0, sonic node, and
   first 2--4 inner nodes.
4. A sonic buffer domain: Rson -> Rbuffer fixed; refine only Rbuffer -> Rmatch
   first. Test Delta_s=0.02,0.03,0.05.
5. Defect-preserving interval splitting: when inserting a new node, solve for
   the midpoint values so the two child collocation residuals are minimized
   with endpoints fixed.
6. Nested regular-domain refinement after the sonic buffer is stable.
7. Only after N65->N129 inner refinement converges, resume high-Mdot
   continuation.

Freeze the outer setup for now:
    R_match=6500 rg
    R_far=1e5 rg
    N_outer=54
    pressure-supported far closure.
```

---

# 17. Bottom line

The project has made real progress. The outer boundary problem is largely
solved.

The next hard problem is mathematical/numerical:

```text
How to refine a collocation BVP whose left endpoint is a sonic critical point?
```

Solve that before asking whether the high-\(\dot M\) no-wind slim branch reaches
the QPE target.
