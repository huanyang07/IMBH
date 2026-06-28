# Codex Next-Step Brief: After the Desingularized Test, Treat the Result as a Radial Projection Fold

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
latest main branch
```

Start from:

```text
Note/GPT_REVIEW_PROMPT_DESINGULARIZED_PHASE_SPACE_RESULT.md
Note/CODEX_SECOND_CRITICAL_DESINGULARIZED_DAE_PLAN.md
outputs/tables/transonic_desingularized_barrier_flow_ds5e4.md
outputs/tables/transonic_desingularized_barrier_flow_ds5e4_trace.json
scripts/run_transonic_desingularized_barrier_flow.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
tests/test_transonic_local.py
```

## Executive conclusion

The desingularized phase-space test has done its job. It shows that the exact
incoming branch does not simply fail because \(A^{-1}\) is numerically bad.
Instead, the regular phase-space curve turns around in the radial coordinate:

```text
R_max = 6.237899204 rg
p_x changes sign
B remains nonsingular enough along the incoming flow
physical quantities remain tame
```

This means:

```text
At fixed lambda0, the exact incoming branch is a regular phase-space curve
whose projection onto R has a local maximum.
```

For a smooth stationary disk profile \(y(R)\), this is a serious obstruction.
A monotonic outward disk branch cannot be obtained by merely continuing this
same curve through the turnaround.

The next step is not more tail weighting, not another Taylor patch, and not
high-Mdot continuation.

The next step is:

```text
1. verify the radial fold with high precision;
2. continue the whole critical solution family in lambda0;
3. search for an accessible true branch point where B becomes rank deficient;
4. only if such a branch point exists, formulate a branch-switching BVP.
```

---

# 1. Interpretation of the desingularized result

The desingularized system is

```math
z=(x,\ln u,\ln T),\qquad x=\ln R,
```

```math
p=\frac{dz}{ds},
```

```math
B(z;\lambda_0)p=0,
\qquad
B=[c\ A].
```

This avoids the singular ODE inversion

```math
g=\frac{dy}{dx}=-A^{-1}c.
```

The latest run reports:

```text
incoming_R6_plus:
    classification = R_turnaround
    R_start = 6 rg
    R_max   = 6.238 rg
    p_x flips once
    min |p_x| ~ 3e-8
    min smin/smax(A) ~ 2.6e-10
    min smin/smax(B) ~ 6.2e-6
    max H/R ~ 5.8e-4
    max |Omega/Omega_K| ~ 1
```

The key point is that \(A\) becomes nearly singular, but \(B\) does not collapse
to rank one for the incoming branch. Therefore the phase-space curve is still
well-defined, while \(R(s)\) reaches a maximum.

## Consequence

At fixed \(\lambda_0\), the exact incoming branch cannot be extended as a
single-valued outward radial profile beyond \(R_{\max}\). It turns back in
radius.

This is best interpreted as:

```text
a radial projection fold,
not yet a physical thermodynamic failure.
```

---

# 2. Do not call this a global physical termination yet

This result is not yet enough to conclude that the IMRI minidisk model fails.

It only says:

```text
this exact fixed-lambda no-wind branch cannot be followed monotonically
outward past R ~ 6.238 rg.
```

Other possibilities remain:

```text
1. A nearby global solution with a different lambda0 may avoid or move the fold.
2. There may be a true branch-switching point where B loses rank.
3. The current algebraic stress/no-wind closure may be missing physics needed
   to pass this region.
4. A discontinuous internal transition could exist, but that would be new
   physics and should not be introduced before exhausting smooth branches.
```

---

# 3. Immediate next experiment: high-precision fold verification

Before building another global solver, verify the fold as a local mathematical
object.

## 3.1 Add an event-refined fold finder

The current integration uses a fixed-step Heun method. Add a root finder for:

```math
p_x(s)=0.
```

Use the two neighboring saved points where \(p_x\) changes sign and solve for
the fold point using one of:

```text
bisection in s with re-integration,
secant in s,
or local collocation on a short segment.
```

Output:

```text
z_fold
R_fold/rg
p_fold
D,C1,C2,K
smin/smax(A)
smin/smax(B)
H/R
Omega/OmegaK
Qadv/Qvisc
```

## 3.2 Compute the second derivative of \(x(s)\)

At the fold:

```math
p_x=\frac{dx}{ds}=0.
```

Compute:

```math
\frac{d p_x}{ds}
```

by finite differencing the phase-space tangent field around the fold.

If:

```text
dpx/ds < 0
```

and \(p_x\) changes from positive to negative, the fold is a genuine local
maximum in \(R\).

Create:

```text
outputs/tables/transonic_desingularized_fold_refinement.md
outputs/figures/transonic_desingularized_fold_refinement.png
```

## 3.3 Robustness scans

Repeat the flow with:

```text
ds = 5e-4, 2e-4, 1e-4, 5e-5
```

and at least two phase-space metrics.

Acceptance:

```text
R_fold changes by < 1e-4 rg
D,C1,C2,K remain small or well-characterized
smin/smax(B) remains comfortably nonzero
physical diagnostics remain tame
```

If this fails, the fold is a numerical artifact. If it passes, proceed.

---

# 4. Continue the whole solution family in \(\lambda_0\)

The handoff reports two important nearby candidates:

```text
fixed-lambda candidate near R ~= 6.118 rg
free-lambda candidate near R ~= 6.219 rg with dlambda0 ~= +0.0012
```

For a single radial disk solution, \(\lambda_0\) is constant with radius. It
cannot change locally at a second critical point. Therefore the free-lambda
candidate should be interpreted as a neighboring global solution family, not
as a local escape hatch.

## 4.1 Lambda continuation map

For a grid of global eigenvalues:

```text
lambda0 = lambda_ref + delta_lambda
delta_lambda = -0.004, -0.003, -0.002, -0.001, 0,
                +0.001, +0.002, +0.003, +0.004
```

do:

1. solve or seed the first sonic critical point;
2. integrate the desingularized phase-space curve;
3. locate the first radial fold \(p_x=0\);
4. record whether \(R(s)\) reaches useful radii.

Output:

```text
outputs/tables/transonic_lambda_family_fold_map.md
outputs/figures/transonic_lambda_family_fold_map.png
```

Columns:

```text
lambda0
delta_lambda
R_son
R_fold
x_fold
p_x crossing?
smin/smax(B)_fold
dpx/ds_fold
max H/R
Omega/OmegaK at fold
Qadv/Qvisc at fold
status
```

## 4.2 Interpretation

### If \(R_{\rm fold}(\lambda_0)\) moves outward strongly

Then continue toward the branch that reaches \(R\gtrsim 7,10,15r_g\). The exact
branch may still connect to the outer disk at a neighboring \(\lambda_0\).

### If \(R_{\rm fold}\) remains near \(6.2r_g\)

Then the no-wind algebraic-stress branch probably cannot produce a monotonic
radial solution beyond that region.

### If the fold disappears

Then restart global BVP construction on that \(\lambda_0\) branch.

---

# 5. Search for a true branch-switching point

A branch switch is not possible at an ordinary projection fold where \(B\) has
rank two. At rank two, the nullspace of \(B\) is one-dimensional, so there is
only one tangent direction, up to sign.

A genuine smooth branch switch requires:

```text
rank(B) < 2
```

so that the nullspace of \(B\) becomes two-dimensional.

## 5.1 Rank-one condition for B

For:

```math
B=
\begin{pmatrix}
b_{00}&b_{01}&b_{02}\\
b_{10}&b_{11}&b_{12}
\end{pmatrix},
```

rank \(B\le 1\) means all \(2\times2\) minors vanish:

```math
M_{01}=b_{00}b_{11}-b_{01}b_{10}=0,
```

```math
M_{02}=b_{00}b_{12}-b_{02}b_{10}=0,
```

```math
M_{12}=b_{01}b_{12}-b_{02}b_{11}=0.
```

Alternatively use:

```math
\sigma_{\min}(B)=0.
```

For a smooth residual, the minors are often easier.

## 5.2 Branch-point search

Search near the reported fixed-lambda and free-lambda candidates.

Unknowns:

```text
x_b
logu_b
logT_b
lambda0
```

Residuals:

```text
M01 = 0
M02 = 0
M12 = 0
```

plus one anchoring condition to stay near the incoming branch, for example:

```text
distance to incoming desingularized curve minimized,
or
x_b fixed in a local scan.
```

Better first diagnostic:

```text
minimize M01^2 + M02^2 + M12^2
```

over a box around each candidate and report the minimum.

Create:

```text
outputs/tables/transonic_B_rank_defect_search.md
outputs/figures/transonic_B_rank_defect_search.png
```

## 5.3 Accessibility test

A branch point is useful only if the incoming exact branch can reach it.

For each B-rank-defect candidate, compute:

```text
minimum distance from incoming desingularized trajectory to z_b
```

in a scaled metric.

If the distance is large, that candidate is not on the current branch.

---

# 6. If an accessible B-rank-defect point exists: formulate a branch-switching BVP

## 6.1 Unknowns

Use two phase-space segments:

```text
Segment 1: first sonic point -> branch point
Segment 2: branch point -> outer/tail disk
```

At the branch point include:

```text
z_b = (x_b, logu_b, logT_b)
p_- incoming tangent
p_+ outgoing tangent
lambda0
```

## 6.2 Residuals at branch point

```math
B(z_b)p_- = 0,
```

```math
B(z_b)p_+ = 0,
```

```math
||p_-||=1,
\qquad
||p_+||=1,
```

```math
M_{01}=M_{02}=M_{12}=0
```

or equivalent rank-defect residuals.

For a true branch switch, impose:

```math
|p_- \cdot p_+| < 1-\epsilon
```

as an inequality diagnostic, not a hard equality.

For a smooth crossing test, set:

```math
p_+=p_-
```

but this should be a separate run.

## 6.3 Outgoing branch condition

The outgoing tangent must satisfy:

```text
p_{+,x} > 0
```

if it is to continue outward to larger \(R\).

If every outgoing null tangent has \(p_x \le 0\), there is no outward branch.

---

# 7. If no accessible branch point exists: do not force a free tail

The short-join scans and tail-weighted solves stalled because they tried to
attach an outer tail to a branch that does not continue monotonically in \(R\).
A free tail with derivative discontinuity is not a smooth solution of the same
first-order system.

Therefore:

```text
Do not accept a solution that matches y but jumps in p or dy/dx at R_join
unless you explicitly introduce a physical shock/source/jump condition.
```

That would be new physics, not the same smooth slim-disk branch.

---

# 8. Possible physical interpretation if fixed-lambda branches all fold

If the \(\lambda_0\) family scan shows that all relevant no-wind branches fold
near \(R\simeq6r_g\), then the current closure likely lacks a smooth global
no-wind transonic solution at \(\dot M/\dot M_{\rm Edd}\simeq0.9\).

Then the next physics choices are:

```text
1. change stress closure;
2. include wind/advection terms consistently in the transonic equations;
3. allow an internal shock or dissipation layer with jump conditions;
4. reconsider whether the high-state branch is time-dependent rather than a
   stationary slim-disk branch.
```

But make that decision only after the \(\lambda_0\) family and rank-defect
audits are complete.

---

# 9. Recommended immediate experiment order

## Experiment A: refined fold event

```text
scripts/run_transonic_desingularized_fold_refinement.py
```

Inputs:

```text
incoming_R6_plus branch
lambda0 fixed
ds scan
metric scan
```

Outputs:

```text
transonic_desingularized_fold_refinement.md
```

## Experiment B: lambda-family fold map

```text
scripts/run_transonic_lambda_family_fold_map.py
```

Outputs:

```text
transonic_lambda_family_fold_map.md
```

## Experiment C: B-rank-defect search

```text
scripts/run_transonic_B_rank_defect_search.py
```

Outputs:

```text
transonic_B_rank_defect_search.md
```

## Experiment D: accessibility of rank-defect candidates

```text
scripts/run_transonic_rank_defect_accessibility.py
```

Outputs:

```text
transonic_rank_defect_accessibility.md
```

## Experiment E: branch-switching BVP only if C/D pass

```text
scripts/run_transonic_branch_switching_phase_space_bvp.py
```

---

# 10. Concrete code additions

## `transonic_local.py`

Add:

```python
def B_rank_minors(logR, y, lambda0, params):
    B, A, c = extended_phase_space_matrix(logR, y, lambda0, params)
    m01 = B[0,0] * B[1,1] - B[0,1] * B[1,0]
    m02 = B[0,0] * B[1,2] - B[0,2] * B[1,0]
    m12 = B[0,1] * B[1,2] - B[0,2] * B[1,1]
    return np.array([m01, m02, m12])
```

Add:

```python
def phase_space_tangent_derivative(logR, y, lambda0, params, p, eps=1e-5):
    # finite-difference derivative of normalized tangent field along p
    z = np.array([logR, y[0], y[1]])
    p_plus = phase_space_null_tangent(z[0]+eps*p[0], z[1:]+eps*p[1:], lambda0, params, previous=p).tangent
    p_minus = phase_space_null_tangent(z[0]-eps*p[0], z[1:]-eps*p[1:], lambda0, params, previous=p).tangent
    return (p_plus - p_minus) / (2*eps)
```

## New scripts

```text
scripts/run_transonic_desingularized_fold_refinement.py
scripts/run_transonic_lambda_family_fold_map.py
scripts/run_transonic_B_rank_defect_search.py
scripts/run_transonic_rank_defect_accessibility.py
scripts/run_transonic_branch_switching_phase_space_bvp.py
```

---

# 11. Acceptance criteria

## Projection fold confirmed if

```text
R_fold converges under ds and metric changes;
p_x = 0 at fold;
dp_x/ds has stable nonzero sign;
smin/smax(B) remains > 1e-8 or otherwise clearly nonzero;
physical quantities remain finite.
```

## Lambda branch viable if

```text
R_fold(lambda0) moves outward enough to allow connection to outer disk,
or p_x remains positive to R_match.
```

## Branch-switching viable if

```text
rank(B)<2 at accessible point;
incoming exact branch reaches that point;
outgoing tangent with p_x>0 exists;
phase-space BVP residuals fall below few x 1e-6;
solution is stable under grid refinement.
```

## No smooth no-wind branch if

```text
all lambda0 branches fold near R~6;
no accessible rank-defect point with outward tangent exists;
and physical diagnostics remain non-pathological.
```

Then move to revised physics rather than more numerics.

---

# 12. Compact Codex prompt

```text
The desingularized phase-space test found an R-turnaround at
R=6.237899 rg. B remains regular enough while p_x changes sign, so this is a
projection fold of the exact fixed-lambda branch, not just an ODE-inversion
failure. At fixed lambda0, this branch cannot be used as a monotonic outward
disk profile beyond Rmax.

Next:
1. Refine the fold event p_x=0 with smaller ds and different metrics.
2. Compute dp_x/ds at the fold to confirm a true maximum in R(s).
3. Continue the whole critical solution family in lambda0 and map R_fold(lambda0).
4. Search for accessible B-rank-defect branch points by minimizing the three
   2x2 minors of B=[c A].
5. Only if an accessible rank-defect point exists, build a branch-switching
   phase-space BVP with incoming and outgoing tangents.
6. Do not attach a free tail with derivative discontinuity unless adding an
   explicit physical jump/shock/source condition.
7. If all lambda branches fold and no branch point exists, conclude that the
   present no-wind algebraic-stress closure lacks a smooth monotonic fixed-Mdot
   branch at this rate, then revise closure or add wind.
```
