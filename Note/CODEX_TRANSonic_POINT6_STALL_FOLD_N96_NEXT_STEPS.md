# Codex Next-Step Brief: Diagnose and Cross the \(\dot M/\dot M_{\rm Edd}\simeq1.05\) Transonic Stall

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
```

Start from:

```text
outputs/tables/transonic_point6_polish_resume.md
scripts/run_transonic_point6_polish_resume.py
src/imri_qpe/layer3_minidisk_1d/transonic_continuation.py
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
```

## Executive diagnosis

The current run has **not demonstrated a physical termination of the slim-disk branch near**
\(\dot M/\dot M_{\rm Edd}\simeq1.05\).

The stronger interpretation is:

```text
The continuation is following a loose residual tube around the critical
manifold, while the sonic equations and bordered Jacobian become increasingly
ill-conditioned. The N96 tests are not yet valid grid-convergence tests because
they start from derivative-inconsistent interpolation of N64 profiles.
```

Three facts dominate the diagnosis:

1. Fixed-\(\dot M\) polishing cannot reduce the N64 square residual below
   \(\simeq1.6\!-\!1.7\times10^{-4}\), despite a requested target of \(10^{-6}\).

2. The resumed points are accepted at the deliberately loose threshold
   \(3\times10^{-4}\), with the determinant residual \(D\) sitting almost
   exactly at that threshold. Meanwhile the unused algebraic compatibility
   residual reaches \(|C_2|\simeq6\times10^{-4}\).

3. The raw bordered condition estimate grows from
   \(\sim7\times10^{10}\) to \(\sim9\times10^{13}\), while
   \(\dot M/\dot M_{\rm Edd}\) barely changes from \(1.037\) to \(1.051\).

Near the stall the disk is only moderately thick:

```text
max H/R ~= 0.165
integrated Qadv/Qvisc ~= 0.108
```

so there is no obvious physical reason for the stationary no-wind slim branch
to end there. A genuine fold is possible, but the current implementation cannot
yet distinguish a fold from sonic-residual/Jacobian pathology.

Do not add wind, stream feeding, tides, or time dependence yet.

---

# 1. What the current table actually establishes

The accepted N64 resume sequence is:

| point | \(\dot M/\dot M_{\rm Edd}\) | max residual | dominant | condition estimate |
|---|---:|---:|---|---:|
| resume 1 | 1.037 | \(2.987\times10^{-4}\) | \(D\) | \(6.65\times10^{10}\) |
| resume 2 | 1.046 | \(2.959\times10^{-4}\) | \(D\) | \(4.53\times10^{11}\) |
| resume 3 | 1.050 | \(2.984\times10^{-4}\) | \(D\) | \(1.08\times10^{13}\) |
| resume 4 | 1.050 | \(3.000\times10^{-4}\) | \(D\) | \(5.64\times10^{13}\) |
| resume 5 | 1.051 | \(3.000\times10^{-4}\) | \(D\) | \(9.31\times10^{13}\) |

This is not conventional Newton convergence toward \(F=0\). It is convergence
to the active acceptance boundary:

```text
max residual ~= RESIDUAL_TOL = 3e-4.
```

The failed \(1.2\) probe reaches only \(1.052\), with residual
\(3.12\times10^{-4}\).

Therefore:

```text
The current continuation frontier is tolerance-limited, not demonstrably
branch-limited.
```

---

# 2. The current “physical” classification is too permissive

In `_status_from_profile`, sonic regularity uses:

```text
abs(D) <= tol
abs(K) <= tol
smin/smax <= tol
```

but does not require both algebraic compatibility diagnostics to be small.

At the resumed points:

```text
K  ~ 1e-6 or smaller
C1 ~ 1.5e-4
C2 ~ 6e-4
```

A true rank-one compatible sonic system should make:

```text
D  -> 0
C1 -> 0
C2 -> 0
```

up to discretization error.

## Required status change

Define a stricter sonic validity measure:

```python
sonic_compatibility_max = max(abs(C1), abs(C2))
```

and require:

```python
sonic_regular = (
    abs(D) <= sonic_tol
    and sonic_compatibility_max <= sonic_tol
    and smin_over_smax <= sonic_tol
    and null_radial_fraction > 0.3
)
```

Use separate tolerances:

```text
development:
    equation_tol = 3e-4
    sonic_tol    = 1e-5

production anchor:
    equation_tol = 1e-6
    sonic_tol    = 1e-8 to 1e-6
```

Do not call the current resume points “physical anchors” until the unused
compatibility residual is also controlled.

---

# 3. Stop using the SVD compatibility \(K\) as the production sonic equation

The script explicitly uses:

```python
sonic_pivot="K"
```

and `_resolve_sonic_pivot("auto")` also returns `"K"`.

This defeats the previously implemented algebraic pivot logic.

The SVD compatibility

```math
K=u_{\rm left}^{T}c/\|c\|
```

is useful as a diagnostic, but its derivative is poorly conditioned when the
small singular vector rotates. It can be nearly zero while \(D\), \(C_1\), and
\(C_2\) are not all sufficiently small.

That is exactly what the current table shows.

## Required change

Use a square sonic pair:

```text
[D, C_selected]
```

where `C_selected` is frozen during each Newton/corrector solve.

Use the existing column-norm pivot rule:

```python
pivot = select_sonic_compatibility_pivot(z_anchor, params)
```

Then call:

```python
square_collocation_residual(..., pivot=pivot)
square_collocation_jacobian(..., pivot=pivot)
```

Keep:

```text
K
unused C
```

as diagnostics only.

## Cross-pivot audit

At \(\dot M/\dot M_{\rm Edd}=0.90,0.965,0.996\), run separate fixed-\(\dot M\)
polishes with:

```text
pivot = C1
pivot = C2
```

A real solution should converge to essentially the same profile and make the
unused compatibility equation small.

If C1 and C2 lead to different profiles, the sonic system is not yet resolved.

---

# 4. First determine whether the stall is a genuine fold

A simple fold in \(\mu=\ln\dot M\) has a clear numerical signature.

Let:

```math
J_z = \frac{\partial F}{\partial z},
\qquad
F_\mu = \frac{\partial F}{\partial\mu}.
```

Let \(u_{\min},v_{\min}\) be the left/right singular vectors of \(J_z\)
associated with its smallest singular value.

Define:

```math
\sigma_{\min}=\sigma_{\min}(J_z),
```

```math
a=u_{\min}^{T}F_\mu,
```

and record the pseudo-arclength tangent component:

```math
t_\mu=\frac{d\mu}{ds}.
```

## Simple-fold signature

```text
sigma_min(J_z) -> 0
a = u_min^T F_mu remains nonzero
t_mu -> 0 and changes sign after the fold
bordered Jacobian remains nonsingular
```

## Numerical-pathology signature

```text
raw and equilibrated condition estimates disagree by many orders of magnitude
bordered Jacobian also becomes singular
tangent is sensitive to finite-difference step or sonic pivot
unused sonic compatibility stays large
```

## Higher-codimension signature

```text
sigma_min(J_z) -> 0
u_min^T F_mu -> 0
bordered Jacobian also becomes singular
```

This could indicate a cusp/branch point, but do not claim that until Jacobian
accuracy and sonic compatibility are validated.

## Add a fold-audit table

Create:

```text
outputs/tables/transonic_fold_audit.md
```

with:

```text
Mdot/Edd
sigma_min(Jz)
sigma_2(Jz)
cond_equilibrated(Jz)
u_min^T F_mu
t_mu
angle(t_new,t_old)
cond_equilibrated(bordered J)
D
C1
C2
K
```

---

# 5. Two current code paths explicitly prevent fold crossing

Even if a genuine fold exists, the current code is biased against crossing it.

## 5.1 Remove the monotonic-\(\dot M\) acceptance condition

The resume script requires:

```python
result.mdot_ratio > current_ratio * 1.00005
```

That is incompatible with pseudo-arclength continuation through a fold, where
\(\dot M\) must stop increasing and then decrease while arclength continues
forward.

Replace it with:

```python
forward_arc = dot((w_new - w_current) / scales, tangent) > 0
accepted = physically_valid and residual_ok and forward_arc
```

Do not require monotonic \(\dot M\).

## 5.2 Remove the forced-\(\mu\) tangent fallback near a fold

In `_jacobian_scaled_tangent`, when \(|t_\mu|\) is small, the code constructs a
fallback tangent with a forced sign of \(\mu\).

Near a real fold, \(t_\mu\to0\) is the expected behavior.

The fallback can push the predictor along the wrong direction and make the
bordered matrix nearly singular.

Required behavior:

```text
- accept a small t_mu;
- orient the tangent only by its metric dot product with the previous tangent;
- do not enforce the previous sign of t_mu;
- allow t_mu to cross zero.
```

Keep the forced-\(\mu\) tangent only as an optional debugging mode, disabled in
production.

---

# 6. The reported condition number is not yet a trustworthy physical condition number

`_condition_estimate()` currently applies `numpy.linalg.cond()` to the bordered
matrix after column scaling by the continuation metric, but before iterative
row/column equilibration.

Therefore the reported values:

```text
1e10--1e14
```

mix:

```text
true branch conditioning
residual-row scaling
variable scaling
sonic-pivot sensitivity
finite-difference Jacobian noise
```

## Required conditioning audit

For both \(J_z\) and the bordered Jacobian:

1. Store the raw matrix.
2. Apply 3--5 Ruiz row/column equilibration iterations.
3. Compute singular values of the equilibrated dense matrix for N64/N96.
4. Report both raw and equilibrated condition estimates.
5. Inspect the smallest left/right singular vectors by variable block.

At N64/N96, dense SVD is computationally feasible and is much more informative
than a single raw condition number.

## Singular-vector localization

Report the norm fraction of \(v_{\min}\) in:

```text
logu nodes
logT nodes
logR_son
lambda0
mu
```

If the near-null vector is dominated by:

```text
R_son/lambda0 -> sonic eigenparameter degeneracy
outer nodes    -> outer boundary problem
one interval   -> mesh/discretization defect
mu             -> continuation metric/tangent problem
```

---

# 7. Improve the bordered linear solve

At every reported resume point the raw condition estimate exceeds \(10^8\), so
the code never uses sparse LU. It switches to:

```python
lsmr(..., damp=1e-8)
```

with a fixed damping.

That is not robust for matrices whose condition estimate changes from
\(10^{10}\) to \(10^{14}\).

## Immediate diagnostic solver

For N64 and N96, use dense rank-revealing SVD or QR for the bordered Newton
step:

```python
U, s, Vt = scipy.linalg.svd(J_eq, full_matrices=False)
cut = rcond * s[0]
step_eq = -Vt.T @ ((U.T @ residual_eq) / maximum(s, cut))
```

Scan:

```text
rcond = 1e-12, 1e-10, 1e-8, 1e-6
```

Choose the least regularized step that decreases every critical residual block.

This is for diagnosis and branch recovery, not necessarily the final large-N
solver.

## Production solver

After scaling is fixed:

```text
- use sparse QR if available;
- otherwise use equilibrated GMRES/LSMR with adaptive Tikhonov damping;
- select damping from predicted-vs-actual merit reduction;
- do not hard-code damp=1e-8.
```

Use a trust-region or Levenberg-Marquardt parameter that increases when the
linear model is poor and decreases after successful Newton-like steps.

---

# 8. The fixed-\(\dot M\) polish failure must be solved before extending the branch

The current anchors cannot be polished below:

```text
1.58e-4 at Mdot/Edd=0.903
1.68e-4 at Mdot/Edd=0.965
1.74e-4 at Mdot/Edd=0.996
```

This is a critical warning.

Before further continuation, pick one anchor, preferably \(0.90\), and answer:

```text
Does the square discrete system actually possess a root near this state?
```

## Root-existence audit

At fixed \(\dot M\):

1. Switch from K to C1 or C2 pivot.
2. Use an equilibrated dense SVD/QR Newton step.
3. Sweep Jacobian finite-difference steps.
4. Run a dense full-residual finite-difference Jacobian at small N.
5. Try N32, N48, N64.
6. Compare midpoint and higher-order collocation.
7. Check whether the minimum residual decreases with N.

Possible outcomes:

### A. Residual decreases with N

The floor is discretization error or interpolation error. Improve collocation
order/mesh.

### B. Residual is N-independent but pivot-dependent

The sonic closure/Jacobian is the problem.

### C. Residual is N-independent and pivot-independent

The discrete boundary-value problem may be inconsistent under the current
outer/sonic conditions, or the Newton Jacobian is inaccurate.

Do not use a point as a continuation anchor until:

```text
square residual < 1e-6
unused compatibility < 1e-6
```

or until a convergence study establishes a defensible discretization floor.

---

# 9. Global Jacobian accuracy is now decisive

The square Jacobian uses block-local finite differences. The sonic block uses a
Richardson finite difference with a step limited to \(10^{-6}\), while the
interval block uses a larger default \(3\times10^{-5}\).

At condition numbers near \(10^{12}\!-\!10^{14}\), finite-difference noise in
the smallest singular direction can completely determine the Newton step.

## Required directional audit

At each anchor near \(0.90,0.965,0.996,1.03\):

```math
\epsilon_J(h)
=
\frac{\|Jv-[F(z+hv)-F(z-hv)]/(2h)\|}
{\|Jv\|+\|[F(z+hv)-F(z-hv)]/(2h)\|+\epsilon}.
```

Scan:

```text
h = 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7
```

Use random directions and targeted directions:

```text
smallest singular vector
R_son direction
lambda0 direction
mu direction
sonic-node logu/logT directions
```

Target for a trustworthy near-null direction:

```text
directional error << sigma_min/sigma_max
```

If this cannot be achieved with finite differences, implement an exact
Jacobian for the sonic and interval blocks.

## Recommended exact-Jacobian path

Priority:

```text
1. analytic derivatives for D and selected C;
2. automatic differentiation or analytic derivatives for interval blocks;
3. exact F_mu rather than finite differencing in mu.
```

The sonic rows are the highest priority because they dominate the stall.

---

# 10. Why the N96 remap test fails

The N96 spot checks are explicitly “remap checks,” not independently solved
N96 solutions.

The current prolongation uses linear interpolation of:

```text
log u(log R)
log T(log R)
```

The energy equation depends on radial derivatives and entropy gradients.
Piecewise-linear interpolation changes those derivatives discontinuously at
coarse-grid knots.

Therefore a coarse collocation solution generally will **not** satisfy the
fine-grid collocation equations immediately after interpolation.

The huge values:

```text
N96 interval-energy residual ~ 7.8
Delta xi_eff ~ 140--165
```

mostly show derivative-inconsistent prolongation, not physical non-convergence.

## Do not use raw remap residual as the grid-convergence criterion

Grid convergence must compare independently converged solutions.

---

# 11. Use nested-grid, derivative-aware multilevel continuation

## 11.1 Prefer a nested mesh

N64 to N96 is not nested.

For a uniform computational coordinate with endpoints, use:

```text
N64 -> N127
```

because:

```text
127 = 2*(64-1)+1
```

so every N64 node is retained.

Alternatively migrate to:

```text
N65 -> N129
```

for a conventional power-of-two interval hierarchy.

## 11.2 Use Hermite/PCHIP prolongation

Best option:

```text
cubic Hermite interpolation in log R using nodal values and ODE slopes
```

For each coarse node obtain:

```text
d logu/d logR
d logT/d logR
```

from the local differential equations away from the sonic point and from the
regular sonic derivative at the first node.

If the sonic derivative is not yet available, use PCHIP as an interim
monotone smoother.

Do not use raw linear interpolation for a derivative-sensitive convergence
test.

## 11.3 Multilevel solve stages

At a rate where N64 is well conditioned, e.g. \(0.5\!-\!0.7\):

1. Prolong N64 to N127.
2. Hold \(\dot M,R_{\rm son},\lambda_0\) fixed and solve only nodal interval +
   outer-boundary equations.
3. Free \(R_{\rm son}\) with \(\lambda_0\) fixed.
4. Free \(\lambda_0\) and enforce the sonic pair.
5. Polish the full N127 square system.
6. Build two independent N127 anchors.
7. Continue the N127 branch toward \(1.05\).

Do not initialize N96/N127 pseudo-arclength directly from two unsolved
interpolated N64 profiles.

## 11.4 Convergence comparison

After both meshes are independently polished, compare in physical radius:

```text
R_son
lambda0
H/R
Sigma
T
Omega/Omega_K
xi_eff away from the sonic cell
integrated Qadv/Qvisc
```

The current `N96 dxi` based on raw interpolation should not be used as a
scientific convergence metric.

---

# 12. Improve continuation metric and step control

The blockwise metric is recomputed from only the most recent secant. Near a
stall, small noisy secants can change the metric and tangent geometry
artificially.

## Required changes

Store the metric in each checkpoint and update it using an exponential moving
average:

```math
s_{k+1}^2
=
(1-\eta)s_k^2+\eta(\Delta x_k)^2,
\qquad
\eta\simeq0.1-0.3.
```

Retain physical floors, but avoid resetting the entire metric from one tiny
step.

Use an explicit arclength step \(\Delta s\), rather than:

```python
step = previous_secant_norm * step_multiplier
```

Adapt \(\Delta s\) using:

```text
predictor-corrector distance
Newton iteration count
line-search reductions
tangent angle
```

Near a fold, \(\Delta\mu\) can vanish while \(\Delta s\) remains finite.

---

# 13. Recommended immediate experiment sequence

## Experiment A: N64 sonic/root audit

At \(\dot M/\dot M_{\rm Edd}=0.90\):

```text
pivot C1
pivot C2
pivot K diagnostic only
```

For each:

```text
N32, N48, N64
dense SVD Newton
Jacobian step scan
residual target 1e-8
```

Goal:

```text
find one actual square root and make D,C1,C2,K all small.
```

## Experiment B: fold discrimination

From polished N64 roots at \(0.90,0.95,1.00\), compute:

```text
sigma_min(Jz)
u_min^T F_mu
t_mu
equilibrated bordered condition
```

Remove monotonic-\(\dot M\) acceptance and forced-\(\mu\) tangent logic.

Continue in arclength even if \(\dot M\) decreases.

## Experiment C: N127 bootstrap

Start at \(0.5\!-\!0.7\), not \(0.96\!-\!1.0\).

Use nested derivative-aware prolongation and staged release of eigenparameters.

## Experiment D: only then resume toward high rate

Once N64 and N127 agree, continue through the possible fold and determine
whether the branch:

```text
turns back in Mdot
continues to larger Mdot
or reaches a genuine higher-codimension singularity.
```

---

# 14. Code changes by file

## `transonic_collocation.py`

Change:

```python
def _resolve_sonic_pivot(...):
    if pivot == "auto":
        return select_sonic_compatibility_pivot(...)
```

Do not map `auto` to `K`.

Add:

```python
def strict_sonic_validity(audit, tol):
    return (
        abs(audit.sonic_D) < tol
        and max(abs(audit.sonic_C1), abs(audit.sonic_C2)) < tol
        and audit.sonic_smin_over_smax < tol
    )
```

Add exact/equilibrated Jacobian diagnostics:

```python
def equilibrated_svd_diagnostics(J):
    ...

def fixed_mdot_fold_diagnostics(z, params, pivot):
    ...
```

Update physical status to require unused compatibility.

## `transonic_continuation.py`

Remove or disable:

```text
jacobian_mu fallback that forces nonzero t_mu
```

Add:

```python
def fold_audit(Jz, Fmu, tangent):
    ...

def ruiz_equilibrate(A, n_iter=5):
    ...

def svd_bordered_step(Aeq, req, rcond):
    ...

def accept_forward_arclength(w_new, w_current, tangent, scales):
    ...
```

Store and smooth the continuation metric.

## `run_transonic_point6_polish_resume.py`

Remove:

```python
result.mdot_ratio > current_ratio * 1.00005
```

Replace with forward-arclength acceptance.

Switch:

```python
sonic_pivot="K"
```

to a frozen algebraic pivot selected from the anchor.

Do not mark `physical=True` unconditionally for loaded source anchors.

Do not run N96 pseudo-arclength directly from unsolved N64 interpolants.

## New script

Create:

```text
scripts/run_transonic_fold_and_multilevel_audit.py
```

Outputs:

```text
outputs/tables/transonic_fold_audit.md
outputs/tables/transonic_multilevel_audit.md
outputs/figures/transonic_fold_singular_values.png
outputs/figures/transonic_n64_n127_profiles.png
```

---

# 15. Go/no-go criteria

## Proceed past \(1.05\) only if

```text
- at least two neighboring N64 anchors are true square roots;
- D, C1, C2, and K are all below the adopted sonic tolerance;
- the Jacobian directional audit passes;
- the fold audit identifies either a regular branch or a simple fold;
- acceptance is based on arclength, not monotonic Mdot;
- the equilibrated bordered condition is controlled;
- an independently converged fine-grid branch agrees with N64.
```

## Interpret a physical fold only if

```text
sigma_min(Jz) -> 0
u_min^T F_mu remains nonzero
t_mu crosses zero
bordered J remains nonsingular after equilibration
N64 and N127 locate the fold consistently
```

## Interpret a physical branch termination only if

```text
the square roots and Jacobians are validated,
the result is mesh-converged,
and the bordered system remains singular after correct arclength treatment.
```

The current calculation does not meet those conditions.

---

# 16. Compact Codex prompt

```text
Diagnose the transonic continuation stall near Mdot/Edd~1.05 before adding new
physics.

1. Stop using sonic pivot K in the square solver. Use D plus a frozen
   algebraic compatibility pivot C1 or C2 selected at the anchor. Keep K and
   the unused C as diagnostics.

2. Tighten physical validity: require max(|C1|,|C2|), |D|, and smin/smax all
   below the sonic tolerance. Current resume points with C2~6e-4 must not be
   labeled physical at tol=3e-4.

3. Remove the acceptance condition requiring Mdot to increase. Accept forward
   arclength even if Mdot turns over.

4. Disable the tangent fallback that forces a nonzero/sign-preserving t_mu.
   Near a fold, t_mu must be allowed to approach and cross zero.

5. At fixed Mdot=0.90,0.965,0.996, solve the square system to <1e-6 (preferably
   1e-8) using C1 and C2 pivots, equilibrated dense SVD/QR Newton steps, and a
   Jacobian finite-difference step audit.

6. Compute a fold audit:
       sigma_min(Jz)
       left/right singular vectors
       u_min^T F_mu
       t_mu
       raw and equilibrated bordered condition
   A simple fold has sigma_min(Jz)->0, u_min^T F_mu != 0, t_mu->0, while the
   bordered Jacobian remains regular.

7. Apply Ruiz row/column equilibration before estimating condition numbers.
   Replace fixed-damping LSMR with dense rank-revealing SVD/QR for N64/N96
   diagnosis, then use adaptive damping for production.

8. Do not treat raw N64->N96 interpolation residuals as grid convergence.
   Use a nested mesh N64->N127, Hermite/PCHIP prolongation, then staged N127
   solving: profile only, free Rson, free lambda0, full sonic system. Start
   the N127 branch at Mdot/Edd~0.5--0.7 and continue independently.

9. Store an EMA-smoothed continuation metric and explicit ds. Do not scale ds
   by the latest tiny secant norm near the stall.

10. Resume high-rate continuation only after true N64 roots, a passing Jacobian
    audit, and an independently converged fine-grid branch are available.
```

---

# 17. Bottom line

The current result is best described as:

```text
The N64 pseudo-arclength code reaches a numerically ill-conditioned critical
region near Mdot/Edd~1.05, but the anchors are not polished roots, the sonic
compatibility is incompletely enforced, the implementation prevents a genuine
fold from being crossed, and the N96 tests start from derivative-inconsistent
interpolation. The observed stall is therefore numerical/continuation-limited,
not yet a physical end of the slim-disk branch.
```
