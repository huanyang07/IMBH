# Codex Plan: Replace the Dynamic First-Order Sonic Patch with a Sonic-Regular Taylor Patch

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
latest main branch
```

Primary handoff and output files reviewed:

```text
Note/GPT_SONIC_REGULARITY_HANDOFF.md
Note/CODEX_TWO_DOMAIN_INNER_REFINEMENT_NEXT_STEPS.md
outputs/tables/transonic_two_domain_inner_residual_profile.md
outputs/tables/transonic_two_domain_sonic_scaling.md
outputs/tables/transonic_two_domain_sonic_buffer_refinement.md
outputs/tables/transonic_two_domain_dynamic_sonic_patch.md
outputs/tables/transonic_two_domain_sonic_microdomain_scan.md
outputs/tables/transonic_two_domain_sonic_microdomain.md
scripts/run_transonic_two_domain_dynamic_sonic_patch.py
scripts/run_transonic_two_domain_sonic_microdomain.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
```

## Executive assessment

The project has made real progress. The two-domain pressure-supported outer
extension is no longer the main blocker. The remaining blocker is local to the
sonic endpoint.

Current state:

```text
Target fixed-Mdot root:
    Mdot/Mdot_Edd ~= 0.90277664
    R_son/rg ~= 5.8
    lambda0 ~= 3.675--3.676
    int_adv ~= 0.07--0.08
```

The latest handoff states the key numerical question correctly:

```text
How do we obtain a mesh-robust sonic regular solution when refining the inner
grid near R_son?
```

The current experiments show:

```text
1. Fixed-slope sonic buffer:
   gives a good local Nreg64 root, but fails under regular-grid refinement.

2. Dynamic first-order sonic patch:
   prevents the basin jump, but its first-order finite-patch truncation error
   becomes the residual floor.

3. Naive sonic micro-domain:
   diagnostically useful, but worse than the dynamic patch, because midpoint
   collocation too close to the singular sonic endpoint produces large micro_R
   residuals.
```

Therefore the next implementation should be:

```text
A sonic-regular Taylor patch whose derivative is selected by the singular
regularity condition, not by finite differencing across a finite patch.
```

Do not resume high-Mdot continuation yet.

---

# 1. Why the current dynamic patch fails

The dynamic patch currently infers the sonic slope from the finite buffer:

```math
g_s \approx \frac{y_b-y_s}{\Delta_s},
```

and constrains:

```math
A_s g_s + c_s = 0.
```

This is better than a fixed slope, but it is still only first order over a
finite patch. The latest dynamic patch table shows:

```text
Nreg64:
    physical ~1.35e-5, dominant C2

Nreg80:
    physical ~4.8e-5, dominant patch

Nreg96:
    physical ~1.5e-4, dominant patch

Nreg112:
    physical ~4.2e-4, dominant patch

Nreg128:
    physical ~5.7e-4, dominant patch
```

The patch residual grows and eventually dominates. This is exactly the expected
failure mode if the finite patch is first-order accurate and the inferred slope
is not the true regular sonic derivative.

The micro-domain experiment confirms that simply replacing the patch with
ordinary collocation intervals is not enough:

```text
N_micro=4:
    physical ~4.3e-4, dominant micro_R

N_micro=2:
    physical ~3.9e-4, dominant micro_R

N_micro=1:
    physical ~2.6e-4, dominant C2
```

The problem is the singular endpoint, not just the number of intervals.

---

# 2. Mathematical structure at the sonic point

The local transonic differential system can be written as:

```math
F(x,y,g;\lambda_0)
=
A(x,y;\lambda_0) g + c(x,y;\lambda_0)
=
0,
```

where:

```text
x = ln R
y = (ln u, ln T)
g = dy/dx = (dlnu/dlnR, dlnT/dlnR)
```

At a regular sonic point:

```math
\det A_s = 0,
```

and the system is singular. A regular solution exists only if the right-hand
side is compatible with the range of \(A_s\).

Let:

```math
A_s r = 0,
```

```math
l^T A_s = 0,
```

where \(r\) and \(l\) are the right and left null vectors of the scaled local
matrix at the sonic point.

The condition:

```math
A_s g_s + c_s = 0
```

does not uniquely determine \(g_s\), because \(A_s\) has rank one. The solution
set is:

```math
g_s = g_p + a r,
```

where \(g_p\) is one particular solution and \(a\) is an unknown scalar.

The scalar \(a\) is determined by differentiating the local equation along the
solution and demanding regularity. This is the sonic L'Hopital condition.

---

# 3. Sonic L'Hopital condition

Define:

```math
B_s(g)
=
\left[
\frac{\partial A}{\partial x}
+
\frac{\partial A}{\partial y}\cdot g
\right]g
+
\frac{\partial c}{\partial x}
+
\frac{\partial c}{\partial y}\cdot g.
```

Differentiating:

```math
A g + c = 0
```

along the solution gives:

```math
A_s h_s + B_s(g_s) = 0,
```

where:

```math
h_s = d^2y/dx^2
```

at the sonic point.

Since \(A_s\) is singular, this equation has a solution only if:

```math
l^T B_s(g_s) = 0.
```

This is the missing condition that selects the physical sonic derivative.

The key implementation target is therefore:

```math
\boxed{
\det A_s = 0,\qquad
A_s g_s+c_s=0,\qquad
l^T B_s(g_s)=0.
}
```

This replaces the current finite-difference slope condition.

---

# 4. How to compute \(B_s(g)\) robustly without deriving every partial derivative

Do not start by deriving analytic partials by hand. Use a directional derivative of the already implemented local residual.

Define:

```python
def local_F(logR, y, g, lambda0, params):
    A, c, *_ = scaled_differential_matrix(logR, y, lambda0, params)
    return A @ g + c
```

Then:

```math
B_s(g)
=
\frac{d}{d\epsilon}
F(x_s+\epsilon,\; y_s+\epsilon g,\; g;\lambda_0)
\bigg|_{\epsilon=0}.
```

Numerically:

```python
def directional_B(logR_s, y_s, g_s, lambda0, params, eps=1e-5):
    Fp = local_F(logR_s + eps, y_s + eps * g_s, g_s, lambda0, params)
    Fm = local_F(logR_s - eps, y_s - eps * g_s, g_s, lambda0, params)
    return (Fp - Fm) / (2.0 * eps)
```

Scan:

```text
eps = 3e-4, 1e-4, 3e-5, 1e-5, 3e-6
```

and choose the plateau. This is much simpler than implementing all partials
immediately and is good enough for the next sprint.

Later, if finite-difference noise becomes limiting, replace this directional
derivative with analytic/JAX partials.

---

# 5. Stage 1 implementation: sonic derivative unknown \(g_s\)

## 5.1 Unknowns

Modify the dynamic sonic patch so that the sonic slope is an independent
unknown:

```text
g_s = (g_u_s, g_T_s)
```

Do **not** infer it from:

```math
(y_b-y_s)/\Delta_s.
```

The global unknown vector should include:

```text
existing two-domain variables
+ g_u_s
+ g_T_s
```

where:

```text
y_s = first inner node
y_b = buffer node
```

## 5.2 Residuals

Replace the current dynamic-patch block with:

### Sonic determinant

```math
D(A_s)=0.
```

Use the already scaled determinant.

### Sonic differential equation

```math
A_s g_s + c_s = 0.
```

This is two residuals.

### L'Hopital derivative regularity

```math
L_s(g_s)
=
l^T B_s(g_s)
=0.
```

This is one residual.

### First-order Taylor buffer relation

```math
y_b-y_s-\Delta_s g_s=0.
```

This is two residuals.

Total new sonic-patch residuals:

```text
D:                         1
A_s g_s + c_s:             2
L_s(g_s):                  1
Taylor buffer relation:    2
total:                     6
```

This has the same residual count as the current dynamic patch if the current
sonic block has four rows and patch block has two rows, but now \(g_s\) is a
real unknown rather than a finite-difference artifact.

## 5.3 Compatibility diagnostics

Do not use `K` as the production compatibility equation in this new patch.

Instead:

```text
solve: D, A_s g_s + c_s, L_s, Taylor buffer
audit: C1, C2, K, smin/smax
```

At a successful solution, \(C_1,C_2,K\) should all be small because
\(A_sg_s+c_s=0\) at singular \(A_s\) enforces compatibility.

## 5.4 Expected improvement

This should remove the spurious finite-buffer determination of the slope.

If successful, \(g_s\) should:

```text
- be stable as Nreg changes;
- be close to the coarse first-interval slope only at low order;
- not blow up to the huge values seen in the dynamic patch;
- make C1,C2,K small as diagnostics.
```

---

# 6. Stage 2 implementation: second-order sonic Taylor patch

If Stage 1 still leaves a \(\Delta_s\)-dependent floor, add a curvature
unknown:

```text
h_s = (h_u_s, h_T_s)
```

and use:

```math
y_b
=
y_s
+
\Delta_s g_s
+
\frac{1}{2}\Delta_s^2 h_s.
```

## 6.1 Preferred residuals

Use a midpoint patch residual rather than trying to derive the full second
derivative analytically.

Define:

```math
x_m = x_s+\frac{1}{2}\Delta_s,
```

```math
y_m = y_s+\frac{1}{2}\Delta_s g_s+\frac{1}{8}\Delta_s^2 h_s,
```

```math
g_m = g_s+\frac{1}{2}\Delta_s h_s.
```

Then enforce:

```math
F(x_m,y_m,g_m;\lambda_0)=0.
```

This gives two residuals.

The second-order sonic patch residual vector becomes:

```text
D(A_s) = 0                                      1 row
A_s g_s + c_s = 0                              2 rows
l^T B_s(g_s) = 0                               1 row
y_b - y_s - Δ g_s - 0.5 Δ^2 h_s = 0            2 rows
F(x_m, y_m, g_m) = 0                           2 rows
```

Total:

```text
8 residuals
```

Extra unknowns relative to the original state:

```text
g_s: 2
h_s: 2
```

This is an overdetermined but well-scaled least-squares patch. It is fine for
the current solver architecture.

## 6.2 Alternative curvature equation

Later, the midpoint equation can be replaced or supplemented by:

```math
A_s h_s + B_s(g_s)=0.
```

But because \(A_s\) is singular and \(h_s\) has a null-component ambiguity, the
midpoint formulation is easier and safer for the next sprint.

## 6.3 Expected convergence

For a fixed \(\Delta_s\), the second-order patch should reduce the patch floor
substantially compared with the first-order dynamic patch.

With a \(\Delta_s\) scan:

```text
Delta_s = 0.04, 0.03, 0.02, 0.015, 0.01
```

expect approximately:

```text
first-order patch:  patch error ~ Delta_s^2
second-order patch: patch error ~ Delta_s^3
```

The exact power may be degraded by nonlinear solver tolerance, but the trend
should improve.

---

# 7. How to select the physical sonic derivative

The equation \(A_sg_s+c_s=0\) plus \(l^TB_s(g_s)=0\) can have more than one
root. Use branch continuity to choose the physical root.

## 7.1 Local scalar root diagnostic

At the source N65 solution, compute:

```math
g_p = \text{minimum-norm solution of } A_sg=-c_s,
```

```math
r = \text{right null vector of } A_s,
```

```math
g(a)=g_p+a r.
```

Then plot or tabulate:

```math
L(a)=l^TB_s(g(a)).
```

Scan:

```text
a in [-500, 500]
```

or a range centered on the first-interval slope.

Find all sign changes and local minima.

Choose the root whose \(g(a)\) is closest to the existing coarse first-interval
slope or to the previous accepted sonic derivative.

This diagnostic should be implemented before the full global patch solve.

## 7.2 Output table

Create:

```text
outputs/tables/transonic_sonic_derivative_roots.md
```

Columns:

```text
case
a
g_u
g_T
L
distance_to_first_interval_slope
chosen
```

This will identify branch ambiguity early.

---

# 8. Scaling of the new residuals

Scale every sonic-patch row.

Recommended:

```text
D: already normalized determinant
A_s g_s+c_s: use scaled local differential matrix, so natural scale ~1
L_s: normalize by ||B_s|| + eps
Taylor y relation: divide each component by y_scale
midpoint F: use scaled local residual
```

For the Taylor relation use:

```text
scale_logu = max(abs(Delta_s*g_u_s), 1e-3)
scale_logT = max(abs(Delta_s*g_T_s), 1e-3)
```

or simply start with:

```text
scale_logu = 1e-2
scale_logT = 1e-2
```

and audit sensitivity.

If Taylor rows are over-weighted, the solver will revert to a finite-buffer
slope. If under-weighted, \(y_b\) will detach from the sonic expansion.

---

# 9. Do not make \(\Delta_s\) too large

The fixed buffer experiments found good local roots for:

```text
Delta_s = 0.02, 0.03, 0.05
```

but also showed shifts in \(R_{\rm son}\) and \(\lambda_0\) as \(\Delta_s\)
changes.

For the Taylor patch, use:

```text
Delta_s = 0.01, 0.015, 0.02, 0.03
```

Recommended baseline:

```text
Delta_s = 0.02
```

Use \(\Delta_s=0.01\) after the Stage 1/2 patch is stable.

Do not refine regular-domain nodes inside the patch. The patch is the local
analytic representation of the singular region.

---

# 10. Validation sequence

## 10.1 Single-root validation at Nreg64

Use the existing target:

```text
Mdot/Mdot_Edd = 0.90277664
R_match = 6500 rg
R_far = 1e5 rg
N_outer = 54
Nreg = 64
Delta_s = 0.02
```

Run:

```text
Stage 1 first-derivative Taylor patch
Stage 2 second-order Taylor patch
```

Targets:

```text
physical residual <= 5e-6 for Stage 1
physical residual <= 2e-6 for Stage 2
D,C1,C2,K all <= few x 1e-6
patch residual no longer dominant
R_son close to 5.8 rg
lambda0 close to 3.675--3.676
int_adv close to 0.073--0.075
```

## 10.2 Regular-domain refinement

Then hold the sonic patch fixed and refine the regular inner domain:

```text
Nreg = 64, 80, 96, 112, 128
```

Acceptance:

```text
physical residual remains <= few x 1e-6
R_son stable within 0.01--0.02 rg
lambda0 stable within 5e-4
int_adv stable within 1e-3--5e-3
g_s stable
h_s stable if using Stage 2
```

## 10.3 Delta-s convergence

Run:

```text
Delta_s = 0.03, 0.02, 0.015, 0.01
```

at fixed Nreg64 and Nreg96.

Acceptance:

```text
R_son, lambda0, int_adv converge as Delta_s decreases
patch residual decreases with expected order
```

## 10.4 Compare against micro-domain

Only after the Taylor patch works, use the micro-domain as a diagnostic, not as
the main solver. Its current midpoint form is too singular-sensitive to be the
production method.

---

# 11. Concrete code changes

## 11.1 `transonic_local.py`

Add:

```python
def local_scaled_residual(logR, y, g, lambda0, params):
    A, c, *_ = scaled_differential_matrix(logR, y, lambda0, params)
    return A @ g + c
```

Add:

```python
def sonic_null_vectors(logR, y, lambda0, params):
    A, c, *_ = scaled_differential_matrix(logR, y, lambda0, params)
    # return left null l, right null r, smin/smax, A, c
```

Add:

```python
def sonic_directional_B(logR, y, g, lambda0, params, eps=1e-5):
    Fp = local_scaled_residual(logR + eps, y + eps*g, g, lambda0, params)
    Fm = local_scaled_residual(logR - eps, y - eps*g, g, lambda0, params)
    return (Fp - Fm) / (2*eps)
```

Add:

```python
def sonic_lhopital_residual(logR, y, g, lambda0, params, eps=1e-5):
    l, r, ratio, A, c = sonic_null_vectors(logR, y, lambda0, params)
    B = sonic_directional_B(logR, y, g, lambda0, params, eps=eps)
    return dot(l, B) / (norm(B) + eps_floor)
```

Add diagnostic:

```python
def sonic_derivative_root_scan(logR, y, lambda0, params, g_reference):
    # compute g_p, r
    # scan a
    # return roots of l^T B(g_p+a r)
```

## 11.2 `transonic_collocation.py`

Add a new patch mode:

```text
sonic_patch_mode = "taylor1"
sonic_patch_mode = "taylor2"
```

For `taylor1`, unknowns include:

```text
g_u_s, g_T_s
```

Residual block:

```python
rows += [D]
rows += [A_s @ g_s + c_s]        # 2 rows
rows += [L_s(g_s)]               # 1 row
rows += [y_buffer - y_s - Delta_s*g_s]  # 2 rows
```

For `taylor2`, unknowns include:

```text
g_u_s, g_T_s, h_u_s, h_T_s
```

Residual block:

```python
rows += [D]
rows += [A_s @ g_s + c_s]
rows += [L_s(g_s)]
rows += [y_buffer - y_s - Delta_s*g_s - 0.5*Delta_s**2*h_s]

x_m = x_s + 0.5*Delta_s
y_m = y_s + 0.5*Delta_s*g_s + 0.125*Delta_s**2*h_s
g_m = g_s + 0.5*Delta_s*h_s
rows += [local_scaled_residual(x_m, y_m, g_m, lambda0, params)]
```

Keep `C1,C2,K` as audit quantities:

```text
do not include them in the production residual initially
```

unless diagnostics show compatibility is not controlled.

## 11.3 Script changes

Create:

```text
scripts/run_transonic_sonic_derivative_root_scan.py
scripts/run_transonic_two_domain_sonic_taylor_patch.py
scripts/run_transonic_two_domain_taylor_delta_scan.py
scripts/run_transonic_two_domain_taylor_refinement.py
```

Outputs:

```text
outputs/tables/transonic_sonic_derivative_roots.md
outputs/tables/transonic_two_domain_sonic_taylor_patch.md
outputs/tables/transonic_two_domain_taylor_delta_scan.md
outputs/tables/transonic_two_domain_taylor_refinement.md
```

---

# 12. Suggested experiment matrix

## Experiment A: derivative root scan

At the existing N65 two-domain source:

```text
scan a for g = g_p + a r
```

Output all L'Hopital roots.

Select the branch closest to the source first-interval slope.

## Experiment B: taylor1 patch at Nreg64

Run:

```text
Delta_s = 0.02
Nreg64
N_outer54
R_far=1e5
```

Compare with:

```text
dynamic first-order patch
```

Goal:

```text
taylor1 should improve or match Nreg64 and make g_s stable.
```

## Experiment C: taylor2 patch at Nreg64

Same setup.

Goal:

```text
patch residual no longer dominant
physical residual <= 2e-6
```

## Experiment D: regular refinement

Run:

```text
Nreg64 -> 80 -> 96 -> 112 -> 128
```

with the sonic Taylor patch fixed.

Goal:

```text
no growth of patch residual
no drift of Rson/lambda0/int_adv
```

## Experiment E: delta scan

Run:

```text
Delta_s = 0.03, 0.02, 0.015, 0.01
```

at Nreg64 and Nreg96.

Goal:

```text
convergent Delta_s trend.
```

---

# 13. How to interpret outcomes

## If taylor1 works

Then the main problem was not the finite patch order but the incorrect
finite-difference sonic slope. Continue with taylor1 and refine.

## If taylor1 fails but taylor2 works

Then the first-order patch truncation error was the dominant floor. Use
taylor2 as production.

## If both fail and C1/C2 remain large

Then the L'Hopital residual implementation or sonic matrix scaling is wrong.
Audit:

```text
A_s g_s + c_s
l^T B_s
D
C1,C2,K
```

and compare finite-difference eps.

## If both fail with interval residuals rather than sonic residuals

Then the regular inner collocation/prolongation is still the bottleneck.
Return to defect-preserving interval splitting.

## If taylor patches work at N64 but fail with Nreg refinement

Then the patch is okay, but regular-domain refinement or branch locking is the
next issue.

---

# 14. Do not do these yet

Do not add:

```text
wind
stream feeding
tidal torque
high-Mdot continuation
time-dependent limit cycle
synthetic emission
```

until:

```text
Nreg64 -> Nreg128 sonic-Taylor refinement works at fixed Mdot/Edd ~= 0.9028.
```

---

# 15. Compact Codex prompt

```text
Replace the dynamic first-order sonic patch with a sonic-regular Taylor patch.

Current dynamic patch infers g_s=(y_buffer-y_s)/Delta_s and enforces
A_s g_s+c_s=0. It prevents basin jumps but fails under Nreg refinement:
Nreg64 physical~1.35e-5, Nreg80~4.8e-5, Nreg96~1.5e-4, Nreg128~5.6e-4,
dominant patch. Naive micro-domain collocation is worse.

Implement sonic regularity as follows.

Let F(x,y,g)=A(x,y)g+c(x,y), x=lnR, y=(lnu,lnT).
At the sonic point A_s is singular. Introduce independent unknown
g_s=(g_u,g_T). Enforce:
    D(A_s)=0
    A_s g_s + c_s = 0
    l^T B_s(g_s)=0
where l is the left null vector of A_s and
    B_s(g)=d/dε F(x_s+ε, y_s+εg, g)|_{ε=0}
computed by centered finite differences.

For taylor1:
    y_buffer - y_s - Delta_s*g_s = 0

For taylor2 add h_s=(h_u,h_T):
    y_buffer - y_s - Delta_s*g_s - 0.5 Delta_s^2 h_s = 0
    F(x_s+0.5Delta_s,
      y_s+0.5Delta_s*g_s+0.125Delta_s^2 h_s,
      g_s+0.5Delta_s*h_s) = 0

Keep C1,C2,K as diagnostics, not production residuals initially.

First implement a root scan g=g_p+a r for l^T B_s(g)=0 at the existing N65
source. Choose the root closest to the coarse first-interval slope. Then run
taylor1 and taylor2 at Nreg64, Delta_s=0.02, R_far=1e5, Nouter=54. Next run
Nreg64->80->96->112->128 and Delta_s scans 0.03,0.02,0.015,0.01.

Success means physical residual <= few e-6, D,C1,C2,K small as diagnostics,
and stable Rson/lambda0/int_adv under Nreg and Delta_s.
```

---

# 16. Bottom line

The next hard task is no longer outer matching. It is the local mathematics of
regular passage through a singular sonic point.

The most promising next implementation is:

```text
sonic derivative unknown + L'Hopital regularity + Taylor patch
```

followed, if needed, by:

```text
second-order Taylor curvature patch.
```

This should replace the finite-difference dynamic patch and provide a mesh-
robust inner refinement path.
