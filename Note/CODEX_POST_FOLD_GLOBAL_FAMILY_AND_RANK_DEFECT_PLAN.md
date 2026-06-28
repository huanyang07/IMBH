# Codex Next-Step Brief: After the Confirmed Projection Fold and Inaccessible Rank Defects

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
latest main branch
```

Start from:

```text
Note/GPT_REVIEW_PROMPT_POST_FOLD_RANK_DEFECT.md
outputs/tables/transonic_desingularized_fold_refinement.md
outputs/tables/transonic_lambda_family_fold_map.md
outputs/tables/transonic_B_rank_defect_search.md
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
tests/test_transonic_local.py
```

## Executive conclusion

The latest diagnostics have clarified the topology of the current no-wind,
algebraic-stress transonic branch.

At the reference eigenvalue,

```text
lambda_ref = 3.674721616
```

the exact incoming branch reaches a robust radial projection fold at

```text
R_fold = 6.23789926 rg.
```

At the fold:

```text
p_x ~= 0
dp_x/ds = -9.592e-4
smin/smax(B) = 1.904e-5
H/R = 5.122e-4
Omega/Omega_K = 0.999586
```

This is a genuine local maximum of \(R(s)\), not a physical blow-up.

The fold is **not** a branch-switching point:

```text
relative B-minor norm at the incoming fold ~= 4.63e-2
smin/smax(B) ~= 1.9e-5
```

Known true B-rank-defect points exist, but they are not accessible from the
incoming branch:

```text
fixed-lambda rank defect:
    R = 6.118209949 rg
    lambda0 = 3.674721616
    access distance = 5.73

free-lambda rank defect:
    R = 6.219062068 rg
    lambda0 = 3.675931361
    access distance = 1.84
```

The immediate implication is:

```text
The current exact fixed-lambda no-wind branch cannot be used as a monotonic
outward disk profile. The known branch-switching candidates are not on this
incoming branch.
```

Therefore the next step should **not** be another tail-weighted solve, another
Taylor patch, or a branch-switching BVP from the current fold.

The next step should be a controlled **global-family continuation** and, in
parallel, a **rank-defect fan exploration**. If both fail, then the current
no-wind algebraic-stress closure should be considered topologically blocked at
this fixed \(\dot M\), and the project should move to revised physics.

---

# 1. What has now been ruled out

## 1.1 Continuing the same branch at fixed \(\lambda_0\)

The branch turns around in \(R\). Continuing it as a curve in phase space is
possible, but it will not give a single-valued monotonic disk profile \(y(R)\)
beyond \(R\simeq6.238r_g\).

## 1.2 Branch switching exactly at the incoming fold

The incoming fold is accessible but not rank-defective. Since \(B=[c\ A]\) has
rank two there, its nullspace is one-dimensional. There is no second outgoing
tangent to switch onto.

## 1.3 Using the known rank-defect points directly as switches for this branch

Both the fixed-lambda and free-lambda rank-defect points are genuine rank
defects, but the current incoming exact branch does not reach them.

A branch-switching BVP is only justified if the incoming branch reaches the
rank-defect point or can be globally continued to it.

## 1.4 Interpreting local lambda sensitivity as a valid global branch

The local \(R=6\) lambda sensitivity shows:

```text
delta_lambda = +0.001 -> R_fold = 6.376928 rg
delta_lambda = +0.002 -> R_fold = 6.486018 rg
delta_lambda = +0.004 -> R_fold = 6.663279 rg
```

This is encouraging, but it is not a valid global family because the strict
fixed-buffer critical seed accepts only the reference lambda. A true global
continuation must allow the sonic state, buffer, and outer profile to move
self-consistently with \(\lambda_0\).

---

# 2. Recommended next mathematical strategy

Use two tracks.

```text
Track A: global eigenfamily continuation
Track B: rank-defect fan exploration
```

Track A asks whether a smooth no-wind branch exists nearby when the entire
global solution is allowed to move with \(\lambda_0\).

Track B asks whether the known rank-defect points seed a different neighboring
solution family, even though they are not accessible from the current branch.

Only after both tracks fail should the project move to revised physics.

---

# 3. Track A: global eigenfamily continuation

## 3.1 Why this is needed

The current lambda map is not enough because it perturbs \(\lambda_0\) while
holding too much of the sonic/buffer structure fixed. That rejects almost every
nonzero lambda offset.

The right problem is:

```text
For each lambda0, solve the first sonic critical state and global outer branch
self-consistently, then locate the radial fold of that global branch.
```

This means the buffer, sonic state, and outer profile must all move with
\(\lambda_0\).

---

## 3.2 Unknowns

Use a full two-domain or phase-space BVP.

Unknowns should include:

```text
first sonic point:
    x_s = log R_son
    y_s = (logu_s, logT_s)

global eigenvalue:
    lambda0

phase-space transition segment:
    z_i = (x_i, logu_i, logT_i)
    p_i = dz_i/ds

regular inner/tail segment:
    logu_i, logT_i
    or phase-space z_i,p_i if needed

outer domain:
    logu_i, logT_i

optional continuation parameter:
    theta_out = outer entropy / far-temperature offset
```

Why include `theta_out`?

At fixed \(\dot M\) and fixed outer boundary conditions, the transonic solution
is normally isolated. To continue in \(\lambda_0\), one extra global parameter
must move. A clean choice is an outer entropy or far thermal offset,
\(\theta_{\rm out}\), that is driven back to zero at the end.

---

## 3.3 Residuals

### First sonic point

Use:

```math
D_s=0,
```

```math
C_{s,\rm pivot}=0,
```

and audit:

```text
C1, C2, K, smin/smax(A), smin/smax(B).
```

### Phase-space segment near the fold

For each node:

```math
B(z_i;\lambda_0)p_i=0,
```

```math
\|p_i\|_M^2-1=0.
```

For intervals:

```math
z_{i+1}-z_i-\frac{\Delta s_i}{2}(p_i+p_{i+1})=0.
```

This allows the solution to reveal whether \(p_x\) remains positive, turns
around, or passes a branch point.

### Regular disk/outer segment

Use the existing two-domain pressure-supported outer extension, which is
already performing well.

### Far boundary

Use the pressure-supported far closure, but include the optional outer thermal
homotopy:

```math
B_{\rm far,T} =
B_{\rm far,T}^{(0)} - \theta_{\rm out}.
```

The final physical solution must have:

```text
theta_out -> 0.
```

---

## 3.4 Continuation variables

Use pseudo-arclength continuation in the augmented variables:

```text
w = (all BVP unknowns, lambda0, theta_out)
```

Do not force monotonic \(\lambda_0\) or monotonic \(R\).

Use continuation in stages:

```text
Stage A1: theta_out free, lambda0 near reference
Stage A2: continue lambda0 while allowing theta_out
Stage A3: search for theta_out=0 crossings
Stage A4: for any theta_out=0 branch, locate R_fold
```

---

## 3.5 Diagnostics

Create:

```text
outputs/tables/transonic_global_family_continuation.md
outputs/figures/transonic_global_family_continuation.png
```

Columns:

```text
lambda0
theta_out
R_son
R_fold
max_R_reached
p_x_min
p_x sign changes
smin/smax(B)_min
H/R_max
Omega/OmegaK range
int_adv
far_residual
theta_out
status
```

A branch is viable only if:

```text
theta_out = 0
p_x > 0 until R_match or outer-domain connection
physical residual <= few x 1e-6
mesh convergence passes
```

---

# 4. Track B: rank-defect fan exploration

## 4.1 Why this is still worth doing

The known B-rank-defect points are not accessible from the current incoming
branch, but they may seed neighboring solution families. The free-lambda point
is closer than the fixed-lambda point:

```text
free-lambda access distance = 1.84
fixed-lambda access distance = 5.73
```

The free-lambda rank defect should be explored as a possible neighboring
global branch.

---

## 4.2 Local fan at a rank-defect point

At a rank-defect point, \(B\) has rank one and the nullspace of \(B\) is
two-dimensional.

Let:

```math
p(\alpha)=\cos\alpha\,p_1+\sin\alpha\,p_2.
```

Scan:

```text
alpha = 0 ... 2pi
```

and keep outgoing directions satisfying:

```text
p_x > 0
finite H/R
Qvisc positive
Omega/OmegaK reasonable
```

For each outgoing direction, integrate/desingularize for a small arclength and
check whether the branch moves outward.

Create:

```text
outputs/tables/transonic_rank_defect_fan_scan.md
outputs/figures/transonic_rank_defect_fan_scan.png
```

---

## 4.3 Rank-defect branch BVP

For promising directions, build a local branch BVP:

```text
rank-defect point -> R = 7, 8, 10 rg
```

using phase-space collocation:

```math
B_i p_i=0,
```

```math
\|p_i\|=1,
```

```math
z_{i+1}-z_i-\frac{\Delta s_i}{2}(p_i+p_{i+1})=0.
```

If a branch reaches outward, attach it to the outer-domain solver.

This branch is not guaranteed to be connected to the original incoming branch.
Treat it as a neighboring family.

---

# 5. Decision rule after Tracks A and B

## If Track A finds a theta_out=0 branch with no fold before R_match

Then the smooth no-wind solution exists. Resume grid validation and eventually
high-\(\dot M\) continuation.

## If Track B finds an outgoing branch from a rank-defect point that reaches the outer disk

Then build a branch-switching or neighboring-family global BVP and test whether
it satisfies the physical outer boundary.

## If neither Track A nor Track B works

Then stop pursuing a smooth no-wind algebraic-stress branch at this
\(\dot M\). The current closure is topologically blocked.

At that point, move to revised physics:

```text
1. wind-regulated transonic equations;
2. modified stress closure;
3. finite-thickness/vertical correction;
4. explicit internal shock/source/jump conditions;
5. time-dependent high-state rather than stationary hot branch.
```

---

# 6. Preferred physics extension if the no-wind branch is blocked

The next physics extension should be wind, not an arbitrary jump.

Reason:

```text
The target QPE burst rate is super-Eddington, and a no-wind smooth stationary
branch is already showing topological obstruction near ~0.9 Edd.
```

The minimal wind extension should modify:

```math
\frac{d\dot M}{dR}=2\pi R\dot\Sigma_w,
```

and the energy equation:

```math
Q_{\rm visc}=Q_{\rm rad}+Q_{\rm adv}+Q_{\rm wind}.
```

Use the existing energy-limited wind closure first. Then redo the transonic
critical/fold audit with wind included.

The test is:

```text
Does wind move/remove the projection fold?
```

---

# 7. Concrete scripts to add

## Track A scripts

```text
scripts/run_transonic_global_family_continuation.py
scripts/run_transonic_global_family_theta_crossing.py
scripts/run_transonic_global_family_fold_map.py
```

Outputs:

```text
outputs/tables/transonic_global_family_continuation.md
outputs/tables/transonic_global_family_theta_crossing.md
outputs/tables/transonic_global_family_fold_map.md
```

## Track B scripts

```text
scripts/run_transonic_rank_defect_fan_scan.py
scripts/run_transonic_rank_defect_branch_bvp.py
scripts/run_transonic_rank_defect_outer_attach.py
```

Outputs:

```text
outputs/tables/transonic_rank_defect_fan_scan.md
outputs/tables/transonic_rank_defect_branch_bvp.md
outputs/tables/transonic_rank_defect_outer_attach.md
```

## Physics-extension script if needed

```text
scripts/run_transonic_wind_fold_audit.py
```

Output:

```text
outputs/tables/transonic_wind_fold_audit.md
```

---

# 8. Concrete implementation notes

## 8.1 Add outer thermal homotopy

In the far-boundary residual, introduce:

```python
theta_out
```

as a global unknown:

```python
B_far_T = B_far_T_physical - theta_out
```

Use a prior/homotopy row when needed:

```python
theta_weight * theta_out = 0
```

for final physical solves.

For continuation, allow \(\theta_{\rm out}\) to float and search for zero
crossings.

## 8.2 Add phase-space BVP residual

Reuse the existing `B=[c A]` helpers.

For each node:

```python
node_residual = B_i @ p_i
norm_residual = dot(p_i, M @ p_i) - 1
```

For each interval:

```python
defect = z_right - z_left - 0.5 * ds_i * (p_left + p_right)
```

Use sparse block Jacobian if possible, but dense least-squares is fine for the
first small fan scans.

## 8.3 Add branch fan diagnostics

At rank defect:

```python
U, S, Vt = svd(B)
null_basis = Vt[-2:, :].T
```

Then scan:

```python
p = cos(alpha) * null_basis[:,0] + sin(alpha) * null_basis[:,1]
```

Normalize and orient. Record:

```text
p_x
physical diagnostics
short-flow success
```

---

# 9. Acceptance criteria

## Global-family continuation success

```text
theta_out crosses zero
R_fold > R_match or no fold before R_match
p_x remains positive
physical residual <= few x 1e-6
mesh validation passes
```

## Rank-defect branch success

```text
rank(B)<2 point confirmed
outgoing p_x>0 branch exists
branch reaches R >= 8--10 rg in local BVP
outer attach residual <= few x 1e-5 initially
```

## No-wind closure failure

Declare only if:

```text
global-family continuation cannot reach theta_out=0 without folding,
rank-defect fan has no usable outgoing branch,
and results are robust under metric/grid/continuation checks.
```

Then move to wind/stress physics.

---

# 10. Compact Codex prompt

```text
The post-fold diagnostics show a robust radial projection fold at
R=6.237899 rg for the exact fixed-lambda branch. B is not rank-defective at
the fold, so no branch switch is available there. Lambda sensitivity shows
R_fold moves outward with larger lambda locally, but strict fixed-buffer
critical seeding only accepts the reference lambda; this is not a valid global
family. True B-rank-defect points exist at R=6.118 fixed lambda and R=6.219
free lambda, but they are not accessible from the incoming branch.

Next implement two tracks:

Track A: global eigenfamily continuation.
Allow the sonic state, buffer, outer profile, lambda0, and an outer thermal
homotopy theta_out to move together. Continue in pseudo-arclength over
(all BVP unknowns, lambda0, theta_out). Search for theta_out=0 branches whose
phase-space curve does not fold before R_match.

Track B: rank-defect fan exploration.
At the free-lambda rank defect, compute the two-dimensional nullspace of B and
scan outgoing tangents p(alpha). Keep branches with p_x>0 and valid physical
diagnostics. Build small phase-space BVPs from the rank defect outward and see
whether any branch reaches R>=8--10 rg and can attach to the outer domain.

Do not build a branch-switching BVP from the incoming fold. It is not rank
defective. Do not force a free tail with derivative discontinuity. If both
tracks fail robustly, stop the smooth no-wind algebraic-stress branch search
and add wind or change stress/vertical closure.
```

---

# 11. Bottom line

The clean mathematical conclusion is:

```text
The current exact no-wind branch is blocked by a radial projection fold, and
the known rank-defect branch points are not accessible from it.
```

The clean next step is:

```text
global-family continuation + rank-defect fan exploration.
```

If those fail, the next step should be revised physics, especially a
wind-regulated transonic branch.
