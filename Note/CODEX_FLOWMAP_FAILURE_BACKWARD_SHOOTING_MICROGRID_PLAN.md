# Codex Next-Step Brief: Flow-Map Sonic Patch Failed Because the Post-Buffer Microgrid Is Too Coarse and the Old Dynamic Branch Is Probably Not the Regular Branch

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
latest main branch
```

Primary files reviewed:

```text
Note/CODEX_OFFSET_SONIC_FLOWMAP_NEXT_STRATEGY.md
outputs/tables/transonic_two_domain_sonic_flowmap_grid_scan.md
outputs/tables/transonic_two_domain_sonic_flowmap_microdomain_scan.md
scripts/run_transonic_two_domain_sonic_flowmap.py
Note/GPT_TAYLOR_LHOPITAL_NEXT_STEP.md
outputs/tables/transonic_sonic_flowmap_branch_audit.md
outputs/tables/transonic_sonic_flowmap_fit.md
```

## Executive diagnosis

The offset sonic flow-map implementation has not succeeded yet, but the failure is diagnostic.

Current results show:

```text
1. Both L'Hopital derivative branches integrate successfully from the sonic point
   to epsilon_buf=0.02.

2. Both branches land far from the previous dynamic-patch buffer state:
       distance ~ 2.88 in log-space,
       dlogu ~ -0.79,
       dlogT ~ +2.77.

3. Local sonic-to-buffer fitting cannot move R_son, y_s, and lambda0 enough to
   match the old dynamic buffer:
       residual ~ 2.75--2.77, dominated by the flow-map mismatch.

4. Global flow-map BVP attempts remain at physical residual ~0.1--0.17,
   dominated by regular_R.

5. Micro-domain attempts also fail at ~0.12--0.15, dominated by micro_R,
   especially the first post-buffer radial interval.
```

This means the current failure is **not simply that the flow-map code is broken**.
It is more specific:

```text
The exact L'Hopital/flow-map branches are not close to the old dynamic-patch
branch at epsilon_buf=0.02, and the first post-buffer interval is much too
coarse for the rapidly curving local solution.
```

The old dynamic first-order patch should now be treated as a useful numerical
regularizer, not as a confirmed sonic-regular solution.

The next numerical strategy should be:

```text
1. Stop matching the exact flow-map branch to the old dynamic-patch buffer.
2. Determine whether the old branch has any nearby true sonic critical point
   by backward shooting.
3. If not, build the global branch outward from the flow-map-consistent branch,
   using a very small epsilon_buf and a strongly graded post-buffer microgrid.
```

Do not resume high-Mdot continuation yet.

---

# 1. What the latest flow-map results mean

## 1.1 Branch audit: the exact sonic branches are far from the old dynamic branch

The branch audit gives, for `epsilon0=1e-5` and `epsilon_buf=0.02`:

```text
branch 0:
    g ~ (-296, 121)
    distance to target ~ 2.88
    dlogu ~ -0.787
    dlogT ~ +2.77

branch 1:
    g ~ (220, -122)
    distance to target ~ 2.88
    dlogu ~ -0.788
    dlogT ~ +2.77
```

Both integrations succeed and keep `H/R` modest, so the flow map is not simply
crashing. It lands on a different local branch from the old dynamic-patch
solution.

## 1.2 Local fit: adjusting the sonic state does not rescue the old target

The local sonic-to-buffer fit allows changes in:

```text
R_son
logu_s
logT_s
lambda0
```

but still returns:

```text
physical residual ~ 2.75--2.77
dominant = flow
```

This tells us the old dynamic buffer state is not reachable from the currently
identified regular sonic branches by small adjustments of the sonic state.

## 1.3 Grid scan: the global BVP seeded from the old branch remains far away

The smooth-a flow-map grid scan gives residuals:

```text
N16 best tested ~ 0.10
N32 ~ 0.14--0.17
dominant = regular_R
```

The branch coordinate `a` settles near:

```text
a ~ -300
```

so the solve is attempting to use a far L'Hopital derivative branch, but cannot
make the rest of the global solution compatible.

## 1.4 Micro-domain scan: the first post-buffer interval is the immediate numerical bottleneck

The micro-domain scan shows:

```text
N16, R_micro=30, N_micro=8:
    physical ~ 0.120
    dominant = micro_R

N32, R_micro=30, N_micro=16:
    physical ~ 0.149
    dominant = micro_R
```

The residual localization is especially important:

```text
first micro interval:
    R ~ 5.54--6.16 rg or 5.57--6.87 rg
    radial residual ~ -0.12 to -0.15
```

This is a huge interval immediately after the sonic buffer. A rapidly curving
flow-map branch cannot be represented accurately by one midpoint interval of
this size.

## 1.5 Exact flow-map seeding every micro-domain node also fails

The diagnostic mode that fills all micro-domain nodes from the local flow map
produces enormous mismatch downstream:

```text
R_micro=10 rg: seed physical residual ~ 21
R_micro=30 rg: seed physical residual ~ 44
```

This says the exact local sonic branch diverges rapidly from the old global
basin once integrated outward.

That is physical/numerical information. It means the previous dynamic branch is
probably not the correct target for exact sonic regularity.

---

# 2. Immediate conclusion

The current question is no longer:

```text
Can we make the exact flow map match the old dynamic-patch buffer?
```

The answer appears to be no.

The correct question is now:

```text
Does the global disk have any sonic-regular branch compatible with the outer
two-domain solution, and if yes, can we construct it without using the old
dynamic patch as the target?
```

---

# 3. Next Strategy A: Backward shooting audit from the old dynamic branch

Before building a new full BVP, determine whether the old dynamic branch has a
nearby true sonic critical point.

## 3.1 Method

Take a reliable old dynamic/two-domain solution and choose a buffer point:

```text
x_b = x_s + Delta_s
y_b = (logu_b, logT_b)
lambda0 = old lambda0
```

Integrate the regular ODE backward:

```math
\frac{dy}{dx}=-A^{-1}c
```

from \(x_b\) toward smaller \(x\).

Monitor:

```text
D
C1
C2
K
smin/smax
condition(A)
H/R
Qvisc sign
Qadv/Qvisc
```

The goal is to see whether the backward trajectory reaches a point where:

```text
D -> 0
C1 -> 0
C2 -> 0
smin/smax -> 0
```

near the old \(R_{\rm son}\).

## 3.2 Parameter scan

Repeat the backward shooting while varying:

```text
lambda0 by +/- 1e-3, +/- 3e-3
buffer logu by +/- 1e-3, +/- 3e-3
buffer logT by +/- 1e-3, +/- 3e-3
```

This is a local sensitivity test.

## 3.3 Possible outcomes

### Outcome A: a critical point is found

Then the old branch can be regularized. Use that critical point and backward
trajectory to seed the new BVP.

### Outcome B: no critical point is found

Then the old dynamic-patch branch is not sonic-regular under the current
equations. Stop trying to force the exact flow map to match it.

## 3.4 Script and output

Create:

```text
scripts/run_transonic_sonic_backward_shooting_audit.py
outputs/tables/transonic_sonic_backward_shooting_audit.md
outputs/figures/transonic_sonic_backward_shooting_D_C.png
```

---

# 4. Next Strategy B: Flow-map-consistent global branch, not old-basin matching

If backward shooting fails, build the branch outward from the exact sonic flow
map.

Do not seed only the buffer node from the flow map while leaving the rest of
the disk on the old dynamic branch. That creates a discontinuity in the branch
geometry and produces the large regular_R/micro_R residuals already seen.

## 4.1 Construct a fully flow-map-consistent inner seed

For each derivative branch:

1. Start at:
   ```text
   x_s + epsilon0
   ```
2. Integrate the regular ODE outward to:
   ```text
   R_match = 6500 rg
   ```
3. Sample this ODE solution on the inner grid.
4. Use that as the entire inner-domain seed, not just the buffer value.

Then create the outer-domain seed from the inner endpoint, not from the old
dynamic solution.

This is a different basin from the old dynamic branch, so it must be solved as
a new branch, not a perturbation of the old one.

## 4.2 Add homotopy to far-boundary conditions

The flow-map-consistent inner branch may not immediately satisfy the
pressure-supported far boundary. Use a homotopy:

```math
F_\eta
=
(1-\eta)F_{\rm easy}
+
\eta F_{\rm full}
```

where:

```text
eta = 0, 0.1, 0.2, ..., 1.
```

Suggested `easy` system:

```text
inner flow-map branch fixed or weakly regularized;
outer domain loosely matched to inner endpoint;
far boundary weakly weighted.
```

Then gradually strengthen:

```text
interface continuity
outer-domain collocation
pressure-supported far boundary
```

This avoids asking the optimizer to jump directly from the old branch to a new
flow-map branch.

## 4.3 Run both derivative branches

Do not assume branch 0 only.

Run:

```text
branch 0
branch 1
```

and compare:

```text
global residual
physical diagnostics
H/R
Qvisc sign
entropy behavior
Omega/OmegaK
lambda0
```

The physical branch is the one that yields a smooth global disk, not the one
closest to the old dynamic slope.

---

# 5. Next Strategy C: Use much smaller epsilon_buf and a graded post-buffer grid

The current `epsilon_buf=0.02` is too large for a derivative of order
\(|g|\sim200--300\).

Use:

```text
epsilon_buf = 0.001, 0.002, 0.005, 0.01
```

with:

```text
epsilon0 = 1e-6 or 3e-6
```

Then start ordinary collocation at:

```text
x_s + epsilon_buf.
```

## 5.1 Strongly graded microgrid

The current micro-domain intervals are much too large. For example the first
interval can span:

```text
R ~ 5.54--6.16 rg
```

or even:

```text
R ~ 5.57--6.87 rg.
```

That is too broad for the steep local branch.

Use a microgrid in \(x=\ln R\) with:

```text
first dx <= 1e-3--3e-3
growth ratio <= 1.15--1.25
```

until:

```text
x - x_s ~ 0.05--0.1.
```

Then connect to the normal inner grid.

## 5.2 Suggested grid construction

Define:

```text
x0 = x_s + epsilon_buf
x1 = x_s + 0.05
x2 = log(R_micro), maybe R_micro = 20--30 rg
```

Use:

```text
micro-near: geometric spacing from x0 to x1
micro-far: moderate spacing from x1 to x2
regular: rest of inner domain to R_match
```

Example:

```text
epsilon_buf = 0.002
x1 - x_s = 0.05
first dx = 0.001
growth ratio = 1.2
```

The old microdomain with only 8--16 nodes over \(R\sim5.5\) to \(30r_g\) is
not sufficiently resolved near the buffer.

## 5.3 Use integrated defects in the microdomain

For the microdomain, use the integrated defect form:

```math
A_i (y_{i+1}-y_i) + \Delta x_i c_i = 0
```

as the solve residual.

Still report the differential residual as the physical audit.

This avoids excessive \(1/\Delta x\)-sensitivity in the Jacobian while solving
on tiny intervals.

---

# 6. Next Strategy D: Reverse-flowmap BVP as a local shooting fit

The previous forward local fit tried:

```text
sonic -> buffer target
```

and failed.

Now solve the inverse problem:

```text
buffer -> sonic
```

because the global/outer branch gives the buffer state more reliably than it
gives the true derivative.

## 6.1 Unknowns

Use:

```text
lambda0
x_s
small corrections to y_buffer: delta logu_b, delta logT_b
```

or first hold \(y_b\) fixed.

## 6.2 Residuals

Integrate backward from \(x_b\) to \(x_s+\epsilon0\). Use one tiny Taylor step
to infer \(y_s\), then evaluate:

```text
D(x_s,y_s,lambda0)
C1(x_s,y_s,lambda0)
C2(x_s,y_s,lambda0)
```

If allowing buffer corrections, add priors:

```text
delta logu_b / sigma_u
delta logT_b / sigma_T
```

with:

```text
sigma_u, sigma_T ~ 1e-3--1e-2.
```

This finds whether a nearby sonic critical point exists for the old global
branch.

## 6.3 Why reverse shooting is useful

Forward flowmap asks:

```text
Does a chosen local critical point land on the old global branch?
```

Backward shooting asks:

```text
Does the old global branch actually approach any regular critical point?
```

The second question is the right diagnostic now.

---

# 7. Interpret possible outcomes

## 7.1 Backward shooting finds a nearby critical point

Then use that point to initialize the offset BVP and resume regular refinement.

This would mean the old dynamic patch was close to a true branch, but the
forward L'Hopital derivative branch selection was wrong or too sensitive.

## 7.2 Backward shooting finds no nearby critical point

Then the old dynamic branch is a numerical regularized branch, not the true
transonic branch.

Do not use it as the target. Build the flow-map-consistent branch from scratch.

## 7.3 Flow-map-consistent branch cannot satisfy outer boundary

Then the current algebraic stress/vertical closure may not support a
sonic-regular no-wind solution at this \(\dot M\).

Before changing physics, verify:

```text
both derivative branches tested
epsilon_buf -> 0 trend checked
microgrid refined
outer homotopy attempted
```

Only after that should we revisit the stress closure or add wind.

---

# 8. Suggested experiment sequence

## Experiment 1: backward shooting audit

Run from the old dynamic/two-domain buffer state.

Output:

```text
D,C1,C2,K versus x
smin/smax versus x
condition(A)
H/R
Qvisc sign
```

for several \(\lambda_0\) perturbations.

## Experiment 2: small-epsilon flowmap branch audit

Run:

```text
epsilon_buf = 0.001, 0.002, 0.005, 0.01, 0.02
branch = 0, 1
```

Output:

```text
distance to dynamic buffer
distance to flowmap-consistent inner seed
H/R max
Qvisc sign
entropy diagnostics
```

## Experiment 3: graded microgrid test

Use:

```text
epsilon_buf = 0.002 or 0.005
first dx <= 0.001--0.003
growth ratio <= 1.2
```

Run branch 0 and branch 1.

## Experiment 4: fully flowmap-consistent seed

Fill the entire inner domain from ODE integration, not just the first buffer
point.

Then solve with outer/far boundary homotopy:

```text
eta = 0 -> 1
```

## Experiment 5: reverse-flowmap local BVP

If Experiment 1 suggests a nearby critical point, solve for it directly.

---

# 9. Concrete code changes

## 9.1 `transonic_local.py`

Add or harden:

```python
def ode_rhs(logR, y, lambda0, params):
    A, c, diag = scaled_differential_matrix(logR, y, lambda0, params)
    if diag.smin_over_smax < threshold:
        raise NearCriticalPointError
    return np.linalg.solve(A, -c)
```

Add:

```python
def integrate_from_sonic_offset(logR_s, y_s, lambda0, branch, eps0, eps1, params):
    ...
```

Add:

```python
def backward_shoot_to_sonic(logR_b, y_b, lambda0, logR_trial, params):
    ...
```

## 9.2 `transonic_collocation.py`

Add a new patch mode:

```text
sonic_patch_mode = "offset_flowmap_microgrid"
```

with:

```text
exact sonic algebraic constraints
flowmap boundary at epsilon_buf
graded microdomain from epsilon_buf outward
integrated defects in microdomain
ordinary inner collocation beyond microdomain
```

Add a parameter block:

```python
epsilon0: float = 1e-6
epsilon_buf: float = 0.002
micro_first_dx: float = 1e-3
micro_growth: float = 1.2
micro_x_extent: float = 0.05
```

## 9.3 Scripts

Create:

```text
scripts/run_transonic_sonic_backward_shooting.py
scripts/run_transonic_sonic_small_epsilon_flowmap.py
scripts/run_transonic_sonic_graded_microgrid.py
scripts/run_transonic_flowmap_consistent_branch.py
scripts/run_transonic_sonic_reverse_fit.py
```

## 9.4 Outputs

Create:

```text
outputs/tables/transonic_sonic_backward_shooting.md
outputs/tables/transonic_sonic_small_epsilon_flowmap.md
outputs/tables/transonic_sonic_graded_microgrid.md
outputs/tables/transonic_flowmap_consistent_branch.md
outputs/tables/transonic_sonic_reverse_fit.md
```

---

# 10. Acceptance criteria

## Local sonic audit

Pass if backward shooting finds a nearby critical point or clearly shows none
exists.

## Small-epsilon flowmap

Pass if branch output converges as:

```text
epsilon_buf -> 0
```

and no branch becomes pathological.

## Graded microgrid

Pass if first-interval residual drops from:

```text
~0.1
```

to:

```text
<= few x 1e-5
```

before global outer coupling.

## Full global root

Pass if:

```text
physical residual <= few x 1e-6
D,C1,C2,K all small
Rson,lambda0,int_adv stable under Nreg and epsilon_buf
no hidden regular_R or micro_R floor
```

## Science milestone

Do not proceed to high \(\dot M\) until:

```text
Nreg64,80,96,112,128
```

or an equivalent nested sequence is stable at fixed:

```text
Mdot/Mdot_Edd ~= 0.90277664.
```

---

# 11. Do not do next

Do not spend the next sprint on:

```text
more Taylor patch variants over Delta_s=0.02
more C1/C2/K soft-weight scans
more high-Mdot continuation
wind
stream/tide physics
time-dependent light curves
```

The immediate problem is local sonic branch compatibility and microgrid
resolution.

---

# 12. Compact Codex prompt

```text
The offset sonic flowmap test shows both L'Hopital branches integrate
successfully to epsilon_buf=0.02 but land far from the old dynamic buffer:
distance~2.88, dlogu~-0.79, dlogT~+2.77. Local sonic-to-buffer fitting also
fails with residual~2.75, dominated by flow. Global flowmap BVPs remain at
0.1--0.17 dominated by regular_R. Microdomain tests remain at 0.12--0.15
dominated by micro_R, with the first post-buffer interval carrying the largest
radial residual.

This suggests the exact sonic branch is not close to the old dynamic branch
and the post-buffer microgrid is far too coarse.

Next implement:
1. Backward shooting from the old dynamic buffer toward the sonic point to see
   whether the old branch actually approaches D=C1=C2=0.
2. Small-epsilon flowmap scans with epsilon_buf=0.001,0.002,0.005,0.01 and
   both branches.
3. A graded microgrid beginning at epsilon_buf with first dx<=1e-3--3e-3 and
   growth ratio<=1.2. Use integrated defects in this microgrid.
4. A fully flowmap-consistent inner seed: fill the whole inner domain by
   integrating the exact branch outward, then use homotopy to attach the outer
   domain. Do not only replace the first buffer point while keeping the old
   dynamic branch elsewhere.
5. Reverse-flowmap local fit: from a buffer state integrate backward and solve
   for x_s, lambda0, and small buffer corrections such that D,C1,C2 vanish at
   the sonic point.

Do not tune more long Taylor patches or resume high-Mdot continuation yet.
```

---

# 13. Bottom line

The current results imply:

```text
The old dynamic patch likely belongs to a different numerical basin than the
exact L'Hopital flow-map branches.
```

The next question is not how to force the exact branch to match the old buffer,
but whether:

```text
(a) the old buffer has any true regular sonic point when integrated backward,
or
(b) a different flowmap-consistent global branch exists when seeded and meshed
    properly.
```
