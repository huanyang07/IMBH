# Codex Next-Step Brief: Flow-Map Branch Is Regular Locally, but Tail Matching Fails

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
latest main branch
```

Primary latest outputs reviewed:

```text
outputs/tables/transonic_sonic_backward_shooting_audit.md
outputs/tables/transonic_sonic_flowmap_branch_audit.md
outputs/tables/transonic_sonic_flowmap_fit.md
outputs/tables/transonic_reverse_seed_flowmap_bvp.md
outputs/tables/transonic_reverse_seed_graded_flowmap_bvp.md
outputs/tables/transonic_reverse_seed_transition_shell_seed_scan.md
outputs/tables/transonic_reverse_seed_transition_shell_solve_short_eps0p001_N96.md
outputs/tables/transonic_reverse_seed_transition_shell_solve_short_eps0p002_N96.md
outputs/tables/transonic_reverse_seed_transition_shell_solve_short_eps0p002_N128.md
outputs/tables/transonic_tail_weighted_continuation_eps0p001_N96.md
outputs/tables/transonic_tail_weighted_continuation_eps0p002_N96.md
scripts/run_transonic_reverse_seed_graded_flowmap_bvp.py
Note/CODEX_OFFSET_SONIC_FLOWMAP_NEXT_STRATEGY.md
```

## Executive diagnosis

The latest runs clarify the situation:

```text
1. The old dynamic-patch branch is not obviously nonsense: backward shooting
   finds a nearby critical point if lambda0 is shifted downward by about
   0.0015.

2. The exact L'Hopital/flow-map branch can be made extremely regular near the
   sonic point.

3. The current failure is no longer the flow map or the microgrid. It is the
   connection from the exact sonic branch to the old outer/tail branch.

4. The dominant residual in the best runs is a tail/post-buffer residual around
   R ~ 7--9 rg, not the sonic residual.

5. Increasing N does not cure the problem; N128 is worse than N96. This means
   the issue is branch matching, not simply resolution.
```

The next strategy should be a **multiple-shooting tail-matching problem**, not
another sonic-patch variant.

Do not resume high-\(\dot M\) continuation yet.

---

# 1. What the latest results show

## 1.1 Backward shooting found a nearby critical point

The backward-shooting audit starts from the old dynamic-patch buffer and
integrates inward. The base case does not reach a clean critical point, but
a focused lambda shift of about:

```text
dlambda ~= -0.00153
```

finds a near-critical state with:

```text
best D/C/K ~ 7e-7
R ~ 5.885 rg
x - x_old_sonic ~ 0.019
```

This is important.

It means the old dynamic branch likely has a nearby true critical point, but
that point is not the old nominal \(R_{\rm son}\simeq5.77r_g\). It sits close
to the old buffer region, near \(R\simeq5.885r_g\).

Interpretation:

```text
The dynamic patch probably displaced the true critical point outward.
```

The correct sonic anchor for exact-flowmap work should therefore be the
backward-shooting critical point, not the old dynamic \(R_{\rm son}\).

---

## 1.2 Forward flow maps from the old sonic state did not match the old buffer

The earlier flow-map audit showed that both exact L'Hopital branches integrate
successfully, but land far from the old dynamic-patch buffer:

```text
distance to target ~ 2.88
dlogu ~ -0.79
dlogT ~ +2.77
```

The local sonic-to-buffer fit could not fix this; it remained dominated by the
flow mismatch.

Interpretation:

```text
The old dynamic buffer is not reachable from the old sonic state by a true
regular L'Hopital flow map.
```

This is consistent with the backward-shooting result: the true critical point
is probably nearer the buffer.

---

## 1.3 Reverse-seeded exact branch improves the near-sonic region

After seeding from the reverse/backward critical point, the best transition-shell runs have excellent near-sonic residuals.

For example, in the `eps0p002_N96` transition-shell run:

```text
near_sonic ~ 2e-6
flow      ~ 1e-6
micro_R   ~ 3e-7
micro_E   ~ 3e-8
trans_R   ~ 3e-5
trans_E   ~ 4e-6
```

So the exact branch is locally regular.

The remaining physical residual is:

```text
physical ~ 0.089
dominant = tail_R
worst R ~ 7.04 rg
```

Interpretation:

```text
The sonic/micro/transition region is no longer the main problem.
The tail is not compatible with the exact sonic branch.
```

---

## 1.4 N128 worsens the same problem

The corresponding N128 run gives:

```text
physical ~ 0.47
dominant = tail_E
worst R ~ 6.33 rg
```

while retaining small sonic/micro residuals.

Interpretation:

```text
The tail mismatch is not disappearing with resolution.
Refinement is resolving the incompatibility more clearly.
```

---

## 1.5 Weighted tail continuation does not solve it

The tail-weighted continuation runs with eps0p001 and eps0p002 remain stuck near:

```text
physical ~ 0.09--0.11
dominant = tail_R
```

even when the optimizer terminates successfully.

Interpretation:

```text
The problem is not simply insufficient nfev or tail weighting.
The branch connection itself is wrong.
```

---

# 2. Revised bottleneck

The current bottleneck is not:

```text
outer boundary
sonic compatibility
micro-domain resolution alone
L'Hopital branch root finding
```

The current bottleneck is:

```text
matching the exact sonic branch to a global outer/tail solution.
```

In other words:

```text
The exact sonic branch and the old dynamic-patch outer branch appear to live in
different numerical basins.
```

The right next question is:

```text
Does there exist a global transonic branch that connects the exact
backward-shooting critical point to the pressure-supported far boundary?
```

This should be tested as a multiple-shooting problem.

---

# 3. Do not force the exact branch onto the old tail

The old dynamic branch should now be used only as an initial guess.

Do not impose strong penalties trying to keep the solution close to the old
tail. That appears to be what produces tail residuals around \(R\sim7\)--\(9r_g\).

Instead:

```text
free the tail and solve the global branch from the exact sonic anchor outward.
```

The outer/far boundary can remain fixed because the two-domain outer extension
was previously validated.

---

# 4. Next Strategy: multiple-shooting junction solve

## 4.1 Split the problem into segments

Use three segments:

```text
Segment A: exact sonic flow-map segment
    R_crit -> R_join

Segment B: free transition/tail segment
    R_join -> R_match

Segment C: outer domain
    R_match -> R_far
```

Recommended initial choices:

```text
R_crit  ~ 5.885 rg
R_join  = 7, 8, 10, 12 rg
R_match = 6500 rg
R_far   = 1e5 rg
```

Segment A should be seeded by the exact L'Hopital flow map from the
backward-shooting critical point.

Segment B should be free. It may use the old dynamic tail as an initial guess,
but it should not be strongly regularized to it.

Segment C can remain the pressure-supported two-domain outer solution.

---

## 4.2 Unknowns

For the first multiple-shooting implementation, use:

```text
critical point:
    logR_crit
    logu_crit
    logT_crit
    lambda0

Segment A:
    branch index fixed initially
    optional branch parameter a if needed
    flow-map/microgrid nodes

Segment B:
    logu/logT nodes from R_join to R_match

Segment C:
    existing outer-domain nodes
```

At first keep branch index fixed to the branch found by backward shooting.

---

## 4.3 Residuals

### Critical point residuals

Use:

```text
D = 0
C_pivot = 0
```

and audit:

```text
C1, C2, K, smin/smax.
```

### Segment A residuals

Use either:

```text
flow-map boundary from Rcrit to epsilon_buf
+ graded microgrid
```

or integrate directly to \(R_{\rm join}\) for seed generation.

### Junction residual at \(R_{\rm join}\)

Enforce continuity:

```text
logu_A(R_join) = logu_B(R_join)
logT_A(R_join) = logT_B(R_join)
```

Optionally also enforce derivative continuity as a diagnostic:

```text
g_A(R_join) - g_B(R_join)
```

but do not include derivative continuity as a hard residual initially.

### Segment B and C residuals

Use ordinary collocation:

```text
Abar_i @ (y_{i+1}-y_i) + dx_i*cbar_i = 0
```

Use integrated defects in the transition/tail region.

### Far boundary residuals

Keep the pressure-supported far boundary closure.

---

# 5. Scan the junction radius

The current worst residuals occur around:

```text
R ~ 6.3--9.3 rg
```

depending on the run.

Run:

```text
R_join = 6.5, 7, 8, 10, 12, 15 rg
```

For each \(R_{\rm join}\):

1. integrate exact branch from critical point to \(R_{\rm join}\);
2. build a free transition/tail segment from \(R_{\rm join}\) outward;
3. solve the BVP;
4. record residuals and branch diagnostics.

Create:

```text
outputs/tables/transonic_multiple_shooting_rjoin_scan.md
outputs/figures/transonic_multiple_shooting_rjoin_scan.png
```

A good \(R_{\rm join}\) should minimize:

```text
tail_R
tail_E
post_R
post_E
```

without spoiling:

```text
D,C1,C2,K
flow/micro residuals.
```

---

# 6. Use continuation in \(R_{\rm join}\)

Do not jump directly from \(R_{\rm join}=7\) to \(R_{\rm join}=15\).

After finding the best initial \(R_{\rm join}\), continue:

```text
R_join = 6.5 -> 7 -> 7.5 -> 8 -> 9 -> 10 -> 12 -> 15 -> 20 rg
```

Use the previous solution as seed.

This will reveal whether the exact sonic branch can be extended into the old
tail region smoothly.

---

# 7. Use a branch-connection residual before solving the full BVP

Before the full nonlinear solve, compute a low-cost diagnostic.

For a given critical point and branch:

1. integrate exact branch outward to \(R_{\rm join}\);
2. evaluate old outer/tail solution at \(R_{\rm join}\);
3. compute:

```text
Delta logu
Delta logT
Delta g_u
Delta g_T
Delta entropy
Delta Omega/OmegaK
```

as functions of \(R_{\rm join}\).

Create:

```text
outputs/tables/transonic_branch_connection_scan.md
```

If the value mismatch never becomes small for any \(R_{\rm join}\), the old tail
is not on the exact sonic branch.

This will prevent wasted global solves.

---

# 8. Treat \(lambda0\) as the main continuation parameter

The backward-shooting audit shows that changing \(\lambda_0\) by only:

```text
-0.0015
```

can move the system from non-regular to nearly critical.

Therefore the next global solve should not keep \(\lambda_0\) too tightly tied
to the old dynamic value.

Use \(\lambda_0\) continuation:

```text
lambda0 = old value
lambda0 - 5e-4
lambda0 - 1e-3
lambda0 - 1.5e-3
lambda0 - 2e-3
```

At each step, recompute the critical point and branch-connection scan.

The correct global branch may live at a slightly different \(\lambda_0\) than
the dynamic patch.

---

# 9. Reverse-shoot the outer branch inward

There is a complementary approach that may be even more informative.

From the pressure-supported outer solution, integrate or solve inward from
\(R_{\rm match}\) toward \(R\sim7r_g\), using the same local ODE.

Then compare with the exact sonic branch integrated outward.

This gives a two-sided shooting mismatch:

```text
M(R_join; lambda0, branch) =
y_sonic_out(R_join) - y_outer_in(R_join).
```

Solve:

```text
M = 0
```

by varying:

```text
lambda0
R_join
possibly one outer entropy/temperature parameter
```

This is the cleanest test of whether an exact global transonic branch exists.

Create:

```text
scripts/run_transonic_two_sided_shooting_match.py
outputs/tables/transonic_two_sided_shooting_match.md
```

---

# 10. Reweighting alone is not enough

The latest weighted continuation runs show that simply changing weights does
not remove the tail mismatch.

The issue is not:

```text
tail_weight too small
```

The issue is:

```text
tail segment is attached to the wrong branch or wrong junction point.
```

Therefore do not spend the next sprint on more `flow_weight`, `micro_weight`,
or `transition_weight` scans unless they are part of a multiple-shooting solve.

---

# 11. Concrete scripts to add

## 11.1 Branch connection scan

```text
scripts/run_transonic_branch_connection_scan.py
```

Inputs:

```text
critical point from backward shooting
branch id
old dynamic/two-domain profile
R_join list
```

Outputs:

```text
outputs/tables/transonic_branch_connection_scan.md
```

Columns:

```text
R_join
branch
lambda0
Delta logu
Delta logT
Delta g_u
Delta g_T
Delta entropy
Delta Omega/OmegaK
H/R_sonic
H/R_outer
Qadv/Qvisc_sonic
Qadv/Qvisc_outer
```

## 11.2 Multiple-shooting junction solve

```text
scripts/run_transonic_multiple_shooting_junction.py
```

Outputs:

```text
outputs/tables/transonic_multiple_shooting_junction.md
```

Stages:

```text
seed
release_segment_B
release_lambda0
full_polish
```

## 11.3 R_join continuation

```text
scripts/run_transonic_multiple_shooting_rjoin_continuation.py
```

Outputs:

```text
outputs/tables/transonic_multiple_shooting_rjoin_continuation.md
```

## 11.4 Two-sided shooting

```text
scripts/run_transonic_two_sided_shooting_match.py
```

Outputs:

```text
outputs/tables/transonic_two_sided_shooting_match.md
```

---

# 12. Success criteria

## Local critical point

Accept if:

```text
D,C1,C2,K <= 1e-6
smin/smax small
Qvisc positive
H/R reasonable
```

## Branch connection

A promising \(R_{\rm join}\) should have:

```text
|Delta logu| < 0.05
|Delta logT| < 0.05
```

before global solving.

If mismatches are order unity everywhere, the old tail is not part of the exact
branch.

## Multiple shooting

Accept if:

```text
physical residual <= few x 1e-5 first
then <= few x 1e-6 after polish
tail_R and tail_E no longer dominate
D,C1,C2,K remain small
Rson/lambda0/int_adv stable under modest changes of R_join and N
```

## Grid validation

After a successful junction solve:

```text
N96 -> N128
R_join shift +/- 10%
R_far fixed at 1e5 rg
```

should not move the solution qualitatively.

---

# 13. Interpreting outcomes

## Outcome A: multiple shooting succeeds

Then the exact sonic branch exists and the old failures were caused by forcing
the wrong tail attachment. Proceed to grid convergence and then high-rate
continuation.

## Outcome B: exact branch never matches the old outer branch

Then the old dynamic-patch branch is not sonic-regular under the current
equations. Build a new branch from exact sonic constraints outward, or revisit
the closure.

## Outcome C: exact branch exists but differs substantially in \(int\_adv\)

Then the previous dynamic solution was not physically diagnostic. Use the exact
branch values for future high-rate continuation.

## Outcome D: no exact branch can satisfy far boundary

Then, after checking both branches and \(\lambda_0\) continuation, revisit the
stress/vertical closure before adding wind.

---

# 14. Do not do next

Do not spend the next sprint on:

```text
more flow/micro/tail weight tuning
more long Taylor patches
more direct matching to old dynamic buffer
high-Mdot continuation
wind or stream/tide physics
```

The current blocker is branch connection between the exact sonic solution and
the global outer solution.

---

# 15. Compact Codex prompt

```text
Latest runs show the exact reverse-seeded flowmap branch is locally regular,
but it does not connect to the old dynamic tail. Backward shooting found a
nearby critical point if lambda0 is shifted by about -0.00153, with D/C/K
~1e-6 near R~5.885 rg. Transition-shell runs using that critical point make
flow/micro residuals tiny, but the global solve stalls at physical~0.09 for
N96 and ~0.47 for N128, dominated by tail_R/tail_E near R~6--9 rg. Weighted
tail continuation does not fix this.

Next: stop forcing the exact branch onto the old tail. Implement a
multiple-shooting junction solve.

1. Use the backward-shooting critical point as the sonic anchor.
2. Integrate exact sonic branch outward to R_join.
3. Use a free transition/tail segment from R_join to R_match; old dynamic tail
   is only an initial guess, not a penalty.
4. Scan R_join = 6.5,7,8,10,12,15 rg.
5. Add branch_connection_scan comparing exact sonic branch to old outer branch:
   Delta logu, Delta logT, Delta derivatives, entropy, Omega/OmegaK.
6. Continue in lambda0 around old value -0.0015.
7. Optionally reverse-shoot the outer branch inward and match it to the exact
   sonic branch by varying lambda0 and R_join.
8. Stop high-Mdot continuation until this fixed-Mdot exact sonic/global branch
   is found or ruled out.
```

---

# 16. Bottom line

The latest runs moved the problem forward.

They show:

```text
near-sonic exact branch: works
micro/transition region: can be made small
tail/global connection: fails
```

So the next numerical method should be a **multiple-shooting branch-connection
solver**, not another patch refinement.
