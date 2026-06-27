# GPT Handoff: Exact Sonic Branch Hits a Second-Critical Barrier

Repository:

```text
https://github.com/huanyang07/IMBH
```

Latest relevant commit before this handoff:

```text
54ab73c Add branch barrier diagnostics
```

## Bottom line

The exact reverse-seeded sonic branch is locally regular near the first sonic
point, but it cannot be extended as an ordinary ODE branch to the old tail.
It hits a second singular/critical structure near:

```text
R ~= 6.238 rg
```

The local thermodynamic state remains tame there:

```text
H/R ~= 5e-4
Omega/Omega_K ~= 0.9996
Qadv/Qvisc ~= 1
```

but the ODE matrix becomes singular:

```text
smin/smax ~= 8e-11
condition number ~= 1e10
max |dy/dlnR| ~= 1e8
```

So the next problem is not simply outer-tail matching. It is how to continue,
regularize, or formulate the branch through/around this second-critical
barrier.

## Current model state

The numerical branch under discussion is the fixed-Mdot transonic minidisk
near:

```text
Mdot/Edd ~= 0.90277664
Rson ~= 5.885 rg
lambda0 ~= 3.6747
```

The old dynamic sonic patch gave a useful approximate branch, but replacing it
with an exact L'Hopital/flow-map sonic treatment revealed a mismatch between
the exact branch and the old tail.

The latest exact sonic branch is seeded from the backward/reverse critical fit.
The first sonic point is good:

```text
D, C1, C2, K ~= 1e-8 to 1e-7
flow/micro residuals ~= 1e-6
```

However, previous global BVP attempts still stall at:

```text
physical residual ~= 0.09
dominant residual = tail_R
worst region ~= 7 to 9 rg
```

The new scans show that before reaching that old tail-residual region, the
exact branch itself encounters a second singular barrier near `6.24 rg`.

## Key new files

### Scripts

```text
scripts/run_transonic_branch_connection_scan.py
scripts/run_transonic_branch_barrier_audit.py
scripts/run_transonic_short_join_free_tail_scan.py
scripts/run_transonic_barrier_critical_probe.py
scripts/run_transonic_second_critical_flowmap_probe.py
```

### Output tables

```text
outputs/tables/transonic_branch_connection_scan.md
outputs/tables/transonic_branch_connection_scan.json
outputs/tables/transonic_branch_barrier_audit.md
outputs/tables/transonic_branch_barrier_audit_points.json
outputs/tables/transonic_short_join_free_tail_scan.md
outputs/tables/transonic_barrier_critical_probe.md
outputs/tables/transonic_second_critical_flowmap_probe.md
```

### Figures

```text
outputs/figures/transonic_branch_connection_scan.png
outputs/figures/transonic_branch_barrier_audit.png
```

## What the branch-connection scan showed

File:

```text
outputs/tables/transonic_branch_connection_scan.md
```

The scan integrated the exact sonic branch outward and compared it to both:

```text
1. the current best exact-BVP N96 profile
2. the old dynamic-patch source profile
```

It included:

```text
R_join/rg = 5.95, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 7, 8, 10, 12, 15
```

Main findings:

```text
Exact branch reaches only R ~= 6.238 rg.
It cannot reach R_join >= 6.3 rg by ordinary ODE integration.
```

Best finite match to the current best exact-BVP profile:

```text
case    = free_reverse_fit
branch  = 1
R_join  = 6.1 rg
dist_y  = max(|Delta logu|, |Delta logT|) ~= 1.48e-3
dlogu   ~= -1.48e-3
dlogT   ~= 8.93e-4
critK   ~= 4.08e-8
```

Best finite match to the old dynamic-patch profile is bad:

```text
R_join ~= 5.95 rg
dist_y ~= 0.69
```

Interpretation:

```text
The old dynamic tail is not on the exact branch.
The exact branch is close to the best exact-BVP profile near R~6.1,
but it cannot be propagated to the old tail region.
```

## Barrier audit

File:

```text
outputs/tables/transonic_branch_barrier_audit.md
```

For the preferred free reverse-fit branch:

```text
case    = free_reverse_fit
branch  = 1
class   = singular_matrix_or_second_critical
reached R/rg = 6.238
reached dx   = 0.05827
min smin/smax = 8.382e-11
max condA = 1.193e10
max |g| = 9.34e7
last D = 8.382e-11
last K = 4.084e-4
last H/R = 5.122e-4
last Omega/Omega_K = 0.9996
last Qadv/Qvisc = 1
message = Required step size is less than spacing between numbers.
```

The dense trace is stored in:

```text
outputs/tables/transonic_branch_barrier_audit_points.json
```

Interpretation:

```text
This is not an obvious physical blow-up.
The solution remains thin and nearly Keplerian.
The radial ODE fails because the differential matrix becomes singular and
the local slope diverges.
```

## Short-join free-tail test

File:

```text
outputs/tables/transonic_short_join_free_tail_scan.md
```

I tested free-tail joins before the singular barrier. N96 was too slow for a
full scan because each sparse-Jacobian evaluation is expensive, so I ran an
exploratory N48 scan:

```text
R_join = 6.0, 6.1, 6.2 rg
```

Results after polish:

```text
R_join=6.0 rg: physical ~= 0.0681, dominant tail_R
R_join=6.1 rg: physical ~= 0.0877, dominant tail_R
R_join=6.2 rg: physical ~= 0.1055, dominant tail_R
```

Interpretation:

```text
Joining earlier helps somewhat but does not remove the residual floor.
The best short join is around 6.0 rg, but it still leaves a tail_R residual
of order 7e-2.
```

## Critical-compatibility probe near the barrier

File:

```text
outputs/tables/transonic_barrier_critical_probe.md
```

I took the last pre-barrier states and solved locally for:

```text
D = 0
C1 = 0
C2 = 0
K = 0
```

Two types of compatible critical candidates were found.

### Fixed lambda0

At the current branch `lambda0`:

```text
R ~= 6.118 rg
critK ~= 4e-17
dlogR ~= -0.0194 from the final barrier state
dlogu ~= -0.121
dlogT ~= -0.241
H/R ~= 4.45e-4
Omega/Omega_K ~= 1
M_eff ~= 1.225
```

### Free lambda0

Allowing lambda0 to move gives:

```text
R ~= 6.219 rg
critK ~= 4e-17 to 1e-16
dlambda0 ~= +0.0012
H/R ~= 4.95e-4
Omega/Omega_K ~= 1
M_eff ~= 1.224
```

Interpretation:

```text
There are algebraically compatible critical points near the barrier.
But compatibility alone does not prove the branch can pass through them.
```

## Second-critical flow-map probe

File:

```text
outputs/tables/transonic_second_critical_flowmap_probe.md
```

I then asked whether the compatible second-critical candidates have regular
L'Hopital derivative branches. The answer was no:

```text
fixed_lambda candidate: no L'Hopital derivative branches found
free_lambda candidate:  no L'Hopital derivative branches found
```

I manually widened the derivative-root scan up to:

```text
a_half_width = 1e6
```

and still found:

```text
0 derivative branches
```

Interpretation:

```text
The candidates are compatible in the algebraic D,C,K sense, but they are not
regular crossing points under the current L'Hopital derivative formulation.
This may be a real degenerate critical point, a scaling/null-vector issue, or
evidence that the branch must be treated as a DAE/two-critical BVP rather than
as an ODE flow map.
```

## Current interpretation

The latest evidence suggests:

```text
1. The first sonic point is regular and usable.
2. The exact ODE branch encounters a second singular critical structure at
   R ~= 6.24 rg.
3. The old dynamic-patch tail is not directly reachable from the exact branch.
4. Short-join free-tail solves before the second barrier reduce but do not
   remove the tail residual floor.
5. Algebraically compatible second-critical points exist, but they have no
   L'Hopital derivative branches in the current formulation.
```

So the next step is probably not more tail weighting, more direct N128, or
high-Mdot continuation. The next step is to understand the second-critical
regularity problem.

## Questions for GPT

Please focus advice on the following.

### 1. What does "compatible but no L'Hopital derivative branch" imply?

At the second-critical candidates:

```text
D = C1 = C2 = K = 0
```

to numerical precision, but scanning the L'Hopital derivative condition finds
no root. Does this indicate:

```text
a. a degenerate/tangent critical point,
b. a fold where logR is the wrong independent variable,
c. a scaling/null-vector artifact in the current L'Hopital test,
d. an incomplete regularity condition,
e. or a sign that no physical smooth transonic branch passes through?
```

### 2. How should the second-critical regularity condition be derived?

Current implementation uses the same sonic derivative machinery as the first
critical point:

```text
A(logR, y, lambda0) g + c(logR, y, lambda0) = 0
D = det-like singularity measure
K = left-null compatibility
g = g_p + a r
L'Hopital scalar condition scans over a
```

What is the mathematically correct regularity condition at a second critical
point where the branch may be tangent to the critical manifold?

### 3. Should we switch to a DAE or two-critical-point BVP?

A likely next numerical form is:

```text
Segment A: first sonic critical point -> before second critical
Segment B: second critical point treated as an algebraic interior point
Segment C: after second critical -> outer tail
```

Unknowns would include:

```text
first critical state
second critical state
lambda0
left/right slopes or tangent variables
free tail nodes
outer two-domain nodes
```

Residuals might include:

```text
D1,C1,K1 at first critical
regularity/tangent condition at first critical
D2,C2,K2 at second critical
regularity/tangent condition at second critical
collocation on both sides
continuity at the second critical
far pressure-supported boundary
```

Is this the right direction? If yes, what exact residuals should be used at the
second critical point?

### 4. Is a local desingularized ODE possible?

Can the system be desingularized near the second critical point by using:

```text
adj(A)(A g + c) = 0
or a null-coordinate chart
or dy/ds, dlogR/ds pseudo-arclength local flow
```

If so, what are the equations and what should be used as the crossing/tangent
condition?

### 5. Is the fixed-lambda vs free-lambda split physically important?

The fixed-lambda critical candidate is at:

```text
R ~= 6.118 rg
```

The free-lambda candidate near the final ODE barrier is at:

```text
R ~= 6.219 rg
dlambda0 ~= +0.0012
```

Does this mean the branch would need lambda0 continuation to pass near the
barrier, or is the `R=6.118 rg` fixed-lambda critical point the relevant one?

## Suggested next Codex experiment, unless GPT advises otherwise

My current best next experiment is:

```text
Build a local second-critical regularity audit.

1. Evaluate raw, scaled, and frozen_scaled L'Hopital scalar functions at the
   fixed-lambda and free-lambda second-critical candidates.
2. Plot L(a) over very wide a ranges and inspect whether roots are truly absent
   or hidden by scaling.
3. Compute the critical-manifold tangent from finite-difference Jacobians of
   [D, K] or [D, C1, C2, K] with respect to [logR, logu, logT, lambda0].
4. Compare this tangent with the ODE null direction r and with the incoming
   branch tangent from the barrier audit.
5. If the tangent structure is well-defined, implement a two-critical-point
   local BVP/DAE seed before attempting a global solve.
```

## Tests

After the latest scripts/results:

```text
python -m pytest
125 passed
```

Also:

```text
py_compile passed for the new scripts
git diff --check passed
```
