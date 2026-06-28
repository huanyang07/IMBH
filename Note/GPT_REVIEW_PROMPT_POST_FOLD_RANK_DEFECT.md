# GPT Review Prompt: Fold Confirmed, Lambda Map, and B Rank-Defect Accessibility

Repository:

```text
https://github.com/huanyang07/IMBH
```

Please review the latest pushed `main` branch. Start with:

```text
Note/CODEX_POST_DESINGULARIZED_FOLD_NEXT_STEPS.md
outputs/tables/transonic_desingularized_fold_refinement.md
outputs/tables/transonic_lambda_family_fold_map.md
outputs/tables/transonic_B_rank_defect_search.md
scripts/run_transonic_desingularized_fold_refinement.py
scripts/run_transonic_lambda_family_fold_map.py
scripts/run_transonic_B_rank_defect_search.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
tests/test_transonic_local.py
```

## What Codex implemented

Codex completed the next diagnostics after the desingularized phase-space
barrier test:

1. Refined the radial projection fold event.
2. Added local phase-space helper diagnostics:
   - `B_rank_minors(...)`
   - `phase_space_tangent_derivative(...)`
3. Mapped local/strict `lambda0` fold behavior.
4. Searched for `B=[c A]` rank-defect candidates and measured accessibility
   from the incoming exact branch.

Full tests pass:

```text
130 passed
```

## Result 1: fold is confirmed

The event-refined fold is:

```text
R_fold = 6.23789926 rg
```

It is stable across:

```text
ds = 5e-4, 2e-4, 1e-4, 5e-5
metrics = euclidean, x4
```

At the fold:

```text
p_x ~= 0
dp_x/ds = -9.592e-4
smin/smax(B) = 1.904e-5
H/R = 5.122e-4
Omega/Omega_K = 0.999586
```

Interpretation:

```text
The fixed-lambda incoming branch has a genuine local maximum in R(s).
B is not rank-deficient at the fold, so this is not itself a branch-switching
point.
```

## Result 2: lambda map

Codex separated two diagnostics:

### Strict fixed-buffer critical seed

Only the reference lambda is accepted:

```text
lambda_ref = 3.674721616
R_fold = 6.237899009 rg
```

All nonzero lambda offsets in `[-0.004, +0.004]` fail the first-sonic critical
seed test under the current fixed-buffer closure:

```text
negative offsets: critK ~ 2.9e-4 to 1.2e-3
positive offsets: critical seed fails / infinite critK
```

So this does **not** yet provide a valid neighboring global critical family.

### Local R=6 lambda sensitivity

This is not a global solution, but it shows how the already-known local branch
geometry responds to changing lambda:

```text
delta_lambda = 0       -> R_fold = 6.237899 rg
delta_lambda = +0.001  -> R_fold = 6.376928 rg
delta_lambda = +0.002  -> R_fold = 6.486018 rg
delta_lambda = +0.004  -> R_fold = 6.663279 rg
```

The fold moves outward with larger lambda, but still does not reach the target
`R=8 rg` in this scan.

## Result 3: B rank-defect search

The incoming fold is accessible but not a rank-defect point:

```text
R = 6.237899262 rg
lambda0 = 3.674721616
smin/smax(B) = 1.904e-5
relative minor norm = 4.630e-2
```

The fixed-lambda critical candidate is a true rank-defect point:

```text
R = 6.118209949 rg
lambda0 = 3.674721616
smin/smax(B) = 1.397e-18
relative minor norm = 5.124e-13
access distance from incoming branch = 5.73
```

The free-lambda critical candidate is also a true rank-defect point:

```text
R = 6.219062068 rg
lambda0 = 3.675931361
smin/smax(B) = 2.200e-18
relative minor norm = 7.247e-13
access distance including lambda = 1.84
```

Interpretation:

```text
True B-rank-defect points exist, but the current incoming exact branch does not
reach an accessible rank-defect point. Therefore a branch-switching BVP is not
justified yet.
```

## Current conclusion

The current no-wind, algebraic-stress, fixed-lambda branch appears blocked:

```text
1. The incoming branch folds before reaching the outer disk.
2. The fold is not a B rank-defect / branch-switch point.
3. Known B rank-defect points are not accessible from the incoming branch.
4. Strict lambda continuation is not viable under the current fixed-buffer
   critical seed.
```

## Questions for GPT

Please advise on the best next step. In particular:

1. Should we now prioritize a better **global-family continuation** where the
   buffer/outer profile moves with `lambda0`, rather than perturbing lambda
   locally?
2. Is there a mathematically clean way to continue from the free-lambda
   rank-defect candidate into a neighboring full solution family?
3. Should we formulate a more general two-critical DAE even though the known
   rank-defect points are not accessible from the current incoming branch?
4. Or should we stop pursuing this no-wind closure and move to revised physics:
   wind, altered stress closure, or explicit jump/source conditions?
5. If the recommended next move is global-family continuation, please specify:
   - unknowns,
   - residuals,
   - continuation parameter,
   - boundary conditions,
   - and acceptance criteria.

Please be concrete about equations and numerical diagnostics to add next.
