# GPT Review Prompt: Desingularized Phase-Space Barrier Result

Repository:

```text
https://github.com/huanyang07/IMBH
```

Please review the latest pushed `main` branch, starting with these files:

```text
Note/CODEX_SECOND_CRITICAL_DESINGULARIZED_DAE_PLAN.md
scripts/run_transonic_desingularized_barrier_flow.py
outputs/tables/transonic_desingularized_barrier_flow_ds5e4.md
outputs/tables/transonic_desingularized_barrier_flow_ds5e4_trace.json
outputs/figures/transonic_desingularized_barrier_flow_ds5e4.png
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
tests/test_transonic_local.py
```

## What Codex implemented

Codex implemented the desingularized phase-space formulation suggested in the
previous review:

```text
z = (x, logu, logT), x = ln R
p = dz/ds
B = [c A]
B(z) p = 0
```

The helper functions added in `transonic_local.py` are:

```text
extended_phase_space_matrix(...)
phase_space_null_tangent(...)
```

The new audit script is:

```text
scripts/run_transonic_desingularized_barrier_flow.py
```

It integrates the right-null tangent of `B=[c A]` with a Heun predictor-
corrector stepper, tracking `R/rg`, signed `p_x`, `smin/smax(A)`,
`smin/smax(B)`, sonic diagnostics, and physical diagnostics.

Full tests pass:

```text
128 passed
```

## Key result

The incoming exact branch starting from `R=6 rg` does not cross outward through
the second-critical barrier. It reaches a projected fold:

```text
R_max = 6.237899204 rg
```

Around the maximum, from
`outputs/tables/transonic_desingularized_barrier_flow_ds5e4_trace.json`:

```text
s          R/rg          p_x             smin/smax(A)      smin/smax(B)
2.433999   6.237899204   +4.4955e-07     3.9290e-09        1.9043e-05
2.434499   6.237899204   -3.0186e-08     2.6394e-10        1.9035e-05
2.434999   6.237899203   -5.0961e-07     4.4578e-09        1.9027e-05
```

Physical quantities stay tame near the fold:

```text
H/R ~= 5.12e-4
Omega/Omega_K ~= 1
```

Interpretation:

```text
B remains regular enough at the incoming fold, but p_x changes sign.
Therefore the phase-space curve is locally regular while its radial projection
turns around. The y(logR) ODE fails because dx/ds -> 0, but the branch does not
continue monotonically to larger R under the current fixed-lambda closure.
```

The fixed-lambda critical candidate near `R ~= 6.118 rg` was also tested, but it
is not a clean local seed:

```text
smin/smax(B) ~= 1.397e-18
classification = B_singular_or_degenerate
```

## Consequence

Codex did not build the phase-space collocation segment to `R=7--8 rg`, because
the gating test failed. The local null-flow itself does not reach those radii;
it turns around at `R ~= 6.238 rg`.

## Questions for GPT

Please advise on the best next mathematical/numerical step.

In particular:

1. Does this result imply the current fixed-lambda exact sonic branch is a
   physical/radial branch termination, or should we search for an outgoing
   branch via a two-critical/branch-switching DAE?
2. Is the nearby free-lambda candidate near `R ~= 6.219 rg` best interpreted as
   a neighboring global solution family, and should we continue the whole
   solution in `lambda0` toward it?
3. What branch condition should be used if we formulate a two-critical BVP with
   incoming and outgoing phase-space tangents?
4. Should the next experiment be:
   - global continuation in `lambda0`,
   - a local branch-switch/nullspace analysis at the fold,
   - a two-critical DAE BVP,
   - or a revision of the physical closure/outer boundary?
5. What acceptance criteria would distinguish a valid outgoing disk branch from
   a numerical artifact?

Please be concrete about equations, residuals, unknowns, and diagnostics to add
next.
