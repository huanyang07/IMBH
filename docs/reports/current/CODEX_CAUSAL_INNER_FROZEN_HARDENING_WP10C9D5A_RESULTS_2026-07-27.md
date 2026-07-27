# Causal Inner Frozen Hardening WP10c9d5a Results

Date: 2026-07-27
Scientific implementation: `038ba35659e76aff0605fffa5fb457e99362063d`
Exact parent: `42dd7f1d4ca048fcbd2faa02e71e0a66db300891`
Scientific tree: `a1e4e33378154d91d17afe001479b063b74ca27f`
Work package: WP10c9d5a

## Classification

```text
frozen_jacobian_hardening_failed_dynamic_localization_blocked
```

The provenance and dense/colored sparsity checks pass. The full numerical
hardening gate does not pass because one predeclared embedded random direction
does not place the stored `4e-5` Jacobian step inside a two-interval stable
finite-difference plateau.

This is a narrow method failure, not evidence for a new physical defect. The
gate was declared before the run and was not relaxed after inspection.

WP10c9d5b dynamic inner-export localization therefore remains blocked. This
result does not authorize:

- a boundary or first-cell ablation;
- a new global spatial candidate;
- WP10c9d6 nonlinear work;
- a production-operator change;
- fixed-`Q` averaging;
- reduced slow-time evolution.

## Provenance correction

Git resolves the scientific WP10c9d5 commit as:

```text
implementation commit  038ba35659e76aff0605fffa5fb457e99362063d
parent commit          42dd7f1d4ca048fcbd2faa02e71e0a66db300891
tree                   a1e4e33378154d91d17afe001479b063b74ca27f
```

The earlier WP10c9d5 report and canonical metadata recorded a different
40-character object sharing the same short prefix. WP10c9d5a corrects:

- the WP10c9d5 runner declaration;
- `summary.json`;
- `provenance.json`;
- the focused canonical-evidence test;
- this report and the WP10c9d5 report.

The correction is metadata only. The WP10c9d5 decisive array archive remains
bitwise unchanged:

```text
384c17dc99c3a739015d5298b8a06c416b134d0b1591a9e4c1c2360aa9dbee8b
```

## Self-contained replay bundle

The binding uniform N64 and embedded N128-exterior/N128-inner contexts are
committed as:

```text
results/canonical/causal_inner_frozen_hardening_wp10c9d5a/replay_inputs.npz
results/canonical/causal_inner_frozen_hardening_wp10c9d5a/replay_contexts.json
```

They include the exact grids, base primitives, scaling, common directions,
stored colored stationary corrections, boundary context, providers, source
rates, and reference base residuals required for the hardening audit.

Both reconstructed contexts reproduce their stored scaled base residual
bitwise. A clean checkout can rerun the audit from these committed inputs
without the ignored generator caches or upstream `outputs/tables` files.

## Dense-versus-colored and sparsity result

The audit compares the stored colored central-difference Jacobian against
independently evaluated dense columns using the same candidate residual and
the same `4e-5` component step.

| Configuration | Dense columns | Dense/colored defect | Off-pattern entry | Result |
|---|---:|---:|---:|---|
| Uniform N64 | `120/120` | `0.0` | `0.0` | pass |
| Embedded N128-inner | first `15` | `0.0` | `0.0` | pass |

Thus:

1. the coloring implementation reproduces the corresponding dense finite
   differences exactly;
2. all uniform-grid columns obey the declared local pattern;
3. the first three embedded cells contain no detected derivative outside the
   declared pattern.

This removes an incomplete coloring or missing local band entry as an
explanation for the earlier WP10c9d5 common-direction result. It does not
certify the finite-difference step itself.

## Seven-step directional audit

Every direction was evaluated at:

```text
5e-6, 1e-5, 2e-5, 4e-5, 8e-5, 1.6e-4, 3.2e-4
```

The declared gates were:

```text
selected 4e-5 matrix-action defect        <= 5e-5
adjacent-action plateau change            <= 2e-5
minimum consecutive plateau changes       = 2
selected 4e-5 step must lie on plateau     yes
```

Directions included:

- the exact common mode;
- that mode restricted to the first three cells;
- two fixed-seed random admissible directions;
- each of the five first-cell primitive coordinate directions.

All nine uniform directions pass. Eight of nine embedded directions pass.

### Binding embedded failure

The only failure is `random_0` on the embedded configuration.

Its matrix-action defects are:

```text
3.58484e-5, 3.21723e-5, 2.76217e-5, 3.01128e-5,
6.18632e-5, 1.41946e-4, 3.07678e-4
```

Its adjacent direct-action changes are:

```text
6.22196e-6, 1.07525e-5, 2.09856e-5,
4.18256e-5, 8.35906e-5, 1.67106e-4
```

The first two adjacent changes define a stable three-action window from
`5e-6` through `2e-5`. The next change, between `2e-5` and the stored `4e-5`
step, is `2.09856e-5`, just above the declared `2e-5` gate. The selected
matrix-action defect itself is acceptable (`3.01128e-5 < 5e-5`), but the
selected step is not on the certified plateau.

The common, first-three-cell, and all five first-cell coordinate directions
pass on the same embedded context. The failure therefore identifies a
direction-dependent finite-difference step limitation rather than a
first-cell sparsity error.

## Decision

WP10c9d5a succeeds in:

1. correcting exact Git provenance without changing scientific arrays;
2. making the binding hardening contexts reproducible from committed data;
3. proving exact dense/colored parity for the tested columns;
4. proving the tested actual sparsity contract;
5. replacing the earlier three-step check with a seven-step directional
   record.

It fails the complete hardening gate. WP10c9d5b is not authorized from this
evidence.

## Recommended next package

The next work should remain numerical and production neutral:

1. predeclare a new derivative construction rather than retroactively changing
   the WP10c9d5a gate;
2. use `2e-5`, or an analytic/AD-compatible directional derivative, to build
   an independent candidate Jacobian;
3. validate it on new held-out random directions and the same common,
   first-three-cell, and first-cell directions;
4. require the selected construction to lie on a certified plateau for every
   direction;
5. rerun the frozen hardening classification without changing the rejected
   WP10c9d5 physical-export arrays.

Only a passing independent hardening package may authorize WP10c9d5b nested
control-volume localization.

## Evidence and verification

Compact decisive evidence is committed under:

```text
results/canonical/causal_inner_frozen_hardening_wp10c9d5a/
```

The binding run took `1194.08 s`. Focused verification completed with:

```text
WP10c9d5a/WP10c9d5/radial focused suite: 11 passed
canonical/hygiene expanded focused suite:    18 passed
full repository suite:                       821 passed, 4 subtests passed
```
