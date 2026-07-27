# Causal Inner Frozen Domain Hardening WP10c9d5a1 Results

Date: 2026-07-28
Analyzed base: `155e18339076fd2b27d419173b92e1d5d608963b`
Work package: WP10c9d5a1

## Classification

```text
inner_domain_derivative_certified_cache_first_localization_authorized_
global_hardening_still_failed
```

WP10c9d5a remains globally rejected. This package does not relax or replace
its predeclared gate. It proves the narrower statement that the stored frozen
candidate derivative is sufficiently stable on residual rows through
`5 rg`, together with the complete three-cell reconstruction halo, to permit
cache-first nested control-volume localization.

This authorizes WP10c9d5b only. It does not authorize:

- global frozen-candidate recertification;
- a production operator;
- a nonlinear candidate;
- fixed-`Q` averaging;
- reduced slow-time evolution.

## Spatial localization of the WP10c9d5a failure

The binding embedded `random_0` direction was projected cell by cell. For the
`2e-5 -> 4e-5` direct-JVP change:

```text
dominant cell                         63
dominant center                       24.0598191 rg
outermost-cell squared fraction       0.9708188
outer-three-cell squared fraction     0.9722852
rows through 5 rg squared fraction    0.0065822
first-three-cell squared fraction     0.0003596
```

Thus the formal global failure is an outer-boundary-region derivative
sensitivity. It is not a first-cell or near-excision derivative failure.

## Scoped original-direction result

The original failed direction was reevaluated without changing its
normalization, the stored matrix, or the step ladder.

| Residual domain | Selected matrix defect | `2e-5 -> 4e-5` change | `4e-5 -> 8e-5` change | Result |
|---|---:|---:|---:|---|
| Complete grid | `3.0113e-5` | `2.0986e-5` | `4.1826e-5` | fail |
| Through `5 rg` | `4.6226e-6` | `4.1340e-6` | `8.2726e-6` | pass |
| Through `5 rg` plus three-cell halo | `5.2235e-6` | `3.8546e-6` | `7.7174e-6` | pass |
| Outermost cell | `5.2172e-5` | `3.6566e-5` | `7.2918e-5` | fail |

The unchanged gates are:

```text
selected matrix-action defect    <= 5e-5
both adjacent plateau changes    <= 2e-5
```

## Dense/colored and sparsity certification

Every primitive column in the 25 cells through `5 rg`, plus the three-cell
halo, was recomputed densely at the stored `4e-5` component step.

```text
selected columns                 140
dense/colored relative defect    0.0
off-pattern relative entry       0.0
maximum per-column defect        0.0
```

This extends the earlier first-three-cell check to every column capable of
affecting the declared localization domain.

## Held-out continuum directions

Four directions were fixed before inspection:

- two smooth fields supported on `1.8-5 rg`;
- two near-excision fields supported on `1.8-3.5 rg`.

They use deterministic continuum profiles and a cell-volume-weighted scaled
primitive norm. All four pass on both the inner rows and the stencil halo.
Their largest selected-step defect is `3.1173e-5`; their largest adjacent
change bracketing the selected step is `1.6087e-6`.

## Branch fingerprint

The exact failed direction was evaluated at:

```text
-4e-5, -2e-5, 0, +2e-5, +4e-5.
```

Across this ladder:

```text
characteristic sign changes              0
admissibility-factor change              0
outer choking/count changes              0
minimum |speed|                           0.32439 c
minimum pairwise family gap               0.002699 c
base maximum descriptor condition         1.89089e4
face of maximum condition                 outer face
incoming excision characteristics         0
```

The hard stationary-speed mask, limiter/admissibility branch, and choking
branch are therefore not controlling this failure. The outer-face descriptor
conditioning remains a live issue for later global hardening.

## Decision

WP10c9d5b cache-first localization is authorized on common physical faces
from excision through `5 rg`. It must:

1. reuse the unchanged WP10c9d5 histories and generators;
2. retain complete mapped and responsive-height storage;
3. decompose every conservative, principal, lower-source, and
   production-anchor storage action;
4. identify a recovery radius only when instantaneous and cumulative M/J/E
   face histories pass at two consecutive surfaces;
5. leave global candidate promotion blocked.

## Evidence and verification

Canonical evidence is committed under:

```text
results/canonical/causal_inner_frozen_domain_hardening_wp10c9d5a1/
```

The binding run took `1444.12 s`. The focused method and canonical suite
passes with `8 passed`.
