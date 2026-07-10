# Mdot=5 Lobatto Radial Transition Results

Date: 2026-07-09

Target:

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact source
- local-Mdot wind
- `eta_E = 98.125`
- `N = 164`

Starting checkpoint:

```text
outputs/checkpoints/m5_eta_lobatto_state_corrector_mass100_node1e3_98p125_N164/stage_00_etaE_98p125_N164.npz
```

## Implementation

Added passive radial matrix conditioning diagnostics for true Lobatto source-element points:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_CONDITIONING_AUDIT=1
```

For every Lobatto left/mid/right point, the audit records:

- `cond(A)`, singular values, and `abs(det(A))`;
- direct slope `g_direct = -A^{-1}c`;
- true Lobatto radial/energy residuals;
- direct-slope norm and source-minus-direct slope norm.

Added opt-in left/right transition block controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_TRANSITION_BLOCK
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_TRANSITION_INTERVALS
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_TRANSITION_ENERGY_WEIGHT_FACTOR
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_TRANSITION_CURVATURE_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_TRANSITION_OUTSIDE_ANCHOR_WEIGHT
```

The transition block extends the Lobatto window by a small number of intervals. In transition intervals, radial ODE rows stay at full weight and energy rows can be downweighted initially. Optional curvature rows penalize second differences of Lobatto midpoint slopes across the transition. Optional outside anchors keep the already-corrected source interior near the seed while allowing the transition and first source-facing element to move.

## Conditioning Audit

Baseline evaluate-only run:

```text
m5_eta_lobatto_radial_conditioning_audit_98p125_N164
```

Key result:

```text
radial peak = 0.197252 at interval 133 left, R=203.104 rg
cond(A) at radial peak = 4.500e3
smin(A) at radial peak = 9.718e-4
abs(det(A)) at radial peak = 4.249e-3
|g_direct| at radial peak = 2.017e2
max cond(A) = 2.008e5 at R=250.431 rg
```

Interpretation:

- The radial peak is not the globally worst-conditioned Lobatto point.
- It is still moderately near-singular and demands an unphysical direct slope of order 200.
- This confirms that matching a finite neighboring slope is not enough; the local radial equation itself is stiff/near-singular at the interface.

## Transition Tests

| run | full | ODE | radial | energy | FV mass | peak | cond@peak | `|g_direct|` | nfev |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `radial_conditioning_audit_98p125_N164` | 5.980e-1 | 1.973e-1 | 1.973e-1 | 6.221e-3 | 1.538e-4 | 133/left @ 203.1 | 4.500e3 | 2.017e2 | 1 |
| `transition_left2_curv1e4_98p125_N164` | 1.330e0 | 2.198e-1 | 2.198e-1 | 1.894e-1 | 2.787e-3 | 145/left @ 232.9 | 7.657e3 | 2.519e2 | 9 |
| `transition_left2_anchor100_curv1e4_98p125_N164` | 6.888e-1 | 2.376e-1 | 2.376e-1 | 1.909e-2 | 7.114e-5 | 131/left @ 198.8 | 4.414e3 | 2.266e2 | 15 |
| `transition_left2_anchor100_nocurv_98p125_N164` | 6.873e-1 | 2.371e-1 | 2.371e-1 | 1.826e-2 | 6.397e-5 | 131/left @ 198.8 | 4.412e3 | 2.266e2 | 13 |
| `transition_left1_anchor100_nocurv_98p125_N164` | 6.432e-1 | 2.301e-1 | 2.301e-1 | 9.701e-3 | 5.973e-5 | 132/left @ 200.9 | 4.511e3 | 2.164e2 | 13 |
| `transition_left1_r10_nocurv_98p125_N164` | 1.243e0 | 5.067e-1 | 1.786e-1 | 5.067e-1 | 2.958e-3 | 132/left @ 200.9 | 8.211e3 | 1.939e2 | 120 |

## Findings

The transition block did not meet the acceptance criteria.

Balanced transition variants:

- reduce FV mass somewhat;
- do not reduce the radial defect;
- move the radial peak to the new left transition boundary;
- worsen the global full residual.

The high-radial-weight variant reduces the radial maximum from `0.197` to `0.179`, but only by letting energy explode to `0.507` and FV mass to `2.96e-3`. This is not a physical or numerically acceptable correction.

The optional curvature regularization is not the main problem. Removing it gives nearly the same result for the anchored two-cell transition. The dominant failure mode is the moving-boundary radial defect.

## Interpretation

The true Lobatto source element remains internally much better than the earlier HS/FV representation, but the left radial interface is not fixed by adding one to two ordinary Lobatto transition elements. The interface behaves like a radial critical/near-critical layer:

- extending the Lobatto window exports the defect to the new left edge;
- local direct slopes remain order `200`;
- radial conditioning at the defect is moderate-to-large but not uniquely singular;
- forcing radial residual harder trades the defect into energy and FV mass.

Therefore, the next formulation step should not be another wider transition halo or stronger smoothness penalty.

## Recommended Next Step

Implement a desingularized radial-interface formulation for the left source transition:

1. Treat the radial interface as a local phase-space/DAE segment rather than an `x=lnR` slope element.
2. Introduce an implicit tangent/null-vector form for the radial row near the interface:

```text
B(z) p = 0,   B = [c_R, A_R]
```

with a normalization for `p`.

3. Couple that radial phase segment to the Lobatto energy/FV mass rows as guards, not as hard slope inversion.
4. Use the current best true Lobatto checkpoint as the seed:

```text
m5_eta_lobatto_state_corrector_mass100_node1e3_98p125_N164
```

5. Keep the new conditioning audit enabled as the regression gate.

Do not continue `eta_E` yet, and do not add more source/wind physics until this radial interface layer is resolved.
