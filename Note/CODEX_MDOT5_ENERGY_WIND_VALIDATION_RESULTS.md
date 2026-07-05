# Mdot=5 Energy-Wind Validation Sprint Results

Date: 2026-07-05

This sprint followed the GPT review in
`Note/CODEX_MDOT5_ENERGY_WIND_ENDPOINT_REVIEW_AND_PLAN.md`.

## Executive Summary

The current energy-only wind branch is now much better certified numerically:

- Anchor freeze completed for `epsilon_w = 0, 0.50, 0.80, 0.98, 0.997` and
  `eta = 6.20, 6.30, 6.35`.
- Directional wind-Jacobian audit passed provisionally.
- Mesh validation passed at `N=768, 896, 1024` for:
  - `epsilon_w = 0.98`
  - `eta = 6.20`
  - `eta = 6.35`
- Manual small-step eta continuation advanced beyond the previous endpoint:
  - `eta = 6.375`
  - `eta = 6.40`
  - `eta = 6.425`
- A frozen closure-sensitivity audit was added.
- A first full closure retargeting pilot succeeded for a moderate wind strength.
- A mass-coupled wind budget diagnostic shows the key physical caveat is large:
  the high-eta energy sink implies a mass loss of order several times
  `Mdot_inner` if converted to an escape-energy wind.

Bottom line: the branch is numerically robust as an **energy-loss / wind-cooling
branch**, but still should not be called a fully physical mass-loaded wind
branch.

## Key Outputs

- Anchor freeze:
  `outputs/tables/m5_energy_wind_anchor_freeze.md`
- Jacobian audit:
  `outputs/tables/m5_energy_wind_interval_jacobian_audit.md`
- Mesh validation:
  `outputs/tables/m5_energy_wind_mesh_validation_summary.md`
- Frozen closure audit:
  `outputs/tables/m5_energy_wind_closure_sensitivity_frozen.md`
- Closure retargeting pilot:
  `outputs/tables/m5_energy_wind_closure_repolish_q02_chi985_w010_eps0968.md`
- Implied mass-coupled budget:
  `outputs/tables/m5_energy_wind_implied_mass_coupled_budget.md`
- Eta extension:
  `outputs/tables/m5_energy_wind_eta_adaptive_manual_6375_640_N896.md`
  and `outputs/tables/m5_energy_wind_eta_adaptive_manual_6425_N896.md`

## Anchor Freeze

The frozen anchors recompute diagnostics directly from checkpoints:

| state | residual | Qwind/Qvisc | Lrad/LEdd | f_adv_global | f_adv_inner | max H/R | Rson/rg |
|---|---:|---:|---:|---:|---:|---:|---:|
| epsilon_w=0 | 2.783e-10 | 0 | 1.2996 | 0.4990 | 0.4716 | 0.3152 | 4.361 |
| epsilon_w=0.98 | 2.157e-10 | 0.1976 | 1.1429 | 0.3710 | 0.3140 | 0.2805 | 4.462 |
| epsilon_w=0.997 | 1.206e-09 | 0.6898 | 0.6931 | 0.0614 | -0.0932 | 0.1789 | 4.935 |
| eta=6.20 | 3.677e-10 | 0.7834 | 0.5811 | 0.0107 | -0.1468 | 0.1510 | 5.141 |
| eta=6.35 | 1.593e-10 | 0.8110 | 0.5412 | -0.0019 | -0.1515 | 0.1406 | 5.227 |

The important caveat remains visible: `wind_sink_integral_over_inner = 0` for
these energy-only wind states.

## Wind Jacobian Audit

The wind-specific directional audit compares `J d` against central finite
differences of the square residual.

Summary:

| state | median directional error | max error at best step | active fraction |
|---|---:|---:|---:|
| no wind | 2.226e-07 | 2.366e-07 | 0 |
| epsilon_w=0.98 | 8.035e-05 | 2.373e-04 | 1 |
| epsilon_w=0.997 | 6.083e-05 | 1.768e-04 | 1 |
| eta=6.20 | 5.062e-05 | 1.344e-04 | 1 |
| eta=6.35 | 4.635e-05 | 1.182e-04 | 1 |

Localized `logT` perturbations near the wind region show interval-energy
relative errors up to about `5.6e-4`. This is acceptable for now and does not
look like a sign error.

## Mesh Validation

All requested N checks passed:

| state | N=768 | N=896 | N=1024 | diagnostic stability |
|---|---:|---:|---:|---|
| epsilon_w=0.98 | 2.985e-09 | 2.157e-10 | 6.201e-09 | stable |
| eta=6.20 | 3.184e-09 | 3.677e-10 | 1.692e-10 | stable |
| eta=6.35 | 2.090e-10 | 1.593e-10 | 5.586e-10 | stable |

The physical diagnostics are essentially unchanged across N:

- `epsilon_w=0.98`: `Qwind/Qvisc ~= 0.1976`, `Lrad/LEdd ~= 1.1429`,
  `f_adv_global ~= 0.3710`, `Rson ~= 4.462 rg`.
- `eta=6.20`: `Qwind/Qvisc ~= 0.7833-0.7834`,
  `Lrad/LEdd ~= 0.5811`, `f_adv_global ~= 0.0107`,
  `Rson ~= 5.141 rg`.
- `eta=6.35`: `Qwind/Qvisc ~= 0.8110`, `Lrad/LEdd ~= 0.5412`,
  `f_adv_global ~= -0.0019`, `Rson ~= 5.227 rg`.

This is the strongest numerical support so far for the energy-only high-wind
branch.

## Closure Sensitivity

A frozen-profile audit scanned:

- target `Qwind/Qvisc = 0.2, 0.5, 0.8`
- `chi_edd = 0.995, 0.990, 0.985`
- `wind_activation_width_fraction = 0.001, 0.0025, 0.005, 0.010`

Findings:

- The active interval fraction is already `1` for these wind states.
- Width changes barely affect the frozen target estimates.
- Changing `chi_edd` shifts the epsilon needed to hit a target but does not
  qualitatively change the frozen wind region.

However, a full repolish showed that frozen-profile target matching is only a
guide. For `chi=0.985`, `width=0.01`:

| epsilon_w | repolished Qwind/Qvisc | residual |
|---:|---:|---:|
| 0.950 | 0.124 | 5.070e-09 |
| 0.964 | 0.169 | 2.414e-10 |
| 0.968 | 0.188 | 6.690e-09 |
| 0.981325 | 0.294 | 1.713e-09 |

The fixed-strength closure scan therefore needs a scalar epsilon controller,
not just frozen-profile estimates.

A high-wind closure repolish attempt for `target Qwind/Qvisc~0.8`,
`chi=0.995`, `width=0.001`, `epsilon=0.985838` was interrupted after several
minutes inside Jacobian construction. This should be retried with staged
continuation, not a one-shot repolish.

## Mass-Coupled Wind Budget

The post-processing mass-coupled diagnostic assumes:

```text
E_w = GM/(2R)
l_w = l
Mdot_wind_prime = 2 pi R^2 Qwind / E_w
```

It shows the high-wind energy-only branch would imply very large mass loss:

| state | Qwind/Qvisc | implied Mwind/Mdot_inner | required Mdot_outer/Mdot_inner |
|---|---:|---:|---:|
| epsilon_w=0.98 | 0.198 | 0.864 | 1.064 |
| epsilon_w=0.997 | 0.690 | 2.967 | 3.167 |
| eta=6.20 | 0.783 | 3.386 | 3.586 |
| eta=6.35 | 0.811 | 3.523 | 3.723 |

This is the main physical caveat. If the energy loss is interpreted as an
escape-energy wind, the stream supply/reservoir boundary must change
substantially. The current model has `Mdot_outer/Mdot_inner = 0.2`, so a
mass-loaded wind with the same energy sink is not a small perturbation.

## Eta Continuation

Manual small-step eta continuation from `eta=6.35` succeeded:

| eta | epsilon_w | residual | Qwind/Qvisc | Lrad/LEdd | f_adv_global | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|
| 6.375 | 0.998296 | 1.237e-09 | 0.8151 | 0.5348 | -0.00362 | 5.241 |
| 6.400 | 0.998338 | 4.870e-10 | 0.8190 | 0.5285 | -0.00520 | 5.255 |
| 6.425 | 0.998379 | 2.473e-10 | 0.8228 | 0.5222 | -0.00666 | 5.269 |

The initial residual grows quickly:

```text
6.375: initial ~0.071
6.400: initial ~0.103
6.425: initial ~0.173
```

So adaptive eta is viable, but a real controller should shrink steps when
`initial_full > 0.03` and should add a tangent predictor before pushing much
farther.

## Current Interpretation

The energy-only wind branch is numerically robust through at least
`Qwind/Qvisc ~= 0.82` and survives N768/N896/N1024 validation. This is now a
well-supported wind-cooled steady solution family.

But because the implied mass-loaded wind would require
`Mdot_outer/Mdot_inner > 3` at high eta, the present branch is not yet a
physical stream-fed mass-loaded wind branch. The next physical model step must
couple wind energy, mass, angular momentum, and the outer reservoir/stream
supply consistently.

## Recommended Next Step

Do not spend the next sprint chasing `epsilon_w -> 1`.

Recommended order:

1. Implement a scalar fixed-strength controller for closure scans:
   target `Qwind/Qvisc`, solve for epsilon after each repolish.
2. Retry high-wind closure sensitivity with staged epsilon/chi/width
   continuation.
3. Design the mass-loaded wind formulation with a reservoir boundary that can
   supply the implied wind mass loss.
4. Only then wire mass/AM-coupled wind into the BVP.
5. After that, return to equilibrium-map and QPE cycle construction.
