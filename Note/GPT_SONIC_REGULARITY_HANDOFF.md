# GPT Prompt: Sonic Regularity Bottleneck in IMBH/QPE Transonic Minidisk Solver

Please review this GitHub repository:

```text
https://github.com/huanyang07/IMBH
```

Focus on the latest `main` branch after the commit that adds sonic-refinement diagnostics and the sonic micro-domain experiment.

## Scientific / Numerical Goal

We are trying to compute the radially consistent slim/wind minidisk branch for an IMBH/QPE model and ultimately use it to generate a physical S-curve / one-zone-cycle model. The current target root is a fixed-Mdot transonic solution near:

```text
Mdot/Mdot_Edd = 0.90277664
R_out / R_g   = 6500 for the inner-domain match
R_far / R_g   = 1e5 for the outer extension
R_son / R_g   ~ 5.8
lambda0       ~ 3.675-3.676
int_adv       ~ 0.07-0.08
```

The important question now is not high-Mdot continuation yet. It is:

```text
How do we obtain a mesh-robust sonic regular solution when refining the inner grid near R_son?
```

## Current Status

The two-domain pressure-supported outer extension appears stable. Outer-grid and far-radius variations are no longer the dominant bottleneck.

The remaining bottleneck is the sonic/inner-grid treatment. A good local root exists near N_inner=65, but staged inner refinement breaks sonic regularity. Residual localization showed the failure starts near the sonic endpoint and later appears as regular inner radial residual growth.

## Files to Read First

Please read these in this order:

```text
Note/CODEX_TWO_DOMAIN_INNER_REFINEMENT_NEXT_STEPS.md
outputs/tables/transonic_two_domain_inner_residual_profile.md
outputs/tables/transonic_two_domain_sonic_scaling.md
outputs/tables/transonic_two_domain_sonic_buffer_refinement.md
outputs/tables/transonic_two_domain_dynamic_sonic_patch.md
outputs/tables/transonic_two_domain_sonic_microdomain_scan.md
outputs/tables/transonic_two_domain_sonic_microdomain.md
scripts/run_transonic_two_domain_sonic_refinement_sprint.py
scripts/run_transonic_two_domain_dynamic_sonic_patch.py
scripts/run_transonic_two_domain_sonic_microdomain.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
```

Useful checkpoint directories:

```text
outputs/checkpoints/transonic_two_domain_sonic_refinement_sprint/
outputs/checkpoints/transonic_two_domain_dynamic_sonic_patch/
outputs/checkpoints/transonic_two_domain_sonic_microdomain/
outputs/checkpoints/transonic_two_domain_staged_regular_refinement/
```

## Latest Experiments and Results

### 1. Fixed sonic buffer

We inserted a sonic buffer from `R_son` to `R_son + Delta_s` and imposed a fixed slope from the source solution. This produced a good Nreg64 root:

```text
Delta_s = 0.02
Nreg64 physical residual ~ 8.9e-7
```

But staged regular-grid refinement failed:

```text
N80  ~ 8.0e-5
N96  ~ 1.0e-3
N112 ~ 5.9e-3
N128 ~ 8.0e-3
```

Conclusion: the fixed-slope buffer gives a good local N64 root but does not mesh-converge.

### 2. Dynamic first-order sonic patch

We replaced the fixed slope by inferring

```text
g_s = (y_buffer - y_s) / Delta_s
```

and constraining it with the scaled local sonic differential equation:

```text
A_s g_s + c_s = 0
```

while auditing `D,C1,C2,K`.

This stabilized the N64 -> N128 regular-resolution ladder:

```text
Nreg64  physical ~ 1.35e-5, dominant C2
Nreg80  physical ~ 4.81e-5, dominant patch
Nreg96  physical ~ 1.52e-4, dominant patch
Nreg112 physical ~ 4.17e-4, dominant patch
Nreg128 physical ~ 5.65e-4, dominant patch
```

Conclusion: dynamic patch prevents the basin jump, but the first-order finite patch truncation error becomes the residual floor.

### 3. Sonic micro-domain experiment

We tried replacing the one-step patch with small ODE-collocation intervals from `R_son` to `R_son + Delta_s`, using the square sonic pair `D,K` for the solve while auditing `C1,C2`.

N64 scans:

```text
dynamic patch previous result: 1.35e-5, dominant C2
micro-domain N_micro=4:      4.27e-4, dominant micro_R
micro-domain N_micro=2:      3.86e-4, dominant micro_R
micro-domain N_micro=1:      2.63e-4, dominant C2
Delta_s=0.05 direct:         1.69e-3, dominant micro_R
integrated_dx N_micro=2:     3.86e-4, dominant micro_R
```

Conclusion: naive midpoint micro-collocation near the singular sonic endpoint is diagnostically useful but worse than the dynamic first-order patch. The issue is not simply differential vs integrated residual algebra, nor just the number of micro intervals.

## What We Need Advice On

Please propose a concrete next numerical strategy to get a mesh-robust sonic regular root.

In particular:

1. Should we implement an analytic sonic Taylor / Frobenius-style regularity patch?
2. If yes, what should the unknowns and residual equations be?
   - `y_s = (log u, log T)`?
   - sonic derivative `g_s`?
   - second derivative `h_s`?
   - eigenparameters `logR_son`, `lambda0`?
3. How should the singular local system

```text
A(y_s, R_son, lambda0) g_s + c(y_s, R_son, lambda0) = 0
```

be regularized at the sonic point?
4. Should we solve for the null-space derivative using differentiated regularity conditions, e.g. derivative of determinant/compatibility along the solution?
5. Which sonic compatibility rows should be solved vs only audited?
   - Current square choices use `D,K`.
   - `C1,C2` are sensitive and sometimes become unused residuals.
6. Would a bordered Newton / pseudo-arclength formulation help here, or should we first fix the local sonic patch?
7. How should we validate success?
   - Suggested gates: physical residual <= few e-6, stable `R_son/lambda0/int_adv`, and N64/N96/N128 agreement.

Please be concrete: suggest equations, residual vector structure, unknown packing, scaling, and staged continuation steps. If possible, identify what to change first in:

```text
scripts/run_transonic_two_domain_dynamic_sonic_patch.py
scripts/run_transonic_two_domain_sonic_microdomain.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
```

The main question is:

```text
What is the best next implementation to replace the first-order dynamic patch with a truly sonic-regular local expansion so the two-domain root becomes mesh-converged?
```
