# GPT Prompt: Taylor/L'Hopital Sonic Patch Failed; Need Next Strategy

Please review this GitHub repository:

```text
https://github.com/huanyang07/IMBH
```

Use the latest `main` branch after the commit that adds Taylor/L'Hopital sonic-regularity diagnostics.

## Context

We are trying to compute a radially consistent transonic slim/wind minidisk solution for an IMBH/QPE model. The current fixed-Mdot target is:

```text
Mdot/Mdot_Edd ~= 0.90277664
R_match ~= 6500 rg
R_far ~= 1e5 rg
R_son ~= 5.7--5.8 rg
lambda0 ~= 3.676
int_adv ~= 0.07--0.08
```

The two-domain pressure-supported outer extension is no longer the main blocker. The remaining blocker is the local sonic/inner-grid regularity treatment.

## Files to Read First

Please read these files:

```text
Note/GPT_SONIC_REGULARITY_HANDOFF.md
Note/CODEX_SONIC_TAYLOR_REGULARITY_PATCH_PLAN.md
outputs/tables/transonic_two_domain_sonic_taylor_summary.md
outputs/tables/transonic_sonic_derivative_roots.md
outputs/tables/transonic_sonic_lhopital_audit.md
outputs/tables/transonic_two_domain_sonic_taylor_patch.md
scripts/run_transonic_sonic_derivative_root_scan.py
scripts/run_transonic_two_domain_sonic_taylor_patch.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
tests/test_transonic_local.py
```

Useful older comparison outputs:

```text
outputs/tables/transonic_two_domain_dynamic_sonic_patch.md
outputs/tables/transonic_two_domain_sonic_microdomain_scan.md
outputs/tables/transonic_two_domain_sonic_microdomain.md
outputs/tables/transonic_two_domain_sonic_buffer_refinement.md
```

## What We Tried

### 1. Dynamic first-order patch

This remains the best numerical baseline:

```text
Nreg64  physical ~ 1.35e-5
Nreg80  physical ~ 4.81e-5
Nreg96  physical ~ 1.52e-4
Nreg112 physical ~ 4.17e-4
Nreg128 physical ~ 5.65e-4
```

It stabilizes refinement but becomes patch-limited.

### 2. Naive sonic micro-domain

Small midpoint-collocation intervals near the sonic point were worse:

```text
N_micro=4: physical ~ 4.27e-4
N_micro=2: physical ~ 3.86e-4
N_micro=1: physical ~ 2.63e-4
```

### 3. Taylor/L'Hopital patch

We implemented local helper functions:

```text
local_unscaled_residual
local_scaled_residual
sonic_null_vectors
sonic_unscaled_null_vectors
sonic_directional_B
sonic_frozen_scaled_directional_B
sonic_unscaled_directional_B
sonic_lhopital_residual_form
```

Then we scanned sonic derivative roots for:

```text
scaled:        derivative of fully scaled residual
frozen_scaled: derivative of raw residual divided by sonic-point scales
raw:           derivative of unscaled residual
```

All three forms found essentially the same two L'Hopital roots:

```text
root A: g ~= (-183.5, 73.6)
root B: g ~= (161.3, -96.1)
```

But the accepted solution branch has much smaller slopes:

```text
source first-interval slope ~= (18.3, 17.8)
dynamic-patch slope         ~= (19.7, -27.9)
```

So the L'Hopital condition, as implemented, selects derivative branches far from the branch that gives the good dynamic-patch root.

### 4. Taylor patch solve results

The Taylor patch did not beat the dynamic patch:

```text
Taylor1, buffer slope:
    physical ~ 1.72e-3, dominant local_F

Taylor2, buffer slope:
    physical ~ 1.59e-3, dominant L

Taylor2, L'Hopital-root slope:
    physical ~ 2.01e-4, dominant C2

Taylor2, frozen-scale L'Hopital root:
    physical ~ 2.01e-4, dominant C2

Taylor2, soft C1/C2/K compatibility instead of L'Hopital:
    weight 0.2: active residual ~ 2.96e-4
    weight 0.05: active residual ~ 2.88e-4
    weight 0: active residual ~ 2.87e-4
```

N80 refinement from the best Taylor2 branch failed:

```text
Nreg80 physical ~ 6.26e-3, dominant regular_R
```

Thus Taylor/L'Hopital as currently formulated is not mesh-stable and not science-grade.

## What We Need Advice On

Please propose the next numerical strategy. We need something better than:

```text
dynamic first-order patch: stable but first-order/patch-limited
Taylor/L'Hopital patch: selects far derivative branch and fails refinement
micro-domain midpoint collocation: too singular-sensitive
```

Specific questions:

1. Is the L'Hopital derivation being applied to the wrong local equations?
   - Should the derivative condition be based on determinant/compatibility derivatives instead of `l^T dF/dx`?
   - Should the derivative use an adjugate compatibility condition rather than the SVD left-null vector?

2. Should the sonic derivative be selected by differentiating the sonic constraints:

```text
D(x,y,lambda)=0
K(x,y,lambda)=0
```

along the solution, instead of using `l^T B(g)=0`?

3. Is there a physically motivated branch criterion we are missing?
   - e.g. entropy monotonicity
   - inward/outward regularity
   - sign of heat/advection terms
   - matching to the subsonic outer branch

4. Should the production method remain closer to the dynamic patch, but be upgraded by:
   - shrinking Delta_s with continuation,
   - Richardson/extrapolating Delta_s -> 0,
   - using a higher-order Hermite patch anchored to ODE slopes away from the sonic point,
   - or solving a local boundary-value problem from a nonsingular offset point?

5. Would a collocation method that excludes the exact sonic endpoint and imposes sonic constraints only as boundary conditions be more robust?

6. What concrete implementation should Codex attempt next?

Please give equations, residual vector structure, unknowns, scaling, branch-selection rules, and a staged validation plan. The immediate success target is:

```text
fixed Mdot/Edd ~= 0.90277664
Nreg64 residual <= few e-6
Nreg80/Nreg96 stable
Rson/lambda0/int_adv stable
no hidden C1/C2/K incompatibility
```

We should not resume high-Mdot continuation until this fixed-Mdot sonic regularity problem is solved.
