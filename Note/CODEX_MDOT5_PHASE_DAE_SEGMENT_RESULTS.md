# Mdot=5 Phase-Space DAE Segment Results

## Target

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact stream source
- local-Mdot wind formulation
- `eta_E = 98.125`
- `N = 164`
- seed checkpoint:
  `outputs/checkpoints/m5_eta_dae_lobatto_tangent_homotopy_gs1000_lt0_98p125_N164/stage_00_etaE_98p125_N164.npz`

## Implementation

Added an opt-in phase-space DAE transition segment to
`scripts/run_mdot5_local_mdot_eta_continuation.py`.

Controls:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PHASE_DAE_SEGMENT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PHASE_DAE_SEGMENT_MODE`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PHASE_DAE_SEGMENT_INTERVALS`
- phase residual weights for radial, energy, F-prime, kinematic, norm, and endpoint rows.

State:

```text
z = (logu, logT, F, logR)
p = dz/ds
F = Mdot / Mdot_inner
```

Residuals:

```text
A_R(z) p_state + c_R(z) p_R = 0
A_E(z) p_state + c_E(z) p_R = 0
p_F - Fprime_target p_R = 0
z_R - z_L - 0.5 ds (p_L + p_R) = 0
||p|| - 1 = 0
```

The segment uses independent left/right tangents per interval endpoint. This
was necessary: a shared tangent at common nodes accidentally reintroduced a
derivative-continuity constraint and reproduced the Lobatto incompatibility.

State-mode phase solves now keep logR knots inside fixed seed Voronoi cells so
the local Mdot profile remains strictly increasing.

The normal profile JSON now includes:

- `global_flux_phase_dae_segment_initial_profile`
- `global_flux_phase_dae_segment_final_profile`

These contain per-point `R_rg`, `p_R`, reconstructed `|dz/dlogR|`, `cond_A`,
radial residual, and energy residual.

## Run Summary

| output stem | mode | intervals | nfev | radial | energy | F-prime | kinematic | p_R behavior |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `m5_eta_phase_dae_segment_eval_intervalp_from_dae_gs1000_98p125_N164` | evaluate | 26 | 1 | 3.85e-5 | 4.93e-3 | 6.38e-5 | 2.79e1 | positive |
| `m5_eta_phase_dae_segment_tangent_intervalp_k4_from_dae_gs1000_98p125_N164` | tangent | 4 | 13 | 5.12e-6 | 1.87e-6 | 2.67e-11 | 1.44e-2 | positive |
| `m5_eta_phase_dae_segment_state_intervalp_k4_profile_98p125_N164` | state | 4 | 80 | 2.39e-5 | 7.93e-5 | 7.91e-6 | 9.42e-3 | positive |
| `m5_eta_phase_dae_segment_tangent_intervalp_k8_from_dae_gs1000_98p125_N164` | tangent | 8 | 13 | 5.43e-6 | 1.87e-6 | 2.67e-11 | 3.51e-2 | positive |
| `m5_eta_phase_dae_segment_state_intervalp_k8_monotone_98p125_N164` | state | 8 | 80 | 4.46e-3 | 3.64e-3 | 2.81e-5 | 8.44e-2 | positive |
| `m5_eta_phase_dae_segment_tangent_intervalp_k4_kin10_98p125_N164` | tangent | 4 | 60 | 5.11e-4 | 3.66e-4 | 2.82e-6 | 1.26e-2 | sign change |

## Interpretation

The phase-space formulation is doing the right thing locally. In the 4-interval
and 8-interval tangent solves, the DAE radial and energy residuals both reach
about `1e-6`, and the F-prime row becomes essentially exact. This confirms that
the stiff transition layer has a regular phase-space DAE branch.

The remaining issue is the kinematic integration/state matching. The fixed-state
tangent solve cannot reduce the trapezoid kinematic defect below about
`1e-2` to `3e-2`. Releasing state nodes helps for the 4-interval segment, but it
is slow and does not scale cleanly to 8 intervals with the current dense
finite-difference Jacobian.

Increasing the kinematic weight is not the solution. A `kinematic_weight=10`
test barely reduced the kinematic residual and degraded radial/energy residuals;
it also introduced a local `p_R` sign change. This is a useful rejection test.

The full 26-interval tangent solve was interrupted because the dense finite-
difference Jacobian made it too expensive. This is now a numerical
infrastructure bottleneck, not evidence against the phase-space formulation.

## Current Answer

We have not yet recovered a production-ready phase-space transition segment,
but we have recovered a locally consistent DAE branch that ordinary logR
Lobatto could not represent. This supports GPT's diagnosis: stop tuning
Lobatto weights and move to a real phase-space/arclength segment.

## Recommended Next Step

Implement a sparse/local Jacobian for the phase DAE residual and then grow the
segment by staged windows:

1. Certify 4-interval state mode to `kinematic < 1e-3` without losing
   `radial, energy < 1e-4`.
2. Grow to 8 intervals using the 4-interval solution as a seed.
3. Grow to 16 and then 26 intervals.
4. Only after the phase segment satisfies DAE and kinematic rows should it be
   coupled back to the outer/global Lobatto regions and used for eta
   continuation.

Do not resume eta continuation yet.

## F-Augmented Direct-Physics Update

The first phase implementation still inherited an interval-tabulated
`dMdot/dlnR` inside the radial/energy matrix while also solving an independent
mass-flux tangent. It also used trapezoidal kinematics. Both choices have now
been replaced.

The production phase state is

```text
z = (logu, logT, F, logR)
p = dz/ds
F = Mdot / Mdot_inner
```

The corrected formulation now:

- evaluates the local state at the point value of `F`;
- exposes the `p_F/F` derivative contribution explicitly;
- uses the direct multiplied physical residual
  `p_R * R(logR,z,dz/dlogR)` as the production radial/energy equation;
- retains the augmented matrix as a conditioning and regular-limit audit;
- uses one shared tangent at every internal phase node;
- uses Hermite-Simpson kinematics in intrinsic coordinate `s`, including an
  independent midpoint tangent constrained by the same DAE;
- adds an arclength-mesh gauge;
- uses a sparse block dependency pattern;
- saves `z`, node tangents, midpoint tangents, and `ds` in checkpoints;
- separates the immutable exterior matching state from the continuation seed;
- extends a saved phase trajectory with a local DAE predictor-corrector seed.

The direct physical residual exposed and rejected one false intermediate
solution: its linearized energy row was `4.8e-6`, but its true physical energy
residual was `1.7e-3`.

## Updated Run Summary

| checkpoint | intervals | radial | physical energy | F-prime | kinematic | endpoint mismatch | p_R behavior | result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `m5_eta_phase_dae_simpson_state_halo8_k4_pass2_balanced_98p125_N164` | 4 | `1.14e-6` | `1.40e-5` | `8.26e-8` | `8.33e-5` | `1.20e-2` | positive | local only |
| `m5_eta_phase_dae_simpson_centered_k12_final_polish_98p125_N164` | 12 | `8.06e-5` | `9.43e-5` | `1.03e-6` | `4.40e-4` | `9.28e-4` | positive | exploratory pass |
| `m5_eta_phase_dae_simpson_k13_fromk12_98p125_N164` | 13 | `4.20e-5` | `9.82e-5` | `1.69e-6` | `4.15e-4` | `2.57e-4` | positive | exploratory pass |
| `m5_eta_phase_dae_simpson_k14_fromk13_98p125_N164` | 14 | `8.10e-5` | `1.52e-4` | `2.24e-6` | `8.80e-4` | `1.05e-3` | two sign changes | fail |

The accepted 13-interval segment spans approximately `194.58--223.36 rg`.
Its reconstructed derivative norm remains below `20.5`, and `p_R` remains
positive (`0.0488--0.835`). This is the first phase-space segment that satisfies
the physical DAE, conservative mass tangent, intrinsic kinematics, and exterior
state matching simultaneously. The explicit certification audit
`m5_eta_phase_dae_simpson_k13_certified_98p125_N164` reports
`accepted_exploratory=true` and `accepted_preferred=false`.

The fourteenth interval reaches a new feature around `R~220.7 rg`. The best
solution develops two `p_R` sign changes, with `p_R_min~-0.046`; its peak radial
and energy residuals lie near the same region. A stronger energy/interface
polish worsens the coupled residuals, so this point is not accepted. It may be a
real phase-space fold or the next source-core discretization issue.

## Updated Conclusion

The original `lnR` Lobatto interface defect has been resolved over a finite,
matched phase-space segment. The accepted endpoint is 13 intervals, not the
full 26-interval source block. Direct `12 -> 16` continuation is too aggressive;
one-element continuation succeeds through interval 13 and then encounters the
new `p_R`-turning feature at interval 14.

The phase solution is still stored as auxiliary checkpoint data and has not yet
replaced the corresponding rows in the global production BVP. Therefore this
does not yet certify the complete `eta_E=98.125` global wind solution and does
not authorize lower-eta continuation.

Next:

1. Promote the accepted 13-interval phase segment into the global production
   residual, replacing the old `lnR` Lobatto rows only on those intervals.
2. Match state, mass flux, and energy flux at both phase/global interfaces; do
   not match `d/dlnR` derivatives.
3. Globally polish at fixed `eta_E=98.125` and require the phase gates plus the
   ordinary global, source, sonic, and outer-boundary gates.
4. Audit the interval-14 `p_R` turn under phase-mesh refinement before extending
   farther into the source core.
5. Resume eta continuation only after the unified global/phase formulation is
   strict.

Verification after the update:

- `py_compile`: passed;
- `git diff --check`: passed;
- full test suite: `160 passed, 2 subtests passed`.
