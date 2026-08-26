# Seven-field physical closure local structural audit

Classification: `seven_field_physical_closure_entropy_failed`.

The first binding nonlinear entropy test fails. No eigenvalue campaign, spatial discretization, nonlinear root, or trajectory was executed.

## Result

For an exact Godunov closure, the entropy-flux one-form `w(q)^T dF(q)` must be closed. At the prospectively selected primary 20 ms cell 36 witness, independent sixth-order-centered derivative ladders give:

- `2 h`: curl defect `1.3700857565797004e+00`, maximum pair `[np.int64(3), np.int64(4)]`.
- `1 h`: curl defect `1.3683589179402538e+00`, maximum pair `[np.int64(3), np.int64(4)]`.
- `0.5 h`: curl defect `1.4022456388346460e+00`, maximum pair `[np.int64(3), np.int64(4)]`.

The frozen symmetry tolerance is `1.0e-10`. The minimum observed defect is `1.3683589179402538e+00`. The dominant, step-stable curl component couples log temperature to specific shear stress.

## Interpretation

The positive state-local shear reservoir and alpha signal calibration close algebraically, but inserting the resulting state-dependent modulus into a `D chi` balance does not produce the nonlinear thermodynamic terms required for a common entropy potential. Post-hoc matrix symmetrization is forbidden by the Stage-2 contract, so the candidate is rejected fail-fast.

This does not reject the Stage-1 symmetric local normal form, finite-inertia height dynamics, or a generalized Israel--Stewart/Maxwell--Cattaneo model. It rejects this particular conservative Godunov realization with `R_pi=D chi` and the proposed state-dependent reservoir.

## Decision

Authorized next: `definitions_only_WP10c9d6c7c3b5c4f25fized_generalized_Maxwell_Cattaneo_architecture_manifest` only. The next definitions package must replace the incompatible stress coordinate/closure with a full nonlinear transient model (or a conjugate shear-strain formulation), freeze exact causality and strong-hyperbolicity gates, and retain the stopped-trajectory boundary. Complete-cycle execution remains unauthorized.
