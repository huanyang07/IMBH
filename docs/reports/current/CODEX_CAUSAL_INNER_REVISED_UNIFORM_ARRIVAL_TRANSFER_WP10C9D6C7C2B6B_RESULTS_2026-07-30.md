# WP10c9d6c7c2b6b — Revised uniform arrival/transfer recertification

- Classification: `revised_uniform_arrival_transfer_recertification_failed_embedded_blocked`
- Passed: `False`
- Operator changed: `False`
- Embedded and nonlinear propagation executed: `False`

## Binding result

Tier I / Tier II arrival / covariant transfer / continuum / projector / scaling gates: `True` / `False` / `True` / `True` / `True` / `True`.

## Arrival histories

| Base | Total history | Target history | Role |
|---|:---:|:---:|---|
| acoustic | True | False | calibration |
| shear | False | False | calibration |
| mixed_shear_acoustic | True | True | calibration |
| difference_shear_acoustic | False | False | prospective_heldout |
| shear_weighted_shear_acoustic | True | True | prospective_heldout |

The binding accuracy uses a continuum-response scale. The initial-energy-normalized gain remains reported as the physical observable, but it is not subjected to the rejected absolute 0.05 c2b4 history gate.

The 769- and 513-node histories are independent sixth-order inward collocation evolutions of the complete continuum DAE. Their difference is included additively in every deterministic uncertainty envelope; no root-sum-square combination is used.

## Binding failures

- Acoustic target gain: fine response-relative history / peak differences `0.07307` / `0.07289`.
- Difference held-out total/target gain histories: fine response-relative differences `0.13142` / `0.13484`.
- Shear target gain history/peak: fine response-relative differences `0.05480` / `0.05301`.
- Shear total unit-shape error direction: `0.74453 < 0.90`.

All of these failures are observable under the complete frozen uncertainty envelope. The corresponding continuum-reference ratios are far below `0.10`, so reference uncertainty does not explain them. Orders remain positive; this package therefore selects a local DAE/observable audit, not an operator redesign.

## Covariant transfer

Maximum exact block/source/receiver closure defect: `4.797e-13`.
The finite-time-quadrature endpoint comparison differs by at most `2.270e-03`; this is reported as a quadrature diagnostic, not substituted for the frozen exact transfer-closure gate.

Raw local opposite-family stored energy remains diagnostic and non-certifying, as frozen in b6a. No numerical or interface redesign is selected by this package.

The next audit must freeze these exact failed histories, compare each finite grid directly with the N769 trajectory, and localize the gain and shape errors by time, radius, DAE block, storage, and target-projector action. The passing mixed and shear-weighted profiles remain controls. Threshold changes, profile tuning, N1024, embedded propagation, and nonlinear work remain forbidden.

## Decision

`WP10c9d6c7c2b6c_uniform_recertification_failure_audit`
