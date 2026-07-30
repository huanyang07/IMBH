# WP10c9d6c7c2b5b — Uniform shear family-transfer audit

- Classification: `raw_local_family_leakage_projector_rotation_sensitive_revised_transfer_observable_manifest_authorized`
- c2b4 and c2b5a classifications: preserved.
- Operator changed: `False`.
- Embedded/nonlinear/fixed-Q/reduced evolution: not run.

## Projector audit

| Level | polynomial algebra | eig/poly difference | minimum gap |
|---|---:|---:|---:|
| N98 | 2.357e-10 | 3.436e-12 | 2.467e-03 |
| N196 | 2.650e-10 | 1.731e-12 | 2.454e-03 |
| N392 | 2.977e-10 | 1.973e-12 | 2.451e-03 |

## Exact shear transfer

| Level | dominant opposite-family block | absolute fraction | partition defect | power defect |
|---|---|---:|---:|---:|
| N98 | candidate_conservative_transport | 0.3892 | 5.168e-16 | 5.472e-14 |
| N196 | candidate_conservative_transport | 0.3887 | 7.471e-16 | 1.251e-13 |
| N392 | candidate_conservative_transport | 0.3962 | 7.125e-16 | 2.149e-13 |

The transfer tensor is the exact transfer of the implemented frozen DAE. A large block contribution is not automatically a numerical defect.

## Shear-leakage projector comparison

| Projector definition | N98 | N196 | N392 | order |
|---|---:|---:|---:|---:|
| local_eigenvector | 1.773526e+02 | 1.820124e+02 | 1.871025e+02 | -0.1274 |
| local_polynomial | 1.773526e+02 | 1.820124e+02 | 1.871025e+02 | -0.1274 |
| common_N392_field | 1.770794e+02 | 1.820414e+02 | 1.871025e+02 | -0.0286 |
| frozen_receiving_band_midpoint_diagnostic | 3.169874e+02 | 3.493253e+02 | 3.583755e+02 | 1.8372 |

## Independent continuum action

| Profile | reference difference | unsolved order | solved order | minimum block order |
|---|---:|---:|---:|---:|
| acoustic | 1.903e-07 | 2.9048 | 1.9912 | 2.6838 |
| shear | 4.661e-07 | 3.0986 | 1.9768 | 2.6756 |
| mixed_shear_acoustic | 5.110e-07 | 3.1382 | 1.9922 | 2.7042 |

## Decision

Equivalent local eigensolver and polynomial projectors agree, the common N392 local-projector field is robust for shear leakage, all independent continuum-action truncations contract, and no stable noncontracting transfer block is selected. Raw local opposite-family stored energy remains nonconvergent while the deliberately frozen-subspace diagnostic converges. The raw quantity therefore mixes transfer with spatial projector rotation and is non-certifying by itself. A definitions-only uniform manifest may retain total positive energy, target arrival, the exact covariant transfer balance, and explicitly projector-qualified quantities.

Authorized next: `WP10c9d6c7c2b6a_revised_uniform_arrival_contract_manifest`.

No historical rejection is relabeled. Embedded discrimination, operator/interface redesign, nonlinear propagation, fixed-Q experiments, reduced evolution, and N1024 remain blocked.
