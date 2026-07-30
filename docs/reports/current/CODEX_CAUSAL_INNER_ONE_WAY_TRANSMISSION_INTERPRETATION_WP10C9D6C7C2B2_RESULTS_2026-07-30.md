# WP10c9d6c7c2b2 — One-way transmission interpretation audit

- Classification:
  `exact_semidiscrete_energy_identity_certified_local_face_transmission_not_certifiable`
- Operator changed: `False`
- The c2b1 rejection is preserved.
- Embedded, nonlinear, fixed-Q, and reduced evolution were not run.

## Binding result

The exact semidiscrete control-volume energy identity is certified. A local
incident/transmitted face-power ratio is not.

For the scaled frozen DAE

```text
D z_t + J z = 0,
G = -D^{-1} J,
E_CV = 1/2 z^T W_CV z,
```

the audit evaluates

```text
dE_CV/dt = z^T W_CV G z
```

and uses the descriptor dual

```text
lambda = D^{-T} W_CV z
```

to resolve every stationary/storage block and every actual shared
conservative face. The maximum defects over N98/N196/N392 are:

```text
stored-generator replay                    0
generator factorization                    2.13e-16
generator-component matrix closure         2.60e-15
block power closure                        4.43e-13
shared-face power closure                  6.51e-15
time-integrated energy/action closure      1.08e-11
```

The larger `2.8e-4` to `1.8e-2` cancellation-relative quadrature numbers are
retained in the canonical summary. They divide a small residual by a nearly
zero final stored-energy change. Normalized by the integrated absolute
generator action, the binding quadrature defect is at most `1.08e-11`.

## Continuum versus exact numerical face ratio

The c2b1 continuum-symmetrizer ratios are:

| Family | N98 | N196 | N392 | order | fine difference | result |
|---|---:|---:|---:|---:|---:|:---:|
| acoustic | 397.768 | 477.216 | 499.980 | 1.803 | 0.04553 | pass |
| shear | 74.715 | 106.469 | 116.178 | 1.710 | 0.08357 | fail |
| mixed | 74.012 | 81.973 | 84.833 | 1.477 | 0.03371 | pass |

Pairing the actual shared-face residual with the exact descriptor dual gives:

| Family | N98 | N196 | N392 | order | fine difference | result |
|---|---:|---:|---:|---:|---:|:---:|
| acoustic | 400.664 | 1281.272 | 1977.442 | 0.339 | 0.35206 | fail |
| shear | 67.363 | 101.841 | 113.576 | 1.555 | 0.10332 | fail |
| mixed | 132.097 | 101.429 | 93.619 | 1.974 | 0.05912 | fail |

These numerical ratios are not a valid replacement contract:

- the acoustic incident face contribution changes sign:
  `-2.2547, -0.8663, +0.5836`;
- the two selected measurement faces account for only
  `0.46%–1.04%` of absolute conservative face action for acoustic/mixed
  packets and `4.11%–6.24%` for shear;
- all three numerical ratios miss the inherited fine-difference gate;
- descriptor inversion spreads the control-volume energy dual beyond the
  two geometric measurement surfaces.

The shear numerical ratio does corroborate the c2b1 trend: its numerical to
continuum ratio approaches one,

```text
0.9016, 0.9565, 0.9776,
```

and its incident/transmitted history shapes agree closely on the fine grids.
However, this is not a binding confirmation of a uniform transport defect
because the proposed face ratio fails its own all-family observability and
localization requirements.

## Why the continuum flux is not the semidiscrete ledger

The local continuum symmetrizer flux does not close the implemented
descriptor/reconstruction energy balance by itself. For the N392 shear
packet, the exact full-horizon block integrals include:

```text
conservative transport                    +26667.943
mapped storage-rate derivative            -15611.362
geometry                                  -11744.295
responsive-height work                     +1550.562
responsive-height storage derivative       -787.618
local stress relaxation                     +307.579
shear principal                             -223.321
height principal                            -119.212
cooling                                      -40.276
net stored-energy change                  +1.84e-8
```

The exact identity closes these large cancellations. Assigning the
descriptor-induced transfer to only the two selected faces is therefore not
an invariant local scattering measurement.

## Scientific decision

This package establishes:

1. the c2b1 tangent and physical Tier-I results remain valid;
2. the exact semidiscrete energy identity is available and block complete;
3. the c2b1 continuum face flux is not the exact semidiscrete face ledger;
4. the descriptor-dual local face ratio is itself not a certifiable
   replacement;
5. neither result selects an interface or production-operator redesign.

The strict c2b1 classification remains:

```text
one_way_uniform_scattering_validation_failed_embedded_discrimination_blocked
```

## Verification

- Focused c2a–c2b2 suite: `44 passed`.
- Full repository suite: `1058 passed`, `4 subtests passed`, `2 failed`.
- The two failures are the pre-existing canonical-status vocabulary entry
  `PROSPECTIVE MANIFEST ONLY` and tracked-file count `1208 >= 850`.
- c2b2 itself uses the accepted status
  `SUPPORTED BUT NOT FULLY CERTIFIED`.
- No scientific or numerical regression failed.

## Authorized next package

Only the following definitions-only package is authorized:

```text
WP10c9d6c7c2b3_definitions_only_semidiscrete_energy_transfer_contract
```

It should:

1. preserve every c2b1/c2b2 value and classification;
2. abandon a ratio of two individually assigned local face powers;
3. define a positive one-way arrival-energy observable from
   symmetrizer energy in fixed upstream and downstream physical bands;
4. use the already frozen travel-time windows and measurement geometry;
5. normalize arrival energy by the positive initial packet energy;
6. retain the exact c2b2 full control-volume block identity as the energy
   ledger;
7. define family arrival/leakage with invariant projectors;
8. freeze spatial bands, time integration, uncertainty envelopes, amplitude
   scaling, and convergence gates before any new propagation;
9. predeclare how a future embedded-minus-uniform arrival-energy difference
   would isolate interface effects;
10. propagate nothing and change no operator.

Only after that manifest is certified may a new uniform arrival-energy
validation be considered. Embedded c2c1, nonlinear evolution, fixed-Q
experiments, reduced slow-time evolution, brute-force refinement, and
operator redesign remain blocked.
