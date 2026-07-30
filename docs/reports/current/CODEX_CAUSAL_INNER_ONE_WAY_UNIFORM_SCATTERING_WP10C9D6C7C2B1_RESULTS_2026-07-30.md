# WP10c9d6c7c2b1 — One-way uniform scattering validation

- Classification:
  `one_way_uniform_scattering_validation_failed_embedded_discrimination_blocked`
- Production operator changed: `False`
- Embedded, nonlinear, fixed-Q, and reduced evolution were not run.

## Binding result

Method / Tier I / Tier II / amplitude-null gates:

```text
PASS / PASS / FAIL / PASS
```

All N98/N196/N392 self-consistent monolithic tangents pass reconstruction,
descriptor/storage, generator-factorization, excision-causality, shared-flux,
and exact prefix-ledger gates. The maximum centered storage-action defect is
`1.35e-9`, the maximum factorization defect is `2.13e-16`, and no incoming
excision characteristic appears.

All twelve binding sign/amplitude cases pass the Tier-I state and direct
13-export contracts:

```text
state order floor                         1.9219
state component-order floor               1.8710
state refinement-error cosine floor       0.98648
instantaneous export RMS-order floor      1.9264
instantaneous export component floor      1.7784
instantaneous export error cosine floor   0.98324
cumulative export RMS-order floor         1.7102
cumulative export component floor         1.4331
cumulative export error cosine floor      0.99676
```

Linear state/flux scaling, quadratic energy scaling, sign reversal, the zero
state, and the exact frozen N98 packet replay all pass.

## Tier-II transmission

The physical core remains strictly one-way. No reflection coefficient was
defined because the positive-speed characteristic subspace is empty.

| Family | T(N98) | T(N196) | T(N392) | order | fine normalized difference | pass |
|---|---:|---:|---:|---:|---:|:---:|
| acoustic | 397.76824 | 477.21559 | 499.98021 | 1.8032 | 0.04553 | yes |
| shear | 74.71527 | 106.46906 | 116.17751 | 1.7096 | 0.08357 | **no** |
| mixed shear/acoustic | 74.01214 | 81.97288 | 84.83260 | 1.4770 | 0.03371 | yes |

The shear result is the only binding miss: its N196/N392 change exceeds the
frozen `0.05` limit. Its order and error direction pass. Window/time-sampling
stability is `4.04e-8` or better, and all complete ledger residuals close
below `1.37e-16`.

## Energy interpretation

The transmission ratios are not passive probabilities. The
normalization-invariant local symmetrizer exchanges substantial energy with
the variable background, lower sources, and the semidiscrete
transport/descriptor operator. For the N392 shear case:

```text
incident energy                             8.20497
transmitted energy                        953.2328
background-gradient work                -4048.402
other lower-source work                -11785.06
responsive-height work                   1550.954
semidiscrete transport/descriptor work   14919.31
stored-energy change                    1.84e-8
```

The complete balance is explicit and closes, but large cancellations make
`T = E_transmitted/E_incident` a strongly conditioned amplification
diagnostic rather than a simple interface-transmission probability. The
semidiscrete remainder is reported separately and is not mislabeled as
physical dissipation.

This is a uniform calculation with no refinement interface. Therefore the
failure is not evidence for an interface defect and does not select an
interface or production-operator redesign.

## Binding decision and next step

Preserve the c2b1 rejection. Do not run embedded c2c1, nonlinear evolution,
fixed-Q averaging, reduced evolution, or a brute-force higher grid.

The authorized next package is:

```text
WP10c9d6c7c2b2_one_way_uniform_transmission_interpretation_audit
```

It should use the completed tangents and histories to:

1. derive an exact discrete control-volume energy identity from the
   semidiscrete generator and descriptor metric;
2. decompose its symmetric generator action by physical residual block;
3. derive the numerical face-energy flux from the actual reconstructed
   shared-face operator;
4. compare it with the current continuum symmetrizer flux;
5. determine whether the large work amplification and shear fine-pair miss
   arise from the observable definition, continuum/discrete energy mismatch,
   or a genuine uniform transport error;
6. preserve every c2b1 value and threshold without retroactive relabeling.

Only a newly frozen, independently meaningful uniform energy diagnostic may
authorize another uniform propagation or embedded discrimination.
