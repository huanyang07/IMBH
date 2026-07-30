# WP10c9d6c7c2b4 — Uniform positive arrival-energy validation

- Classification:
  `one_way_uniform_arrival_energy_validation_failed_embedded_discrimination_blocked`
- Passed: `False`
- Operator changed: `False`
- Propagation executed: `True`, uniform N98/N196/N392 only.
- Embedded, nonlinear, fixed-Q, and reduced evolution were not run.

## Binding result

The replacement fixed-band energy is algebraically sound, but the complete
frozen accuracy contract fails:

```text
method / exact ledger                         PASS
Tier-I state and 13 physical exports          PASS
positive energy / family partition            PASS
quadratic amplitude and sign symmetry         PASS
zero null                                     PASS
all positive arrival-energy cases             FAIL
```

All twelve acoustic, shear, and mixed sign/amplitude cases fail at least one
prospectively frozen arrival-history or scalar gate. Therefore embedded
arrival-energy discrimination is not authorized.

## What passes

### Total time-averaged arrival

For

\[
\mathcal A
=
\frac{\int_{\mathcal W}E_{[6,49]}(t)\,dt}
{|\mathcal W|E_{\rm initial,[52,95]}},
\]

the representative total values are:

| Family | N98 | N196 | N392 | order | relative fine difference | pass |
|---|---:|---:|---:|---:|---:|:---:|
| acoustic | 1007.4017 | 1152.2871 | 1190.4623 | 1.924 | 0.03207 | yes |
| shear | 477.7341 | 526.4439 | 547.3101 | 1.223 | 0.03812 | yes |
| mixed shear-acoustic | 1068.2429 | 1056.3977 | 1053.1974 | 1.888 | 0.00300 | yes |

The corresponding target-family integrated arrivals also pass for all three
families. The very large values are allowed by the frozen definition:
background, descriptor, relaxation, geometry, height, and lower-source work
can amplify the perturbation energy. No upper bound of one was assumed.

### Exact semidiscrete energy

The positive stored-energy history agrees with the exact c2b2
descriptor-dual stored energy to

```text
1.37e-15.
```

Across all three levels:

```text
maximum block-power closure                 4.42e-13
maximum shared-face closure                 6.50e-15
maximum time-integrated action defect       1.08e-11
```

The five-family energy partition closes to `4.45e-15`. Quadratic
amplitude/sign scaling defects and the exact-zero control are both zero.

This confirms that the failure is not caused by negative energy, an
eigenvector normalization, a broken descriptor ledger, or a sign/amplitude
implementation error.

## Binding failures

### 1. Acoustic peak narrowly misses the scalar gate

The normalized acoustic peak is

```text
3711.1946, 4682.3296, 4931.1569
```

with order `1.965`, but its N196/N392 relative difference is

```text
0.0504602 > 0.05.
```

The time-averaged total and target acoustic arrivals pass. This is a narrow
strict-threshold failure, but it remains binding and is not rounded into a
pass.

### 2. Shear leakage is not in an asymptotic scalar regime

Opposite-family shear leakage is

```text
177.3526, 182.0124, 187.1025.
```

The two differences grow from `4.6599` to `5.0901`, giving

```text
order = -0.1274.
```

Its history has:

```text
RMS order                    0.6291 < 0.75
maximum order                0.6736 < 0.75
refinement-error cosine      0.7091 < 0.90.
```

This is the only failure that is not explained merely by the absolute
initial-energy scale of the history contract. It requires a family-transfer
and projector-rotation audit before any new interface experiment.

### 3. The amplified histories fail the frozen absolute scale

Every total, target, and leakage history is already normalized by initial
source energy. The frozen maximum fine-difference threshold is `0.05` in
that scale. The observed absolute N196/N392 maxima are:

| Family | total | target | leakage |
|---|---:|---:|---:|
| acoustic | 249.09 | 33.24 | 215.85 |
| shear | 163.51 | 127.44 | 36.20 |
| mixed shear-acoustic | 54.84 | 29.52 | 27.99 |

Most histories nevertheless have positive convergence orders and stable
directions. Their response-relative fine differences are:

| Family | total | target | leakage |
|---|---:|---:|---:|
| acoustic | 0.0505 | 0.0749 | 0.0481 |
| shear | 0.0492 | 0.0559 | 0.0341 |
| mixed shear-acoustic | 0.00989 | 0.00849 | 0.01315 |

These conditioned values are diagnostic only. They cannot retroactively
replace the prospectively frozen initial-energy normalization or pass c2b4.
They show why the next audit must distinguish a genuinely inaccurate energy
history from a physically amplified observable whose absolute scale is not
suited to a `0.05` error bound.

## Nuisance and horizon treatment

The receiving-band, time-stride, arrival-window, restart, and invariant
subspace uncertainties are combined conservatively without root-sum-square.
Their medium/fine uncertainty ratios remain below `0.1` for the reported
scalar gates.

The c2b3 `1.5×` padding table extends past the frozen experiment end for the
shear and mixed packets. The binding run applies the already inherited c2a3
rule

```text
upper window endpoint = min(declared endpoint, experiment horizon).
```

This clipping is deterministic, was applied before inspecting histories, and
does not move a window using observed data.

## Scientific interpretation

The result does not select an interface or production-operator defect:

- the run is uniform and contains no refinement interface;
- Tier-I state and slow-relevant physical exports pass;
- exact stored energy and the complete semidiscrete ledger pass;
- total and target integrated arrival scalars converge;
- failures are concentrated in the strict amplified-history scale,
  a borderline acoustic peak, and shear family leakage.

The prior classifications remain unchanged:

```text
c2b1 local face-transmission contract rejected
c2b2 local semidiscrete face ratio not certifiable
c2b3 positive arrival contract frozen
c2b4 strict uniform arrival validation rejected
```

No threshold is relaxed and no failed result is relabeled.

## Verification

- Positive-energy helper and canonical tests: `7 passed`.
- Full focused c2a–c2b4 chain: `42 passed`.
- Full repository suite: `1068 passed`, `4 subtests passed`, `2 failed`.
- The two failures are the pre-existing canonical-status vocabulary value
  `PROSPECTIVE MANIFEST ONLY` and tracked-file count `1227 >= 850`.
- No scientific or numerical test failed.
- No embedded or nonlinear trajectory was run.

## Authorized next package

Only:

```text
WP10c9d6c7c2b5_frozen_arrival_energy_failure_audit
```

The audit must change no operator and must:

1. preserve every c2b1–c2b4 value, gate, and classification;
2. separate absolute initial-energy-scale error from
   response-relative/continuum-extrapolated error without using the latter
   to pass c2b4;
3. identify the time and radius of the acoustic peak miss;
4. decompose shear leakage into physical family coupling, spatially varying
   projector rotation, background work, and numerical truncation;
5. compare unsolved DAE truncation, mass-solved rate error, total positive
   energy, and projector-resolved energy;
6. require any claimed shear mechanism to persist on both refinement pairs;
7. decide prospectively whether the arrival observable needs a new
   normalization contract, the shear leakage needs a local audit, or the
   Tier-II route should be abandoned.

Embedded c2c1/c2c2, operator/interface redesign, nonlinear propagation,
fixed-Q experiments, reduced slow-time evolution, and N1024 refinement
remain blocked.
