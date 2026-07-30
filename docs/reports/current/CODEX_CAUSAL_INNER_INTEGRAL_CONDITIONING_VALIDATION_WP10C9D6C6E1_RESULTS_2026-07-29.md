# Causal Inner Integral-Conditioning Validation

## WP10c9d6c6e1 results — 2026-07-29

Analyzed base:

```text
8e7b567d5f64b28db8405726586e1bf78fe9da67
```

Frozen manifest:

```text
7eee9c710df8ee48418e0e54007d2f5a02360c07f42af2a750df5d15b3cc9f92
```

## Binding classification

```text
frozen_integral_profiles_ineligible
```

The fail-fast eligibility gate rejects the two frozen
continuum-balanced shear stress profiles. No tangent was built and no
profile was propagated.

The manifest, c6c rejection, c6d cancellation diagnosis, operator, and all
thresholds remain unchanged.

## Eligibility results

Five ordinary unseen profiles pass every frozen eligibility gate:

```text
p3__inward_shear
p3__outward_shear
p5__inward_shear
p5__outward_shear
p3__material
```

Their decisive spectral results are:

| Profile class | `theta_99` | Nyquist alias fraction | Result |
|---|---:|---:|---|
| `p3` | `0.21476` | `6.31e-4` | pass |
| `p5` | `0.26998` | `9.73e-4` | pass |

The frozen limits are `theta_99 <=0.30` and alias fraction `<=1e-3`.

The two balanced profiles have:

| Quantity | Inward | Outward | Gate |
|---|---:|---:|---:|
| `theta_99` | `0.33134` | `0.33134` | `<=0.30` |
| Nyquist alias fraction | `0.003285` | `0.003285` | `<=0.001` |
| Global family purity | `0.99999996` | `0.99999996` | `>=0.995` |
| Minimum active-cell purity | `0.9998887` | `0.9998887` | `>=0.98` |
| Projection defect | `1.891e-12` | `1.891e-12` | `<=2e-12` |
| Endpoint-cell fraction | `0.003863` | `0.003863` | `<=0.005` |

Thus only the spectral and alias gates fail.

## Balance construction

The balance rule itself is well conditioned:

| Quantity | Result |
|---|---:|
| Inward primary coefficient | `-1.3769974448938944` |
| Outward primary coefficient | `-1.3769974448938942` |
| 769/513 relative coefficient difference | `4.90e-13` |
| Secondary initial cancellation ratio | `2.45e-13` |

Both balance gates pass by wide margins.

The rejection has a clear cause: subtracting the full-domain `sin^2` and
`sin^4` windows removes much of their low-frequency content. The remaining
balanced shape has a substantially larger relative high-wavenumber tail,
moving it outside the already certified N128 analytic-window class.

This is not evidence that the integral-conditioning rule fails. The rule
was not exercised because its required stress profiles were not eligible.
It is also not permission to increase `theta_99` or alias thresholds.

## Scientific consequence

The prospective uniform class is not recertified. Embedded discrimination
remains blocked.

The result does establish:

- ordinary unseen `p3` and `p5` profiles lie inside the frozen spectral
  window;
- exact continuum balance can be constructed reproducibly;
- the naive `p2-p4` balance is not an admissible resolved stress profile;
- the fail-before-propagation contract works as intended.

## Recommended next step

Do not propagate only the five eligible ordinary profiles: the alternate
conditioning route would remain untested.

The next bounded task should be an operator-neutral, no-propagation
band-limited cancellation feasibility audit. Freeze a deterministic search
before evaluating it:

1. use a declared low-wavenumber envelope basis;
2. enforce the same continuum lower-height angular balance;
3. choose the eligible candidate by a fixed lexicographic rule using only
   initial spectrum, alias, endpoint, purity, and 769/513 balance metrics;
4. forbid any propagated-history objective;
5. require both shear signs to select the same envelope construction;
6. write and hash the selected profile definitions in a new manifest;
7. propagate only in a later commit.

If no balanced profile exists inside the certified spectral class, stop
trying to validate the alternate route with an artificial exact-cancellation
packet. Preserve c6d as the available mathematical diagnosis and reconsider
the prospective component contract directly.

## Stop gates

Do not:

- edit or relabel c6e0/c6e1;
- raise spectral or alias limits;
- propagate a subset of the frozen profiles;
- tune a balance coefficient to a time history;
- change the operator or production defaults;
- begin embedded or nonlinear work;
- begin fixed-Q averaging or reduced slow evolution;
- run N1024.

## Verification

```text
9 passed
```

Canonical evidence is stored in:

```text
results/canonical/
causal_inner_integral_conditioning_validation_wp10c9d6c6e1/
```
