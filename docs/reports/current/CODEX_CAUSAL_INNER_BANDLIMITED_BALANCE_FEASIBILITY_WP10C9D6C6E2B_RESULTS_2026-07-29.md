# Causal Inner Band-Limited Balance Feasibility

## WP10c9d6c6e2b results — 2026-07-29

Analyzed base:

```text
9f4f4b3a720a404619663206878ffc475228eb3f
```

Frozen search manifest:

```text
7dba2fd9db6cc093eff8a3307dfe851036fee3fbb243a5100b1f0819d4b44c02
```

Selected-profile manifest:

```text
4c47cd2b2e26e1a0e5b8e377edbf9ff3ebdc646e333ceca0414bcc366ff048a4
```

## Binding classification

```text
no_eligible_bandlimited_balance_profile
```

None of the six prospectively frozen balance candidates passes the
unchanged initial spectral-class gates for both shear signs. No profile is
selected, no tangent is built, no state is propagated, and no operator or
threshold is changed.

The c6c rejection, c6d cancellation diagnosis, c6e0/c6e1 results, and c6e2a
manifest remain unchanged.

## Feasibility results

Both inward- and outward-shear versions give the same limiting metrics to
reported precision:

| Candidate | `theta_99` | Nyquist alias fraction | Result |
|---|---:|---:|---|
| `p2_cos1` | `0.257709` | `0.001540` | fail alias |
| `p2_cos2` | `0.331340` | `0.003285` | fail spectrum and alias |
| `p3_cos1` | `0.300660` | `0.002017` | fail spectrum and alias |
| `p3_cos2` | `0.368155` | `0.004102` | fail spectrum and alias |
| `p4_cos1` | `0.337476` | `0.002515` | fail spectrum and alias |
| `p4_cos2` | `0.411107` | `0.004945` | fail spectrum and alias |

The frozen limits are:

```text
theta_99              <= 0.30
Nyquist alias fraction <= 0.001
```

The closest candidate is `p2_cos1`. It passes the `theta_99` limit but its
alias fraction is `1.54009e-3`, so it remains ineligible. The threshold is
not relaxed for this near miss.

## The failure is spectral, not algebraic

For `p2_cos1`, the shared decisive values are:

| Quantity | Result | Gate |
|---|---:|---:|
| Balance coefficient | `5.499111952689` | absolute value `<=50` |
| 769/513 coefficient defect | `2.58e-12` | `<=1e-6` |
| Secondary cancellation ratio | `1.29e-12` | `<=1e-6` |
| Inward/outward coefficient difference | `1.62e-16` | `<=1e-10` |
| Global family purity | `0.999999979` | `>=0.995` |
| Minimum active-cell purity | `0.999974325` | `>=0.98` |
| Endpoint-cell fraction | `0.003351` | `<=0.005` |
| Projection defect | `1.146e-12` | `<=2e-12` |

The other candidates likewise satisfy the balance construction, family,
endpoint, and projection contracts. Exact cancellation systematically
removes enough low-frequency content that the relative high-wavenumber tail
leaves the already certified N128 class.

This demonstrates that the artificial exact-cancellation stress-profile
route is incompatible with the frozen admissibility class for the declared
candidate library. It does not show that the underlying local operator or
cellwise lower-height-work map is defective.

## Scientific decision

Do not:

- expand the search after seeing these results;
- raise the spectral or alias limits;
- select `p2_cos1` because it is close;
- propagate an ineligible candidate;
- use propagated histories to optimize another balance envelope;
- amend the c6c or c6d classifications;
- change the operator;
- begin embedded, nonlinear, fixed-Q, or reduced slow-time work.

The synthetic exact-cancellation stress-profile route is closed.

## Recommended next step

The next prospective package should freeze a proof-style band-envelope
certificate that does not require an artificial profile with an exactly
vanishing full-domain integral.

The alternate route should use the already established mathematical
structure:

1. every active cell and every disjoint physical band must contract;
2. band refinement errors must retain their signed directions;
3. the sum of absolute band-error bounds must satisfy the fixed physical
   export tolerance;
4. cell, band, Gram-matrix, and full-ledger closures must be exact;
5. continuum-reference uncertainty must remain below `0.1` of the fine
   spatial difference;
6. the full-domain cancellation ratio must be no larger than `0.25`.

Freeze that contract before any propagation. Use only ordinary unseen
spectrally eligible profiles, including the already frozen `p3/p5` shear and
`p3` material definitions, and retain the direct component-order route
unchanged. The new prospective contract may certify a declared resolved
profile class; it may not retroactively pass c6c.

Embedded discrimination remains blocked until the prospective uniform
profile class passes its complete frozen contract.

## Verification

```text
5 passed
```

Canonical evidence is stored in:

```text
results/canonical/
causal_inner_bandlimited_balance_feasibility_wp10c9d6c6e2b/
```
