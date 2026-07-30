# Causal inner complete-symbol limiter — WP10c9d6c6a1

Date: 2026-07-29

Analyzed base commit: `76e34e8ef0b4688b0c371c27cb2288c880419961`

Analyzed parent: `14bc3e753c2530ef8799d5ad092854156a6c6551`

## Binding classification

```text
full_symbol_limiter_convergent_accumulation_windowed_contract_audit_authorized
```

WP10c9d6c6a1 changes no physical or numerical operator. It preserves the
binding WP10c9d6c6a result

```text
symbol_derived_packet_resolution_contract_failed
```

without relaxing either

```text
complete semigroup error <= 0.025
minimum usable theta     >= 0.20.
```

It authorizes only:

```text
WP10c9d6c6a2_variable_coefficient_windowed_contract
```

It does not certify a packet-resolution contract or authorize a packet
manifest, packet propagation, embedded work, nonlinear evolution,
production promotion, fixed-Q averaging, or reduced slow evolution.

## Question

The c6a local frozen symbol first fails at

\[
\theta=0.18,\qquad R=7.98937\,r_g,
\]

because its complete `0.125 s` semigroup error is

\[
0.0252238>0.025,
\]

while its principal semigroup error is only `0.00159390`.

WP10c9d6c6a1 asks:

1. Is one DAE component nonconvergent?
2. Is the excess caused by unstable or non-normal amplification?
3. Does the fixed-radius `0.125 s` calculation overestimate the error
   accumulated by a transported characteristic family?

## Exact DAE decomposition

Each numerical local symbol is left-normalized by its zero-wavenumber
descriptor:

\[
D_h(\theta)\,\dot q+
\sum_g A_{h,g}(\theta)q=0.
\]

The six discrete/continuum players are:

1. temporal descriptor;
2. complete principal transport;
3. mapped-storage rate derivative;
4. responsive-height-storage rate derivative;
5. physical local stress relaxation;
6. geometry, cooling, stream, and remaining lower sources.

For every subset of those players, a hybrid DAE is assembled and propagated.
The matrix-valued numerical/continuum difference is allocated through the
exact Shapley average over all 64 hybrids. This makes the contributions,
including noncommuting cross terms, sum to the complete propagator
difference to roundoff.

## Method results

| Diagnostic | Observed | Gate |
|---|---:|---:|
| Component-ledger closure | `1.98e-16` | `<=1e-11` |
| Complete-generator parity | `1.23e-13` | `<=1e-11` |
| Shapley matrix closure | `2.89e-17` | `<=1e-11` |
| Maximum numerical/continuum propagator-norm ratio | `1.00395` | `<=1.25` |
| Minimum significant component order | `1.95768` | `>=1.25` |

Every significant player converges at approximately second order:

| Player | Minimum observed order |
|---|---:|
| Descriptor | `1.98603` |
| Principal | `2.00990` |
| Mapped storage rate | `1.98263` |
| Height storage rate | `1.95768` |
| Stress relaxation | `1.96613` |
| Lower sources | `1.98596` |

No nonconvergent descriptor, storage, relaxation, path, or lower-source
component is selected.

## Limiter-point attribution

At `R=7.98937 rg`, `theta=0.18`, and `t=0.125 s`, the Shapley contribution
norms relative to the common propagator scale are:

| Player | Relative norm | Cosine with total error |
|---|---:|---:|
| Descriptor | `0.0208701` | `+0.80577` |
| Principal | `0.0273804` | `-0.78170` |
| Mapped storage rate | `0.00828155` | `+0.39523` |
| Height storage rate | `0.000168663` | `+0.18319` |
| Stress relaxation | `0.000592096` | `+0.04060` |
| Lower sources | `0.0273799` | `+0.96723` |

Several individual norms exceed the final total because the physical pieces
strongly cancel. In particular, the principal and lower-source
contributions are large and nearly opposed. These values therefore do not
support a one-block repair.

The complete-error accumulation exponents across successive time doublings
are

\[
0.99469,\qquad0.96260,\qquad0.85116,
\]

inside the prospectively frozen `[0.75,1.25]` interval. Together with the
nearly second-order cross-grid contraction and the propagator-norm ratio
near one, this classifies the limiter as ordinary convergent finite-time
accumulation, not an unstable or nonconvergent operator term.

## Ray-method correction

Two preliminary ray evaluations are retained as failed method evidence:

| Method | Maximum step/error ratio | Result |
|---|---:|---|
| Midpoint exponential with speed sorting | `1.22473` | fail |
| Coupled RK4 with speed sorting | `0.615651` | fail |

The physical ray errors were already near `0.005` in both cases, and the
769/513 continuum-reference differences were negligible. The failure was
the ray integrator’s own step sensitivity.

The cause was pointwise family sorting by instantaneous characteristic
speed. Several speed orderings exchange near `4.2`, `4.4`, and `4.9 rg`.
Rays starting near `5 rg` crossed those locations and changed family labels.

The corrected method:

- tracks the five branches by neighboring eigenvector overlap;
- phase-aligns their right eigenvectors;
- retains the same 30 physical rays;
- retains the same `0.00125/0.0025 s` steps;
- retains the same `0.125 s` horizon and all scientific gates;
- uses coupled fourth-order integration.

The minimum neighboring tracked-branch overlap is

\[
0.998278>0.90.
\]

The corrected maximum step/error ratio is

\[
0.00106582<0.10,
\]

and the maximum continuum-reference/error ratio is

\[
1.13\times10^{-8}<0.10.
\]

The failed midpoint and speed-sorted RK4 reports and arrays remain in the
canonical evidence.

## Variable-radius preflight

The corrected maximum errors are:

| Start radius | `theta=0.18` | `theta=0.20` |
|---:|---:|---:|
| `5 rg` | `0.00403621` | `0.00489586` |
| `8 rg` | `0.00401892` | `0.00418657` |
| `11 rg` | `0.00475855` | `0.00503722` |

Thus the maximum at the previously required usable wavenumber is

\[
0.00503722<0.025.
\]

This is substantially smaller than the failed fixed-radius `0.125 s`
operator-norm result. It demonstrates that coefficient variation and
continuous characteristic transport materially change the accumulated
local error estimate.

It does not yet certify packets. A ray-ordered local-symbol model omits
finite packet width, spectral convolution, inter-family spatial coupling,
and window-boundary effects.

## Authorized next package

WP10c9d6c6a2 must construct an independent variable-coefficient windowed
contract before any physical packet definitions are propagated.

Requirements:

1. Freeze analytic band-limited window probes before propagation.
2. Use the full variable-coefficient N128/N256/N512 monolithic tangents.
3. Use a separately converged continuum or Richardson-controlled reference.
4. Retain the c6a `0.025`, `theta>=0.20`, and 99-percent spectral-energy
   requirements.
5. Include all five overlap-tracked characteristic families and mixed
   directions.
6. Require window, time-step, continuum-reference, and boundary-loss
   uncertainty below 10 percent of the binding spatial difference.
7. Keep c6a rejected even if the new windowed contract passes.

### Binding c6a2 decisions

- Every prospectively frozen window probe passes and a usable range reaches
  `theta>=0.20`: authorize a packet-definition manifest only.
- A qualified window probe fails: stop and localize that exact probe.
- Reference or window uncertainty fails: repair the method, not the physical
  operator.
- No useful range exists: stop packet validation and reconsider the
  complete numerical architecture.

## Stop gates

WP10c9d6c6a1 does not authorize:

- changing the c6a gates;
- a packet-definition manifest;
- uniform physical packet propagation;
- operator redesign;
- embedded coupling work;
- nonlinear evolution;
- production promotion;
- fixed-Q averaging;
- reduced slow-time evolution;
- N1024 refinement;
- tide, wind, hot-state, S-curve, or QPE-cycle physics.

## Reproducibility

Canonical evidence is stored in:

```text
results/canonical/causal_inner_full_symbol_limiter_wp10c9d6c6a1/
```

It includes exact hybrid contributions, full Gram matrices, cross-grid
orders, all three ray-method histories, tracked branches and velocities,
source hashes, provenance, environment, and SHA-256 checksums.

## Verification

The focused c3-through-c6a1 method, campaign, and canonical-evidence suite
passes:

```text
41 passed
```

A repository-wide replay was also sampled through 136 passing tests with no
failure, then stopped after 7 minutes 53 seconds because the historical
numerical campaigns make that replay substantially longer than this bounded
work package. It is not represented as a completed full-suite pass.
