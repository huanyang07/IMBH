# Causal inner packet-resolution contract — WP10c9d6c6a

Date: 2026-07-29

Analyzed base commit: `14bc3e753c2530ef8799d5ad092854156a6c6551`

Analyzed parent: `c082f62f62f9c5c9f28e61c7f25f4d353a5f7a09`

## Binding classification

```text
symbol_derived_packet_resolution_contract_failed
```

WP10c9d6c6a changes no physical or numerical operator. It preserves both:

```text
prospective_heldout_uniform_validation_failed
```

and

```text
narrow_profile_preasymptotic_width_crossover_no_redesign
```

without amendment.

The exact monolithic local symbol and its cross-grid continuum limit are
certified. The prospectively declared *usable* packet range is not:

\[
\theta_{\rm certified}=0.17
<
\theta_{\rm minimum}=0.20.
\]

Therefore no prospective packet manifest or packet propagation is authorized.
Embedded, nonlinear, production, fixed-\(Q\), and reduced slow-time work remain
blocked.

## Purpose

WP10c9d6c5 found that strict refinement-error direction followed packet width:
both log-width `0.065` controls failed and both width `0.130` controls passed.
That diagnostic did not define a prospective resolution threshold.

WP10c9d6c6a replaces an informal cells-per-width rule with a physical symbol
contract. The gates were frozen before the binding run:

| Quantity | Gate |
|---|---:|
| Complete finite-time semigroup error | `<= 0.025` |
| Principal finite-time semigroup error | `<= 0.025` |
| Principal phase accumulation | `<= 0.025 rad` |
| Principal log-amplitude error | `<= 0.025` |
| Group-speed relative error | `<= 0.025` |
| Family leakage | `<= 0.010` |
| Principal-basis condition number | `<= 1e4` |
| Continuum-reference/discrete-error ratio | `<= 0.10` |
| Minimum usable certified wavenumber | `theta >= 0.20` |
| Cross-grid symbol order | `>= 1.50` |

The finite-time checks use

\[
t=(0.03125,\ 0.0625,\ 0.125)\ {\rm s}.
\]

The `0.025` semigroup budget reserves half of the existing `0.05` physical
export-difference budget.

## Exact generalized symbol

At one strictly interior cell, the complete frozen block row is converted
from the tangent's scaled coordinates to fixed physical field coordinates.
Its exact local stencil defines

\[
M_h(\theta)
=
\sum_\ell M_\ell e^{i\ell\theta},
\qquad
E_h(\theta)
=
\sum_\ell E_\ell e^{i\ell\theta},
\]

and

\[
G_h(\theta)
=
-M_h(\theta)^{-1}E_h(\theta).
\]

This construction includes:

- the reconstructed mapped and responsive-height temporal descriptor;
- the self-consistent base-rate storage derivative;
- conservative transport;
- shear- and height-principal terms;
- local stress relaxation;
- geometry, cooling, stream, and lower responsive-height work.

It does not reinterpret the one-sided excision row as periodic. Boundary-
overlapping packets require both the spectral contract and the separately
certified one-sided DAE-truncation contract.

The independent continuum symbol uses the 769-node smooth continuum
background. A 513-node construction supplies the reference-uncertainty check.

## Method and cross-grid results

The selected symbols are evaluated near:

\[
R/r_g=
2.20,\ 3.00,\ 5.00,\ 8.00,\ 11.00.
\]

All selected rows are strictly interior. The extracted stencil is the exact
active `[-1,0,1,2]` block row.

| Diagnostic | Maximum/minimum observed | Gate |
|---|---:|---:|
| Row-symbol parity defect | `1.44e-13` | `<=1e-11` |
| Omitted stencil fraction | `9.97e-15` | `<=1e-11` |
| 769/513 continuum semigroup difference | `2.93e-11` | `<=2.5e-3` |
| Continuum-reference/discrete ratio | `1.73e-8` | `<=0.10` |
| Minimum N128/N256/N512 symbol order | `1.94753` | `>=1.50` |

At fixed physical wavenumbers corresponding to N128
\(\theta=0.10\) and `0.20`, every radius contracts at nearly second order.
The finest observed pair orders extend to approximately `1.99`.

Thus the failed usable-range gate is not caused by:

- a malformed N512 tangent;
- a nonconvergent local symbol;
- omitted stencil couplings;
- continuum-reference uncertainty;
- a boundary row;
- an unresolved characteristic cluster.

## Binding limiter

Every radius passes through

\[
\theta=0.17.
\]

The first failed point is:

\[
\theta=0.18,
\qquad
R=7.98937\,r_g.
\]

At that point:

| Quantity | Value | Gate | Result |
|---|---:|---:|---|
| Complete semigroup error | `0.0252238` | `<=0.025` | **FAIL** |
| Principal semigroup error | `0.00159390` | `<=0.025` | pass |
| Phase accumulation | `0.00125657` | `<=0.025` | pass |
| Log-amplitude error | `0.000474195` | `<=0.025` | pass |
| Group-speed relative error | `0.00442570` | `<=0.025` | pass |
| Family leakage | `0.000579493` | `<=0.010` | pass |

The excess over the binding complete-semigroup gate is:

\[
2.2383\times10^{-4}.
\]

The classification is nevertheless binding. The `0.025` gate and minimum
usable `0.20` range are not relaxed after inspection.

The result localizes the limitation to the complete DAE evolution over the
declared `0.125 s` horizon. Principal dispersion, principal damping, and
family leakage are not the controlling errors.

## Existing c5 packet spectra

The spectra use the actual finite-volume cell-average Gaussian projection,
fixed physical field scales, and a 99-percent energy quantile.

| Packet width | Cells per sigma on N128 | `theta_99` | Spectrally eligible |
|---:|---:|---:|---|
| `0.065` | `1.592` | `1.1290` | no |
| `0.130` | `3.184` | `0.57064` | no |

All four c5 boundary packets retain their certified one-sided DAE-truncation
eligibility. None passes the new spectral requirement. Their historical
pass/fail outcomes were not used to choose the symbol threshold and remain
nonbinding for this contract.

This does not contradict the empirical passage of the width-`0.130` packets.
It says that the stricter prospective 99-percent spectral/half-export-budget
contract does not certify them.

## Interpretation

WP10c9d6c6a establishes:

1. The exact monolithic local symbol converges at nearly second order.
2. Its principal characteristic propagation is accurate well beyond the
   binding certified range.
3. The complete DAE semigroup over `0.125 s`, not the principal symbol, limits
   the frozen error budget.
4. The declared range is too small to authorize the planned resolved-packet
   suite under the prospective contract.

It does **not** establish an operator inconsistency or authorize a spatial
redesign. The failure is a deliberately conservative accuracy-budget result.

## Bounded next diagnostic

No next physical propagation is authorized by this package.

Before reconsidering a packet contract, a separate operator-neutral audit
should:

1. decompose the complete-semigroup difference through a Duhamel ledger into
   principal, descriptor/storage-rate, relaxation, and lower-source
   contributions;
2. distinguish local residence/crossing-time accumulation from the fixed
   `0.125 s` frozen-radius horizon;
3. compare the local frozen estimate with a variable-coefficient
   windowed-continuum propagator;
4. freeze any revised finite-time budget before examining new packet
   histories.

Possible decisions are:

- the complete error is a stable, ordinary second-order zero-order
  accumulation and a separately justified crossing-time contract has a usable
  range: define a new prospective contract in a new work package;
- one DAE block has a nonconvergent or physically excessive contribution:
  authorize only a targeted method audit;
- no usable range exists under an independently justified budget: stop packet
  validation and reconsider the numerical architecture.

The failed c6a gate must remain recorded in every case.

## Stop gates

WP10c9d6c6a does not authorize:

- a packet-definition manifest;
- prospective uniform packet propagation;
- threshold tuning;
- a boundary, storage, path, or source redesign;
- embedded coupling work;
- nonlinear evolution;
- production promotion;
- fixed-\(Q\) averaging;
- reduced slow-time evolution;
- N1024 refinement;
- tide, wind, hot-state, S-curve, or QPE-cycle physics.

## Verification

The focused c3-through-c6a lineage suite completed with `24 passed`.

The repository-wide run completed with `901 passed` and four passing
subtests. It initially exposed two c6a packaging defects: the provenance
record omitted the repository-standard `source_parent_commit`, and the
repository-wide canonical catalog had not been refreshed for the five new
files. Both were corrected without recomputing or changing the scientific
arrays. A targeted rerun of canonical-artifact, c6a, and repository-hygiene
tests then completed with `15 passed` and one remaining failure.

That remaining failure is the pre-existing tracked-file policy:

```text
1003 tracked files < 850 required
```

It is a repository-hygiene issue, not a numerical or scientific failure, and
is deliberately left for a separate non-scientific commit.

## Reproducibility

Canonical evidence is stored in:

```text
results/canonical/causal_inner_packet_resolution_wp10c9d6c6a/
```

It contains:

- frozen gates and eligibility classes;
- exact radius/wavenumber symbol metrics;
- matched principal eigenvalues;
- cross-grid fixed-physical-wavenumber errors and orders;
- c5 packet spectral curves;
- source hashes, environment, provenance, and SHA-256 checksums.
