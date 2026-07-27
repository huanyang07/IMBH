# Causal Inner Radial Fluctuation WP10c9d4b Results

Date: 2026-07-27
Analyzed parent: `10546da78561ccb4a5f60a203b8b80a47fa26be3`
Work package: WP10c9d4b

## Classification

```text
radial_five_field_candidate_gate_passed_frozen_linear_discrimination_authorized
```

The production-neutral radial complete-fluctuation candidate passes its
declared method gates. WP10c9d5 frozen-linear candidate discrimination is
authorized.

This result does **not** authorize:

- changing the production operator;
- selecting a nonlinear finite-amplitude path or Riemann solver;
- running a new nonlinear truth trajectory;
- fixed-`Q` averaging;
- a reduced slow-time model;
- tide, wind, hot-state, S-curve, or QPE-cycle work.

## What was implemented

The fixed-geometry d4a fluctuation contract is extended to the actual
nonuniform Kerr--Schild radial grid. The candidate uses:

1. exact grid face and cell measures;
2. the unchanged production quadratic reconstruction and admissibility
   contract;
3. one shared conservative face flux;
4. separate shear-principal and responsive-height-principal fluctuations;
5. actual endpoint-measured within-cell conservative transport,
   \[
   {\cal A}_{i+1/2}F(p^R_i)
   -
   {\cal A}_{i-1/2}F(p^L_i);
   \]
6. a declared path linear in `log R` and in the primitive chart;
7. separate local Maxwell--Cattaneo relaxation, geometry, cooling, stream,
   and lower-order responsive-height blocks;
8. unchanged one-sided outgoing excision and frozen-exterior coupling state
   contracts.

The candidate is isolated in
`causal_inner_radial_fluctuation.py`. No production default or production DAE
residual was changed.

## Complete residual ledger

Every candidate cell residual is the explicit sum

\[
R =
R_{\rm conservative}
+R_{\rm shear,principal}
+R_{\rm height,principal}
+R_{\rm stress,relax}
+R_{\rm geometry}
+R_{\rm cooling}
+R_{\rm stream}
+R_{\rm height,lower}.
\]

The maximum block-ledger defect is exactly zero. The structural source
double-count defect is exactly zero.

An additional physical partition check evaluates the same smooth profile in
two ways:

1. the total shear and height rates along the complete radial profile;
2. explicit-geometry lower rates plus the complete principal matrix acting on
   the analytic primitive derivative.

Their maximum relative discrepancy is

```text
4.61621e-9
```

against the declared `1e-7` gate. Thus the structural zero is supported by an
independent physical source-rate comparison rather than being the only
evidence.

## Shared conservative face contract

At every interior or frozen-exterior face, the conservative part of the
complete characteristic fluctuation defines one candidate face flux. The
same value is used by the cells on both sides. Shear and height source
fluctuations remain separately visible.

Across the N12/N24/N48 ladder:

```text
maximum shared-face or telescoping defect = 2.17070e-16
maximum interface/path partition defect  = 5.63977e-14
incoming excision characteristics        = 0
minimum reconstruction factor            = 1.0
```

The inner face remains exactly the existing one-sided physical excision flux.
The outer audit face retains the declared live frozen-exterior state and one
shared flux.

## Independent radial manufactured balance

The common continuum family is an explicit C2 primitive-chart profile over

```text
1.8 <= R/rg <= 6.648
```

with a small mixed five-field sinusoidal perturbation. The same analytic
function is sampled on N12, N24, and N48 logarithmic grids.

The reference residual is evaluated independently using:

- exact physical endpoint fluxes;
- 12-point Gauss--Legendre integration of the continuum principal matrix;
- the analytic primitive derivative;
- independently evaluated explicit-geometry lower sources.

No residual subtraction or old-production truncation error is used.

| Grid | Active-cell relative L2 error |
|---:|---:|
| N12 | `7.10637e-4` |
| N24 | `1.08523e-4` |
| N48 | `1.51640e-5` |

The observed orders are:

```text
N12 -> N24: 2.71111
N24 -> N48: 2.83927
```

Both exceed the binding `1.8` order gate. The N48 error is more than two
orders of magnitude below the `2e-3` fine-error gate.

This is a manufactured nonequilibrium physical-residual test, not a claim
that an exact stationary disk equilibrium has been constructed.

## Three distinct stationary Jacobians

The audit retains:

\[
J_{\rm prod},\qquad
J_{\rm candidate}^{\rm FD},\qquad
J_{\rm candidate}^{\rm assembled}.
\]

The assembled candidate is the sum of the eight explicit physical block
Jacobians. It is required to close against the finite-difference derivative
of the complete candidate residual. It is deliberately **not** required to
equal the old production Jacobian.

The finite-difference step sweep is:

| Relative step | FD/assembled defect | Candidate change from prior step |
|---:|---:|---:|
| `5e-6` | `2.31186e-9` | — |
| `1e-5` | `1.39353e-9` | `6.12918e-8` |
| `2e-5` | `1.04825e-9` | `2.47525e-8` |
| `4e-5` | `5.06755e-10` | `7.07213e-9` |

The monotone decrease identifies finite-difference cancellation rather than
a missing block. The selected `4e-5` point passes the unchanged `1e-9` gate.

The normalized production/candidate Jacobian difference is:

```text
0.505104
```

This confirms that production remains a distinct baseline and that the new
candidate was not forced to reproduce the old finite-resolution operator.

## Interpretation

WP10c9d4b establishes that one complete radial five-field candidate can:

- preserve a shared conservative face flux;
- place every principal and lower-order source exactly once;
- retain outgoing excision;
- converge against an independent radial physical residual;
- and supply a block-complete stationary Jacobian.

It does not establish that this candidate fixes the negative embedded
physical-export refinement orders from WP10c9d0. That is the binding purpose
of WP10c9d5.

The candidate also still uses the audit straight path and midpoint
finite-difference eigensystem. Those are acceptable for frozen-linear
discrimination, not for nonlinear promotion.

## Next gate: WP10c9d5

Construct unchanged production and candidate frozen generators, then run:

1. inward and outward shear packets;
2. material and acoustic families;
3. the unchanged common mode;
4. predeclared held-out mixed modes;
5. uniform N64/N128/N256 and embedded N128-exterior
   N128/N256/N512-equivalent inner ladders;
6. the complete physical export vector from WP10c9d0.

Use total physical shear energy as the primary shear gate. Treat
selected-family amplitudes only as transfer diagnostics.

The candidate must demonstrate positive contraction of every significant
cumulative physical export, preferably order at least `0.75`, and a fine
normalized complete-export difference no larger than `0.05`. Reject the
candidate if it does not materially improve the current negative embedded
orders.

## Reproducibility

Compact decisive evidence is committed under:

```text
results/canonical/causal_inner_radial_fluctuation_wp10c9d4b/
```

It contains the manufactured profile, fine candidate/reference residuals,
candidate and production shared fluxes, all three stationary Jacobians, the
eight candidate block Jacobians, configuration, provenance, and checksums.

Final repository verification:

```text
810 passed, 4 subtests passed
```

## Artifacts

- implementation:
  `src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_fluctuation.py`
- runner:
  `scripts/run_causal_inner_radial_fluctuation_audit_wp10c9d4b.py`
- method tests:
  `tests/test_causal_inner_radial_fluctuation.py`
- canonical evidence test:
  `tests/test_causal_inner_radial_fluctuation_wp10c9d4b.py`
- canonical evidence:
  `results/canonical/causal_inner_radial_fluctuation_wp10c9d4b/`
