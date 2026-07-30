# Causal Inner Variable-Coefficient Windowed Contract

## WP10c9d6c6a2 results — 2026-07-29

Analyzed base:

```text
8d7f4ebcf5ab3fe97dfdc54abf2eb82c5ffb0858
```

Parent:

```text
76e34e8ef0b4688b0c371c27cb2288c880419961
```

## Binding classification

```text
variable_coefficient_windowed_contract_certified_packet_manifest_authorized
```

The frozen variable-coefficient window class passes. A packet-definition
manifest is authorized next.

This result does **not** authorize:

- propagation of a new physical packet suite;
- embedded-grid discrimination;
- nonlinear evolution;
- production promotion;
- fixed-Q experiments;
- reduced slow-time evolution.

The historical c6a classification remains unchanged:

```text
symbol_derived_packet_resolution_contract_failed
```

## Question answered

The fixed-radius c6a complete symbol exceeded its frozen half-budget at
`theta=0.18` over `0.125 s`, while the c6a1 variable-radius ray preflight
stayed well below that budget at `theta=0.20`.

c6a2 asks whether the ray result survives when the calculation includes:

- finite analytic windows;
- the complete variable-coefficient monolithic tangent;
- all spatial coupling in the N128/N256/N512 operators;
- all five physical characteristic families;
- one mixed five-field direction;
- finite-domain boundary loss;
- and a separately controlled continuum estimate.

It does.

## Frozen probe construction

The audit uses the unchanged C4 continuum background and exact
proper-measure finite-volume projection from c3.

A 513-node physical characteristic field is evaluated over the complete
uniform domain and sign-aligned by adjacent dimensionless overlap. The
construction reports:

| Quantity | Value |
|---|---:|
| Minimum adjacent overlap | `0.9999999822` |
| Maximum dimensionless norm defect | `1.11e-16` |
| Maximum eigenpair defect | `3.91e-16` |
| Maximum basis condition number | `81.74` |
| Minimum coordinate-speed gap | `0.0050097 c` |

Two analytic finite-interval windows were frozen before propagation:

\[
w_p(x)=\sin^p\!\left[
\pi\frac{x-x_{\rm in}}{x_{\rm out}-x_{\rm in}}
\right],
\qquad x=\ln R,
\]

with `p=2` and `p=4`.

The probe set contains:

- five `p=2` low-wavenumber controls;
- five `p=4` binding family probes;
- one `p=4` mixed probe.

The five family labels are:

```text
inward_acoustic
inward_shear
material
outward_shear
outward_acoustic
```

The mixed coefficients, fixed before propagation, are:

```text
(0.35, -0.40, 0.50, -0.45, 0.30)
```

## Spectral contract

The packet spectrum is calculated from the actual N128 proper-measure cell
averages after division by fixed physical field scales.

| Window class | `theta_99` | Maximum alias fraction | Role |
|---|---:|---:|---|
| `p=2` | `0.1840777` | `4.68e-4` | low control |
| `p=4` | `0.2454369` | `8.02e-4` | binding |

All values pass the unchanged alias gate:

\[
f_{\rm alias}\le10^{-3}.
\]

The binding class reaches beyond the unchanged usable-range requirement:

\[
\theta_{99}=0.24544\ge0.20.
\]

The maximum endpoint-cell amplitude fraction is

\[
1.4541\times10^{-3}<5\times10^{-3}.
\]

Primary order-24 and secondary order-12 projections select the same
`theta_99` bin for every probe.

## Variable-coefficient propagation

Every probe is propagated over

\[
0\le t\le0.125\ {\rm s}
\]

with the unchanged full monolithic tangents on:

```text
uniform_N128
uniform_N256
uniform_N512
```

The finer proper-measure cell averages are exactly restricted to N128.
A three-level observed-order Richardson reference is compared with an
independent fixed-second-order Richardson reference.

The inherited tangent method reports pass on all three grids. The largest
directional stationary defect is `1.80e-8`; generator factorization defects
remain below `2.67e-16`.

## Binding results

Across all eleven probes:

| Gate quantity | Worst value | Gate |
|---|---:|---:|
| Observed history order | `1.95418` | `>=1.50` |
| Significant component order | `1.70126` | `>=1.25` |
| Refinement-error cosine | `0.96217` | `>=0.90` |
| Maximum N128/Richardson error | `0.00388429` | `<=0.025` |
| Reference-choice/fine-difference ratio | `0.030876` | `<=0.10` |
| Window-projection/fine-difference ratio | `1.73e-11` | `<=0.10` |
| Restart/fine-difference ratio | `3.05e-11` | `<=0.10` |
| Boundary-integral uncertainty/fine-difference ratio | `6.10e-10` | `<=0.10` |

The binding `p=4` probes are especially consistent:

- observed orders: `1.9656–2.0048`;
- significant component orders: `1.7896–1.9794`;
- error cosines: `0.9893–0.9969`;
- maximum N128/reference errors: `0.000740–0.003884`.

The low controls also pass every state and reference gate.

## Boundary-integration method correction

The initial c6a2 run used a 65-point composite trapezoid to estimate
cumulative boundary loss. Every physical state-convergence gate passed, but
that boundary quadrature uncertainty was

\[
0.736\text{--}1.179
\]

times the already-small N256/N512 cumulative boundary difference. The
predeclared uncertainty gate was `0.10`, so the initial classification was:

```text
variable_coefficient_windowed_contract_failed_packet_manifest_blocked
```

That failed summary is preserved in:

```text
trapezoid_preflight_summary.json
```

The corrected audit changes only the integration method. For a linear
generator `G` and initial direction `v`,

\[
G\int_0^T e^{tG}v\,dt
=
e^{TG}v-v.
\]

The integral is solved directly and subjected to one step of iterative
refinement. The physical probes, windows, tangents, time horizon, spatial
grids, and scientific gates are unchanged.

The maximum refined solve residual is

\[
8.40\times10^{-15},
\]

and the maximum propagated boundary-integral uncertainty is

\[
6.10\times10^{-10}
\]

of the binding fine spatial difference.

The trapezoid result remains reported as a diagnostic. It demonstrates that
coarse time sampling was insufficient for the small cumulative
boundary-difference denominator; it is not evidence of a spatial failure.

## Interpretation

The c6a fixed-radius result was conservative but too restrictive for these
finite variable-coefficient windows. The independent global calculation
shows that the complete spatially varying operator has:

- near-second-order grid contraction;
- stable refinement-error direction;
- small N128 error relative to a controlled reference;
- and a usable analytic window class extending past `theta=0.20`.

This does not erase c6a. The two audits answer different questions:

- c6a rejects a worst-case fixed-radius operator-norm contract;
- c6a2 certifies one explicitly declared finite-window,
  variable-coefficient class.

The certified statement must not be generalized to arbitrary packet width,
support, family mixture, or boundary overlap.

## Authorized next package

The only authorized next package is:

```text
WP10c9d6c6b_packet_definition_manifest_only
```

It must make no propagation run and no operator change.

The manifest should:

1. Freeze every proposed physical packet analytically.
2. Record family, support, center, width, sign, amplitude, and role.
3. Project each definition to N128 without propagation.
4. Require its actual `theta_99` and alias fraction to lie inside the
   certified c6a2 class.
5. Require endpoint/boundary overlap to be no larger than the certified
   window class unless a separate one-sided DAE contract already covers it.
6. Freeze local-family leakage and characteristic-field conditioning.
7. Separate binding packets from below-threshold stress controls.
8. Hash the complete manifest before any propagation.

Only after an independent review of that frozen manifest may the project
consider one prospective uniform packet-propagation package.

## Stop gates

Do not:

- relabel or amend c6a;
- change the `0.025` or `theta>=0.20` gates;
- fit a packet after observing propagation;
- propagate the c6b manifest in the same commit;
- start embedded coupling work;
- start nonlinear evolution;
- change production defaults;
- start fixed-Q averaging;
- start reduced slow-time evolution;
- run N1024 as a rescue;
- add tide, wind, hot-state, S-curve, or cycle physics.

## Reproducibility

Canonical evidence is stored in:

```text
results/canonical/causal_inner_windowed_contract_wp10c9d6c6a2/
```

It contains:

- frozen configuration and gates;
- both initial and corrected classifications;
- characteristic-field arrays;
- projected N128 probe definitions;
- spectra and cumulative spectral energy;
- time-resolved coarse/medium and medium/fine errors;
- Richardson endpoint references;
- source hashes, provenance, and SHA-256 checksums.

## Verification

The c6a-through-c6a2 focused method, campaign, and canonical suite passes:

```text
52 passed
```
