# WP10c8y Continuum-Matched Inner-Mode Results

- Date: 2026-07-26
- Base commit: `6764fc117ce453b4deb5c6b1c275a19c7352b4be`
- Classification:
  `common_mode_passed_boundary_insensitive_underresolution`
- Production boundary changed: no
- New nonlinear truth evolution: no
- Standalone N512 confirmation: not authorized
- Conservative embedded-patch preflight: authorized
- Production embedded patch, fixed-`Q` averaging, and reduced architecture:
  not authorized

## Executive verdict

WP10c8y removes the initial-perturbation caveat that stopped WP10c8x.
It defines one smooth compact continuum perturbation in `ln R`, constructs
exact nonlinear equal-coordinate pairs at N64/N128/N256, and expresses all
three in one physical normalization.

The exact N128/N256 initial state and fresh-rate profiles now pass every
predeclared gate:

```text
                                        cosine    amp defect    relative L2
full active state                       0.999996    0.00161       0.00331
full active production rate             0.997412    0.00739       0.07206
full active outgoing-linear rate        0.997889    0.00764       0.06518
common-exterior state                   0.999997    0.00163       0.00289
common-exterior production rate         0.997170    0.00902       0.07543
common-exterior outgoing-linear rate    0.997170    0.00902       0.07543
```

The binding requirements were:

```text
signed cosine >= 0.99
amplitude defect <= 0.05
relative L2 difference <= 0.10
```

The bounded `0.125 s` history nevertheless fails for both retained boundary
families:

| Family | State order | Rate order | Fine minimum cosine | State difference | Rate difference |
|---|---:|---:|---:|---:|---:|
| Production | `0.4240` | `-0.6654` | `0.89292` | `0.13674` | `0.52900` |
| Outgoing-linear flux | `0.4239` | `-0.6656` | `0.89292` | `0.13674` | `0.52900` |

The two boundary histories themselves converge rapidly toward one another.
At N256 their common-exterior production/outgoing-linear differences are

```text
state maximum relative L2    1.42e-7
rate maximum relative L2     9.07e-7
state minimum cosine         0.99999999999999
rate minimum cosine          0.99999999999959
```

Therefore:

> The inner fast response remains spatially underresolved after the initial
> mode is matched, and the failure is not controlled by the tested
> production-versus-outgoing-linear excision trace.

The result authorizes a bounded conservative embedded-inner-patch preflight.
It does not authorize installing a production patch, replacing the boundary,
running fixed-`Q` averaging, or selecting a reduced state.

## Continuum perturbation definition

The perturbation is not a mesh eigenvector. It is defined analytically in

```text
x = [ln(R/r_g) - ln(R_inner/r_g)]
    / [ln(R_active/r_g) - ln(R_inner/r_g)]
```

using degree-zero through degree-four Chebyshev functions multiplied by

```text
sin(pi*x)^2 exp(-3*x).
```

The endpoint envelope makes the profile and its first derivative vanish at
the excision and active-core edges. Only the radial-transport and causal
stress chart components are supplied before the exact coordinate-null
projection:

```text
beta_R
chi_specific
```

The primitive metric is a positive degree-five Chebyshev fit to the
independently corrected N256 physical-input amplitudes. It is evaluated as
one continuum function on all meshes.

The selected profile is the direction closest to the declared
stress-dominated template among candidates that minimize the combined
N128/N256 production and outgoing-linear state/rate discrepancy. Its
template cosine is `0.76649`. Thirteen of thirty smooth candidates pass the
linear preflight, so the result is not dependent on a unique nearly singular
optimizer direction.

This construction is a diagnostic matched perturbation. It does not prove
that every unresolved equal-coordinate direction is spatially convergent.

## Exact nonlinear fiber lifts

Every pair is corrected nonlinearly onto its local 12-coordinate fiber.

| Mesh | Coordinate rank | Condition estimate | Pair defect | Maximum correction fraction |
|---|---:|---:|---:|---:|
| N64 | `12` | `1.92e4` | `3.55e-15` | `1.56e-6` |
| N128 | `12` | `2.58e3` | `1.78e-15` | `1.53e-6` |
| N256 | `12` | `1.82e3` | `1.56e-15` | `1.52e-6` |

All plus/minus state gates pass. The N128/N256 buffer half-difference is
exactly zero; the N64 buffer leakage is `1.67e-13` in the common primitive
metric. The signed seed-direction cosines exceed
`0.9999999999987`.

The pair amplitude is `1e-3` in the declared continuum chart. The larger
reported pointwise ratios, approximately `0.0036`, result from the smooth
basis combination and are still safely within the physical state gates.

## Frozen-history result

The certified WP10c8x N64/N128/N256 descriptor generators are reused by
hash. Only their similarity chart and initial vector are changed to the new
common continuum metric and exact pair.

Both boundary families remain causal and propagation-safe. Neither passes:

- state order is about `0.424 < 0.75`;
- rate order is about `-0.665 < 0.75`;
- the N128/N256 rate cosine reaches `0.89292 < 0.90`;
- the maximum N128/N256 rate difference is `0.529`;
- only one matched stress-rate zero crossing exists, so a converged
  frequency cannot be reported.

The outgoing-linear flux changes neither the contraction nor the phase
classification. Its damping diagnostic is somewhat closer, but that is not
a binding improvement.

No time shift is used in any gate.

## Diagnostic modal/subspace audit

The snapshot audit is descriptive only. It does not identify a physical
eigenmode.

At `99.9%` snapshot energy:

```text
N128 retained dimension    4
N256 retained dimension    5
largest common-grid principal angle    34.0 degrees
```

The fitted reduced spectra are not mesh converged. The leading oscillatory
imaginary rates are approximately

```text
N128    24.9 /s
N256    59.3 /s
```

and the corresponding real parts change from approximately `-5.68 /s` to
`+1.34 /s`. The leading stress profile also changes from roughly
`15.5` to `42` cells per diagnostic wavelength. These values must not be
used as a physical frequency or damping law. They show that the same smooth
initial perturbation excites different discrete modal mixtures at N128 and
N256.

## Main problems and solutions

### Problem 1: near-horizon bulk phase is underresolved

The common initial perturbation is now controlled, but its later rate field
does not contract. The failure is substantially larger than the remaining
initial mismatch.

#### Solution

Retain permanent fine resolution over the inner active region during the
next audit. Do not try to repair the result with another initialization
correction or a timestep change.

### Problem 2: the tested inner trace is not the controlling discretization

Production and outgoing-linear histories become essentially identical at
N256 while both fail the cross-mesh contract.

#### Solution

Do not replace the production inner boundary. The next spatial experiment
must refine the near-horizon transport volume and its propagating support,
not only the excision face formula.

### Problem 3: a frozen-exterior local model is not a production patch

The current local operator uses an audit-only frozen outer trace. It cannot
exchange mass, angular momentum, and Killing energy with a slowly evolving
coarse exterior.

#### Solution

Implement a nonoverlapping conservative fine/coarse interface with one
shared production flux and equal-and-opposite ledger contributions. Keep
responsive-height storage in the cell descriptor.

### Problem 4: the diagnostic mode dimension is not converged

The N128/N256 snapshot dimensions, principal angles, frequencies, damping,
and radial wavelengths differ materially.

#### Solution

Use these diagnostics only to size and place the patch. Select no abstract
inner coordinate until a patch-refined nonlinear history and held-out
perturbation demonstrate a mesh-convergent localized subspace.

## Locked next plan: WP10c8z

### Phase 1 — Freeze WP10c8y evidence

Freeze and hash:

- the analytic continuum chart and common primitive metric;
- all exact N64/N128/N256 plus/minus pairs;
- initial state/rate profiles and gates;
- production and outgoing-linear histories;
- boundary-family equivalence data;
- diagnostic subspace data.

Keep production physics, `q_34`, BDF2, and the five-shell layout unchanged.

### Phase 2 — Build a conservative embedded-patch kernel

Construct a nonoverlapping radial grid with:

- permanent fine cells from the excision edge through at least
  `6.648 r_g`;
- a buffer/coupling location initially at `12.777 r_g`;
- the existing coarse exterior beyond the coupling face.

At the coupling face:

1. reconstruct one admissible fine-side trace;
2. reconstruct one admissible coarse-side trace;
3. evaluate exactly one production Rusanov transport vector;
4. insert it with equal magnitude and opposite sign in the fine and coarse
   residuals;
5. use the same flux in every physical ledger.

No frozen exterior trace is allowed in the production-coupling test.

### Phase 3 — Method-level certification

Before a physical history, require:

- constant-state and smooth-profile preservation;
- exact internal-flux telescoping;
- mapped and responsive-height storage separation;
- no incoming excision characteristic;
- no artificial incoming coupling characteristic;
- correct widened Jacobian sparsity/coloring;
- dense-versus-colored Jacobian agreement on a small patch;
- reduction to the existing uniform-grid operator when fine and coarse
  spacings coincide;
- restart/history determinism.

### Phase 4 — Patch-resolution ladder

Use the exact WP10c8y continuum pair and compare at least:

```text
N128-equivalent exterior + N256-equivalent inner patch
N128-equivalent exterior + N512-equivalent inner patch
```

Add one coupling-location variation that keeps the comparison region fixed.
Report state/rate histories both inside the active core and outside
`6.648 r_g`.

Require:

```text
state and rate order >= 0.75
same-time signed cosine >= 0.90
zero-crossing defect <= 0.10
frequency defect <= 0.10 when measurable
damping defect <= 0.25
shared-flux ledger defect <= nonlinear tolerance
coupling-location history defect <= 0.02
```

The N512 work is a patch-resolution preflight, not a standalone confirmation
of the rejected uniform local history.

### Phase 5 — Nonlinear local confirmation

Only if Phase 4 passes:

1. evolve the exact plus/minus pair with the nonlinear patch DAE;
2. demonstrate temporal convergence;
3. reproduce the linear history at smaller amplitude;
4. repeat at the original amplitude;
5. test one held-out analytic continuum perturbation.

### Phase 6 — Architecture decision

- Patch converges and coupling is conservative: authorize a bounded
  nonlinear embedded-patch truth experiment.
- Patch needs still finer inner resolution but contracts: refine locally
  once more and measure cost.
- Patch remains nonconvergent: redesign the bulk near-horizon spatial
  operator; do not add reduced coordinates.
- Significant response reaches the coupling face: enlarge the patch before
  interpreting the mode.
- Several converged localized modes remain: retain the patch or a measured
  local state vector.
- Response becomes distributed: move to the conservative staggered radial
  finite-volume/PDE architecture.

No fixed-`Q` averaging, initial-slip model, reduced coordinate selection,
macrostep, tide, wind, hot-state, or cycle calculation is authorized in
WP10c8z.

## Machine evidence

```text
outputs/tables/causal_inner_common_mode_audit_wp10c8y.json
outputs/tables/causal_inner_common_mode_audit_wp10c8y_arrays.npz
```

Runner:

```text
scripts/run_causal_inner_common_mode_audit_wp10c8y.py
```

The JSON records hashes for the runner, core DAE/fiber/spatial modules,
WP10c8x evidence, arrays, and all reused parent operators.
