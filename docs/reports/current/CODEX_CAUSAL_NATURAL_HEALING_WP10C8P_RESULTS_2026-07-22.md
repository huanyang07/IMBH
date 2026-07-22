# WP10c8p natural coordinate-fiber healing audit

Date: 2026-07-22
Base commit: `bf6f3e0cc67c2091ed61b76fad316c2ee472478e`
Production physics changed: no
Production exact-max Rusanov flux changed: no
Production descriptor or BDF integrator changed: no
Moment ladder changed: no

## Decision

WP10c8p evolves the exact decisive WP10c8o equal-coordinate pair under the
unmodified production DAE. Both meshes reject only rapid healing through
`0.025 s`. The controlling interface-4 angular-momentum-flux half-spread
changes from `0.3245299508` to `0.3245265470` at N64 and from
`0.2660855001` to `0.2660844380` at N128. Those are fractional decays of only
`1.05e-5` and `3.99e-6`. The conservative coarse/fine uncertainties are
`2.06e-7` and `2.81e-7` gate units, respectively.

The final N64/N128 spread disagreement is `0.05844211 < 0.10`, essentially
the same as the initial WP10c8o disagreement. Both meshes therefore support
the same narrow decision:

> rapid natural healing of the frozen decisive fiber direction is rejected
> through `0.025 s`.

This work package is a rapid-healing screen. A failure at `0.025 s` does not
prove permanent memory, authorize one auxiliary, or mandate a coarse PDE.

## Locked experiment

The audit consumes, without re-optimization, the exact decisive N64/N128
pairs saved by WP10c8o. Every lifted state discards its parent multistep
history and parent predictor. Each trajectory uses:

- one fresh BDF1 startup with a zero predictor;
- fixed-step BDF2 thereafter;
- an identical physical timestep schedule on the plus and minus sides;
- exact outputs at `0`, `0.0025`, `0.005`, `0.01`, and `0.025 s`;
- complete coarse/fine trajectories with 20/40 subdivisions;
- independent split/restart replay;
- fresh nonlinear coordinate rates at every exact output.

The retained coordinate set, five-shell layout, source, boundary conditions,
spatial reconstruction, exact-max Rusanov flux, and every physical state gate
are unchanged.

## Binding gates

The rapid-healing classification requires all of the following:

- maximum coarse/fine spread uncertainty no larger than `0.025` gate units;
- relative uncertainty no larger than `10%` whenever the measured spread is
  at least `0.10`;
- uncertainty-inclusive final spread no larger than `0.10`;
- at least factor-two decay of every initially significant spread;
- no late regrowth;
- final retained-coordinate drift no larger than `0.10`;
- interface-specific impulse no larger than `0.10` of either adjacent shell's
  frozen WP10c8i scale;
- physical shell mass/angular-momentum/Killing-energy ledger defects no larger
  than `1e-3`;
- exact flux decomposition, state gates, fresh-rate audits, and deterministic
  replay all passed.

The fifth causal-stress evolution ledger is retained as a diagnostic but is
not substituted for the declared physical `M/J/E_K` shell ledger.

## N64 result

All four N64 trajectories complete without a rejected step. The 20-step
coarse sides require 101 Jacobians each; the 40-step fine sides require
201/202 Jacobians. Across all four trajectories:

| Quantity | Worst value | Gate |
|---|---:|---:|
| Scaled nonlinear residual | `9.01e-12` | `<=1e-11` |
| Scaled algebraic residual | `3.46e-14` | `<=1e-11` |
| Discrete BDF ledger defect | `4.22e-12` | `<=1e-10` |
| Cumulative physical-ledger defect | `4.69e-4` coarse; `1.17e-4` fine | convergent/pass |
| Physical shell `M/J/E_K` ledger defect | `6.20e-4` coarse; `3.12e-4` fine | `<=1e-3` |
| Pair physical-ledger defect on frozen shell scales | `2.67e-15` coarse; `7.51e-16` fine | `<=1e-3` |
| Flux decomposition defect | `2.57e-15` | `<=1e-12` |
| Maximum temporal spread uncertainty | `2.06e-7` | `<=0.025` |
| Final retained-coordinate drift | `9.09e-8` | `<=0.10` |
| Final interface-4 impulse fraction | `1.92e-8` | `<=0.10` |
| Split replay state difference | exactly zero | bitwise equality |

Every fresh coordinate-rate audit passes. The largest coordinate-rate
half-spread in the complete output stack is `9.09e-8` gate units, so natural
coarse-coordinate drift is negligible on this burst.

### Persistent transport vector

The fine-grid gate-normalized interface-4 half-spreads are:

| Time (s) | Rest-mass flux | Angular-momentum flux | Killing-energy flux |
|---:|---:|---:|---:|
| `0` | `0.21680834` | `0.32452995` | `0.21723405` |
| `0.0025` | `0.21680808` | `0.32452961` | `0.21723380` |
| `0.005` | `0.21680783` | `0.32452927` | `0.21723354` |
| `0.01` | `0.21680732` | `0.32452859` | `0.21723303` |
| `0.025` | `0.21680579` | `0.32452655` | `0.21723150` |

The controlling flux does not decay by the required factor of two and never
approaches the `0.10` rapid-healing reserve.

### Exact production-flux decomposition

At the initial time, the plus/minus interface-4 transport difference is
almost entirely the central perfect-fluid contribution:

| Component | Mass share | Angular-momentum share | Killing-energy share |
|---|---:|---:|---:|
| Central perfect fluid | `0.9999822` | `0.9992276` | `0.9999800` |
| Central causal stress | `0` | `0.0009421` | `2.83e-6` |
| Rusanov dissipation | `1.78e-5` | `-0.0001698` | `1.72e-5` |

The two sides retain different exact-max controller codes at interface 4,
but the Rusanov contribution is too small to explain the unresolved
transport. The counterexample is therefore a local perfect-fluid
trace/transport ambiguity, not primarily numerical switching or unresolved
causal stress.

The five time samples form an almost rank-one `M/J/E_K` response, with
second-to-first singular-value ratio `2.21e-7`. This row is diagnostic only:
one direction, one amplitude, one anchor, and less than one e-folding cannot
authorize a dynamic auxiliary.

## N128 confirmation

All four N128 trajectories also complete without a rejected step. The
20-step coarse sides require 101 Jacobians each; the 40-step fine sides
require 194/197 Jacobians. Across those trajectories:

| Quantity | Worst value | Gate |
|---|---:|---:|
| Scaled nonlinear residual | `9.96e-12` | `<=1e-11` |
| Scaled algebraic residual | `6.27e-13` | `<=1e-11` |
| Discrete BDF ledger defect | `8.15e-11` | `<=1e-10` |
| Cumulative physical-ledger defect | `4.72e-4` coarse; `1.18e-4` fine | convergent/pass |
| Physical shell `M/J/E_K` ledger defect | `6.22e-4` coarse; `3.13e-4` fine | `<=1e-3` |
| Pair physical-ledger defect on frozen shell scales | `3.40e-15` coarse; `1.99e-14` fine | `<=1e-3` |
| Flux decomposition defect | `4.63e-15` | `<=1e-12` |
| Maximum temporal spread uncertainty | `2.81e-7` | `<=0.025` |
| Final retained-coordinate drift | `8.61e-8` | `<=0.10` |
| Final interface-4 impulse fraction | `1.58e-8` | `<=0.10` |
| Split replay state difference | exactly zero | bitwise equality |

Every N128 fresh coordinate-rate audit passes. The largest coordinate-rate
half-spread is `8.61e-8` gate units, again making coarse-coordinate drift
negligible over the burst.

The fine N128 interface-4 half-spreads are:

| Time (s) | Rest-mass flux | Angular-momentum flux | Killing-energy flux |
|---:|---:|---:|---:|
| `0` | `0.17775096` | `0.26608550` | `0.17810002` |
| `0.0025` | `0.17775085` | `0.26608539` | `0.17809991` |
| `0.005` | `0.17775075` | `0.26608529` | `0.17809981` |
| `0.01` | `0.17775054` | `0.26608508` | `0.17809959` |
| `0.025` | `0.17774990` | `0.26608444` | `0.17809896` |

The N128 initial flux decomposition reaches the same conclusion as N64:

| Component | Mass share | Angular-momentum share | Killing-energy share |
|---|---:|---:|---:|
| Central perfect fluid | `0.9999885` | `0.9990840` | `0.9999858` |
| Central causal stress | `0` | `0.0009428` | `2.83e-6` |
| Rusanov dissipation | `1.15e-5` | `-2.68e-5` | `1.13e-5` |

The N128 transport-history singular-value ratio is `3.49e-7`. This is
consistent with N64 but remains nonbinding for an auxiliary because the audit
contains only one fiber direction, one amplitude, one anchor, and no measured
e-folding.

## Interpretation and locked next action

The matched N64/N128 data reject rapid natural healing of the decisive hidden
transport state. They do not decide whether that state relaxes on the known
`0.15-0.17 s` stress timescale, contains slower memory, or requires a
spatially distributed coarse state.

- extend N64 geometrically to `0.05`, `0.10`, and at most `0.125 s` before
  selecting a memory architecture;
- authorize one auxiliary only after at least two clear e-foldings, mesh and
  amplitude agreement, rank-one full `M/J/E_K` response, no plateau or second
  mode, and a held-out fiber/anchor confirmation;
- if the response is persistent, non-rank-one, or spatially distributed,
  prefer a conservative coarse radial finite-volume/PDE model;
- use constrained healing only if retained-coordinate drift becomes material;
- do not add tide, wind, a macrostepper, or a loading-time claim in this
  package.

## Evidence

Primary machine-readable results:

- `outputs/tables/causal_natural_healing_wp10c8p.json`
- `outputs/tables/causal_natural_healing_wp10c8p_arrays.npz`
- `outputs/checkpoints/causal_five_field_wp10c8p/`

The NPZ retains exact output states, all 34 coordinates and fresh rates, all
four macro-interface transport histories, exact interface-4 flux pieces,
shell storage/source/boundary/height-work ledgers, state profiles, controller
codes, temporal uncertainties, and diagnostic transport singular vectors.

Final hashes:

- JSON: `22fe0325433fd03c7a5d8a1ef51aebab91ec97ee7bc2501b4ac3aac42ee79e89`
- NPZ: `aa3931d60384a9324171171e368e123db42f18f59c6bd5e55390952aac2acc88`
- N128 minus replay:
  `b4d1fc68c1c3006cb3d5d23979019bb27930fea086e8e78e6171d2db42559702`
- N128 plus replay:
  `838eb796cb7e5972be447c7b92355ddc2d9e4127cddbc6f1b4e40db1e7e2018a`
