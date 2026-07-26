# WP10c8v local inner-phase spatial preflight

Date: 2026-07-26

Base commit:
`3ccdb9532359acbaa197e066a800a9119dfe60ef`

Parent evidence: WP10c8t and WP10c8u

Full truth meshes reused: N64 and N128

Local equivalent meshes: N64, N128, and N256

Production physics changed: no

Production inner boundary changed: no

Production outer Roche boundary changed: no

Production spatial operator changed: no

New nonlinear truth evolution run: no

Formal fast-time average certified: no

Reduced architecture selected: no

## Executive result

WP10c8v tests whether the N64/N128 shell-0 phase failure isolated by WP10c8u
contracts under one local factor-two spatial refinement.

The binding classification is:

> `inner_fast_phase_spatially_unresolved_local_preflight`

The local audit is numerically credible:

- its active N64/N128 generator blocks reproduce the committed full-domain
  generators;
- its complete `0.125 s` active histories reproduce the full-domain
  histories;
- its matrix-exponential time-sampling refinement is exact at the stored
  common times;
- the enlarged frozen-exterior buffer is not controlling the result.

The spatial phase gate nevertheless fails. The maximum state/rate history
differences contract with observed orders only

```text
state  0.22687
rate   0.35462
```

against the predeclared minimum `0.75`. The N128/N256-equivalent rate history
still reaches signed cosine `-0.64978`; its final cosine is only `0.56855`.
The diagnostic shell-0 stress-rate frequency changes from `35.62 Hz` at
N128 to `54.18 Hz` at N256-equivalent, a relative difference `0.34257`, and
the fitted damping envelope changes by `0.51542`.

WP10c8v therefore does not authorize fixed-`Q` averaging, an initial-slip
map, a new reduced coordinate, or a selected embedded-patch architecture.

## Local-domain contract

The first implementation attempt placed the frozen exterior at
`6.6484 rg`, only two N64 cells outside its active core. Its instantaneous
active generator defect was small, but a direct full/local history
comparison showed that the artificial boundary reached the active response
within `0.125 s`. That result was rejected before interpreting the
factor-two refinement.

The binding configuration instead uses:

| Global-equivalent mesh | Local cells | Active cells |
|---:|---:|---:|
| N64 | 24 | 16 |
| N128 | 48 | 32 |
| N256 | 96 | 64 |

The local outer edge is

```text
12.77724 rg
```

and the common active edge is

```text
6.64838 rg.
```

Thus the frozen exterior is separated from the scientific core by
`8/16/32` nested buffer cells.

The local operator uses the exact production:

- inner one-sided excision flux;
- Kerr-Schild geometry and finite-volume measures;
- quadratic admissible reconstruction;
- exact-max causal Rusanov flux;
- stress transport and relaxation;
- responsive-height vector storage;
- Gauss cell storage and source quadrature;
- radiative cooling.

The stream source is identically zero in this inner radial interval.

The local truncation requires an audit-only outer-face mode:
`frozen_exterior_rusanov`. It evaluates the same production two-state
Rusanov flux against one frozen exterior primitive trace. The default
`CausalFiveFieldDAEContext` remains the physical Roche mode, and all existing
production paths retain that default.

N64 and N128 use their exact committed `t=0.025 s` production anchors and
rates. N256-equivalent uses a shape-preserving prolongation of N128. It is
therefore a resolution preflight, not an independently equilibrated N256
truth state.

## Boundary and temporal method gates

After transforming every local generator to the corresponding full-domain
primitive scale, the active-block defects are:

| Mesh | Relative Frobenius defect | Maximum relative entry defect |
|---:|---:|---:|
| N64 | `3.82868e-5` | `2.13656e-5` |
| N128 | `2.14159e-5` | `1.33300e-5` |

Both pass the `1e-4` gate.

More importantly, direct propagation of the complete local and full
operators gives:

| Mesh | Maximum state-history defect | Maximum rate-history defect | Minimum signed cosine |
|---:|---:|---:|---:|
| N64 | `1.19895e-3` | `5.77525e-4` | `0.99999932` |
| N128 | `2.64062e-4` | `2.12203e-4` | `0.99999997` |

These are far below the spatial phase discrepancies. The negative
refinement result is not produced by the local outer boundary.

The `201/401` matrix-exponential sampling comparison is bitwise equal at all
common times on all three grids. Temporal sampling is not the controlling
error.

## Spatial phase result

The full normalized state/rate histories give:

| Pair | Maximum state L2 difference | Minimum state cosine | Maximum rate L2 difference | Minimum rate cosine |
|---|---:|---:|---:|---:|
| N64/N128 | `1.31902` | `-0.09986` | `2.63624` | `-0.98936` |
| N128/N256 | `1.12709` | `0.27047` | `2.06174` | `-0.64978` |

There is some reduction in amplitude error, but it is much too slow and the
same-time rate direction still reverses.

The active stress-rate signal gives:

| Mesh | Zero crossings (s) | Diagnostic frequency | Envelope log slope |
|---:|---|---:|---:|
| N64 | `0.01721, 0.05216` | unresolved | `-11.20 s^-1` |
| N128 | `0.00580, 0.01894, 0.03380, 0.05367, 0.06194` | `35.62 Hz` | `-5.88 s^-1` |
| N256 | `0.00736, 0.01644, 0.02588, 0.03578, 0.04427` | `54.18 Hz` | `-12.13 s^-1` |

For N128/N256:

```text
maximum zero-crossing relative defect  0.33336
frequency relative defect              0.34257
damping relative defect                0.51542
```

The gates are `0.10/0.10/0.25`. All fail.

The frequency and envelope fits remain diagnostics. A short non-normal
linear transient does not establish a physical eigenmode.

## Radial propagation

The rate-activity centroid begins near the horizon:

```text
N64   2.2349 rg
N128  2.0836 rg
N256  2.1211 rg
```

By `0.125 s` it has moved outward:

```text
N64   4.5513 rg
N128  4.9831 rg
N256  4.8682 rg
```

The corresponding N128/N256 final widths are `0.9222/0.6646 rg`. This
supports the interpretation that a near-horizon phase error propagates
through the inner shell during the WP10c8t horizon.

## Term attribution

The finite directional decomposition reconstructs the stationary generator
action with maximum relative defects:

```text
N64   7.70e-8
N128  3.68e-8
N256  4.97e-8
```

On every mesh the largest initial active contribution is the exact inner
boundary/excision transport. Its peak is the first cell and `log Sigma`
field:

| Mesh | Peak radius | Absolute normalized rate |
|---:|---:|---:|
| N64 | `1.8750 rg` | `75.37` |
| N128 | `1.8371 rg` | `257.23` |
| N256 | `1.8185 rg` | `297.03` |

The N128/N256 restricted absolute defects are controlled by:

```text
inner boundary transport     5.0021
Rusanov transport            0.9983
central perfect transport    0.8616
stress relaxation            0.2575
perfect-fluid geometry       0.2451
```

Cooling contributes only `2.20e-4` to that difference, and the stream
contribution is exactly zero. The numerical blocker is therefore the
boundary-adjacent characteristic/transport phase, not stream injection or a
thermal source clock.

This does not prove that the physical excision boundary is incorrect. In
the continuum all characteristics inside the horizon should leave the
computational domain. It shows that the present discrete one-sided boundary
trace plus near-horizon transport has not reached a common phase law at the
tested resolutions.

## Interpretation and limits

WP10c8v establishes:

1. the WP10c8u N64/N128 phase split can be reproduced by the local evolving
   descriptor;
2. a sufficiently buffered local domain reproduces the committed
   full-domain response;
3. one N256-equivalent local refinement does not enter a convincing
   asymptotic phase-convergence regime;
4. inner boundary and neighboring transport terms dominate the
   nonconvergence.

It does not establish:

- continuum nonconvergence;
- a physical oscillation frequency or damping rate;
- a nonlinear N256 truth trajectory;
- a fixed-`Q` invariant fast measure;
- negligible odd or even averaged forcing;
- a final embedded-patch resolution;
- a reduced architecture.

The N256 anchor is prolonged rather than independently re-equilibrated. The
failure is too large to support the next averaging stage, but the result
must not be promoted into a continuum no-go theorem.

## Decision

WP10c8v does not authorize:

- the WP10c8u fixed-`Q` odd/even averaging experiment;
- elimination of the inner mode;
- an initial-slip correction;
- a scalar or two-component fitted mode;
- reduced macrostepping;
- tide, wind, hot-state, stability, or cycle claims.

An embedded fine inner patch is now the leading numerical candidate, but its
boundary treatment and required resolution are not yet certified.

## Locked next plan: WP10c8w

### Phase 1 — Independent local anchor and pair

Construct an independently consistent local N256-equivalent anchor on the
buffered `1.8-12.78 rg` domain rather than prolongating N128. Restore the
same coarse coordinates and the original mode-0 physical direction, then
re-establish:

- primitive and face consistency;
- mapped and responsive-height storage;
- exact inner and frozen-exterior fluxes;
- local descriptor rank;
- the equal-coordinate fiber correction;
- temporal convergence.

The prolongated WP10c8v state remains a regression input, not the binding
truth state.

### Phase 2 — Inner-excision sensitivity

On the same common exterior interval, test audit-only variants of:

- inner-edge placement safely inside the horizon;
- current one-sided reconstructed trace;
- cell-centered excision trace;
- one additional conservative characteristic-compatible trace.

No variant may replace production merely because it changes the frequency.
A candidate must reduce exterior phase error while retaining:

- purely outgoing inner characteristics;
- exact conservation and ledgers;
- positivity and causal stress gates;
- second-order smooth spatial behavior away from the boundary;
- N128/N256 and subsequent refinement contraction.

### Phase 3 — One higher local refinement or embedded patch

Only after the independent N256 anchor and boundary audit pass, add one
N512-equivalent local refinement or an embedded fine patch covering the
near-horizon cells. Keep the outer buffer at or beyond `12.78 rg`.

Require:

```text
observed state/rate contraction order >= 0.75
same-time signed cosine              >= 0.90
zero-crossing relative defect        <= 0.10
frequency relative defect            <= 0.10
damping relative defect              <= 0.25
full/local history defect            <= 0.02
```

Also require the response outside a fixed radius such as `2.2 rg` to be
insensitive to the chosen excision edge within the scientific reserve.

### Phase 4 — Architecture gate

- Convergent local phase with bounded nonlinear N256/N512 truth error:
  authorize the fixed-`Q` odd/even averaging experiment.
- Convergent only with a permanently finer inner patch: prototype a
  conservative embedded inner patch coupled to the coarse exterior.
- Persistent grid-frequency scaling or boundary-placement sensitivity:
  repair the inner spatial/excision discretization before selecting any
  reduced state.
- Several converged localized modes: retain a local state vector or patch,
  not independently fitted shell moments.

Every later reduced state must undergo a new worst-case exact
equal-coordinate slow-rate fiber audit.

## Machine evidence and reproducibility

Primary evidence:

```text
outputs/tables/causal_inner_phase_spatial_preflight_wp10c8v.json
outputs/tables/causal_inner_phase_spatial_preflight_wp10c8v_arrays.npz
```

Local operator caches:

```text
outputs/checkpoints/causal_five_field_wp10c8v/
```

The machine JSON records:

- parent restart and operator hashes;
- dense WP10c8u pair hashes;
- production DAE and tangent implementation hashes;
- local operator hashes and build costs;
- active generator and history reproduction;
- temporal and spatial phase histories;
- zero crossings, centroid/width, frequency, and damping diagnostics;
- exact term-attribution reconstruction;
- all binding decision gates.

Artifact hashes:

```text
JSON    24f550f71d970d7ce484a81f49994c5eb2867a06bf6cf90466f89af94614fe7c
arrays  862e366652306b799d71bdaa6d2912bec11a4fe4c70a824336846f8b2aaad191
runner  65f4d2d257eb0e81839996682bdf1d5ff586852a3dafede65a7e4dcf297fa49b
```
