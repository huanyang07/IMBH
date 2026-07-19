# Causal Spatial Reconstruction WP10c7g-h Results

Date: 2026-07-19

## Verdict

WP10c7g successfully implements and certifies an optional limited
piecewise-linear reconstruction for the causal Rusanov face flux. The
piecewise-constant operator remains frozen as the default backend.

The method-level result passes:

```text
smooth-PLM finest-pair manufactured order       1.91013
required manufactured order                     1.8
diagnosed-band total-transport tangent order    2.11612
diagnosed-band full-tangent order               2.17191
N32/N64 full-tangent reduction factor           5.23472
required full-tangent reduction factor          5.0
N8 colored/dense Jacobian defect                 1.26995e-10
N16/N32 consistency rank                         245/245, 485/485
```

WP10c7g therefore correctly authorized one bounded N32/N64 trajectory.

WP10c7h is a decisive negative result. All four reconstructed-flux fixed
BDF2 campaigns pass their solver, state, temporal, source, restart, and
ledger contracts, but the spatial response still fails:

```text
N32 raw S32/S64 Delta log(H/R) uncertainty      1.47427e-4
N64 raw S32/S64 Delta log(H/R) uncertainty      1.47243e-4
preferred temporal uncertainty                  2.50000e-4

N32/N64 S64 Delta log(H/R), full domain          0.0446191
maximum temporal confound                        0.000147427
combined spatial plus temporal error             0.0447666
declared spatial gate                            0.005
combined gate fraction                           8.95331

N32/N64 S64 Delta log(H/R), 15-60 rg             0.0214116
diagnosed-band gate fraction                     4.28232
```

The reconstruction is a real improvement, not a certification. Relative to
the WP10c7f piecewise-constant N32/N64 difference of `0.134682`, the new
full-domain maximum is smaller by a factor of `3.018`, and the remaining
inner-band maximum near the old controlling radius is smaller by a factor
of `6.290`. The full-domain maximum has moved to the first N32 cell center
at `1.95316 rg`, where the deliberately unchanged first-order physical
boundary trace now controls. The interior discrepancy near `19.2204 rg`
also remains more than four times the gate, so a boundary-only repair cannot
certify the trajectory.

Therefore:

```text
WP10c7g reconstruction method                    certified
WP10c7h bounded numerical experiment             certified as a stop result
WP10c7h N32/N64 spatial adequacy                  rejected
N128 reconstructed-flux trajectory               not authorized
longer no-tide evolution                         not authorized
next work                                        WP10c7i balance audit
```

No longer duration, tide, wind, stability calculation, hot-state search, or
cycle search is authorized.

## Locked Scope

The implementation adds three selectable spatial backends:

```text
piecewise_constant     frozen default and regression control
plm_unlimited          exact linear-profile oracle
plm_smooth             production audit backend
```

The reconstructed primitive chart is:

```text
(ln Sigma, beta_R, beta_phi, ln T, specific causal stress)
```

Slopes are spacing aware in `ln(R)`. The smooth mode uses a differentiable
same-sign limited slope and one coupled admissibility factor per cell. Both
left and right reconstructed charts feed:

- the conserved-state jump;
- the central physical flux;
- the causal characteristic-speed envelope;
- the Rusanov dissipative term.

The inner and outer physical maps are unchanged one-sided piecewise-constant
traces. No Riemann solver, source term, characteristic boundary condition,
thermodynamic closure, DAE variable/count, temporal-storage equation,
temporal method, or physical parameter is changed.

## Frozen Backend

The context defaults to `piecewise_constant`. Explicitly selecting that mode
reproduces the prior face states and complete DAE residual bitwise.

The prior WP10c7f evidence therefore retains its meaning. Reconstruction is
opt in and does not silently reinterpret earlier checkpoints or reports.

## WP10c7g Reconstruction Oracles

The method audit uses N16/N32/N64/N128 nested logarithmic grids.

For constant primitive charts:

```text
maximum reconstructed left/right defect          0
```

For a profile exactly linear in `ln(R)`, the unlimited interior face
reconstruction is exact to:

```text
maximum absolute defect                          1.77636e-15
```

All primitive and face algebraic maps close exactly:

```text
maximum algebraic-map residual                   0
```

The minimum observed orders on the finest N64/N128 pair are:

| Reconstruction | Minimum order |
|---|---:|
| Unlimited PLM | `1.97187` |
| Smooth PLM | `1.91013` |

The minimum smooth-PLM order over every pair, including the pre-asymptotic
N16/N32 pair, is `1.64746`. The locked method gate applies to the finest
pair, while the complete order sequence remains in the machine evidence.

## Widened Jacobian

Piecewise-linear face reconstruction expands the interior primitive
dependency stencil to four cells. The exact sparsity and coloring metadata
were updated accordingly.

At N8:

```text
Jacobian dimensions                              125 x 125
sparsity nonzeros                                1655
color count                                      23
maximum omitted/colored/directional defect       1.26995e-10
maximum cross-step finite-difference spread      5.03395e-4
```

The parity defect passes the `2e-8` gate, and the finite-difference step
spread passes the `5e-3` gate. The smooth limiter is therefore compatible
with the colored finite-difference Jacobian for this bounded state.

## Common-State Tangent Audit

The exact source-compatible continuum profile is independently sampled on
N16/N32/N64. The DAE-consistent tangent is decomposed into central
transport, Rusanov transport, flux closure, and source components.

Within the previously diagnosed `15-60 rg` band, smooth PLM gives:

| Tangent component | N16/N32 to N32/N64 order |
|---|---:|
| Central transport | `1.95826` |
| Rusanov transport | `3.03972` |
| Total transport | `2.11612` |
| Full `d log(H/R)/dt` | `2.17191` |

The N32/N64 discrepancy reduction relative to piecewise constant is:

| Tangent component | Reduction factor |
|---|---:|
| Central transport | `2.29809` |
| Rusanov transport | `29.1041` |
| Total transport | `4.78601` |
| Full `d log(H/R)/dt` | `5.23472` |

The predeclared gates require total-transport and full-tangent order at
least `1.8`, plus at least a factor-five reduction in the full observable
tangent. They pass.

The full-domain PLM orders remain lower because the physical boundary traces
are intentionally first order:

```text
full tangent order                              1.34935
total transport order                           1.19500
```

These values were retained as boundary telemetry and were not promoted to a
full-domain method certification. WP10c7h was required to determine whether
the boundary contribution remained practically important.

The full consistency systems remain full rank:

```text
N16                                             245/245
N32                                             485/485
maximum tangent/decomposition defect            9.28230e-11
```

## Fresh WP10c7h Histories

Old piecewise-constant checkpoints are not used as reconstructed-operator
initial states. Each mesh instead:

1. samples the same analytic source-compatible continuum profile;
2. rebuilds exact primitive and reconstructed face maps;
3. computes its own DAE-consistent tangent predictor;
4. takes one BDF1 startup step to create complete BDF2 history.

The N32 and N64 initial state hashes are:

```text
N32  7a8109ce8ac9c674e2155eb449cbc5fae9a0d19df7f0c702373148f43c73f322
N64  aeab263db3b400e79bd2eb297fc3dbc129a1a90542bac1aae8d4338dfa2082ea
```

The initial tangent reconstruction defects are below `9.29e-11`. Source
restriction closes at `1.72942e-16`, against `5e-13`.

## Fixed BDF2 Campaigns

WP10c7h runs N32 and N64 at S32 and S64 over the exact WP10c7f duration:

```text
extension                                        1.537457597966907e-2 s
physics                                          exact C2 stream
tide/wind                                        off/off
temporal method                                  one BDF1, then fixed BDF2
```

All 192 fixed steps pass:

| Mesh/rung | Steps | BDF1/BDF2 | Max residual | Physical ledger | Jacobians | Function evaluations |
|---|---:|---:|---:|---:|---:|---:|
| N32 S32 | `32` | `1/31` | `4.25e-13` | `1.529e-4` | `32` | `1664` |
| N32 S64 | `64` | `1/63` | `2.13e-13` | `3.815e-5` | `64` | `3328` |
| N64 S32 | `32` | `1/31` | `2.73e-13` | `1.574e-4` | `32` | `1664` |
| N64 S64 | `64` | `1/63` | `9.78e-12` | `3.921e-5` | `64` | `3326` |

The campaign opts into modified Newton with one finite-difference Jacobian
reused across the step's Newton corrections. A failed reused-matrix line
search forces a fresh Jacobian. The default configuration still rebuilds
every Newton iteration, preserving prior campaign behavior.

Every checkpoint reloads bitwise. No initial or final state requires
admissibility rescaling:

```text
minimum admissibility factor                    1.0
admissibility-rescaled cells                    0
```

## Temporal Separation

Raw S32/S64 thickness-response uncertainties are:

| Mesh | Full domain | `15-60 rg` |
|---|---:|---:|
| N32 | `1.47427e-4` | `1.73538e-5` |
| N64 | `1.47243e-4` | `1.73815e-5` |

Both full-domain values pass the locked `5e-4` maximum and the preferred
`2.5e-4` target. The N32/N64 spatial mismatch is about `303` times the
maximum temporal confound.

Temporal accuracy is not the failing mechanism.

## Spatial Result

N64 cell responses are restricted exactly onto N32 Kerr-Schild control
volumes. At S64:

| Response | Full-domain maximum | `15-60 rg` maximum |
|---|---:|---:|
| `Delta log(H/R)` | `0.0446191` | `0.0214116` |
| `Delta log T` | `0.0110375` | `0.00532304` |
| `Delta log integrated pressure` | `0.0887592` | `0.0426576` |
| `Delta log specific internal energy` | `0.0892432` | `0.0428462` |
| `Delta log surface density` | `0.000479126` | `0.000174504` |

The full-domain thermodynamic maxima occur at `1.95316 rg`. The interior
energy, pressure, temperature, and thickness maxima occur together at
`19.2204 rg`, while the surface-density discrepancy is much smaller.

This retains the WP10c7e-f interpretation:

1. the principal response difference is thermodynamic, not mass loading;
2. responsive height amplifies the energy/pressure difference into `H/R`;
3. PLM substantially reduces the inherited interior face-transport error;
4. the unchanged first-order physical boundary trace becomes visible after
   that reduction;
5. an interior transport-versus-geometry balance defect still remains.

Source restriction, temporal error, nonlinear convergence, conservation,
checkpoint history, and admissibility rescaling are all too small to explain
the failure.

## Why N128 Is Not Authorized

The reconstructed trajectory has only one spatial pair, so no new asymptotic
trajectory order is claimed. More importantly, the method audit already
shows that:

- the full-domain order is boundary limited;
- the interior response remains `4.28` times the gate;
- the full-domain response plus temporal uncertainty remains `8.95` times
  the gate.

N128 would mix a known boundary-order defect with a still-large interior
balance defect. It would not test a complete candidate method. Uniform
refinement remains closed.

## Locked WP10c7i

The next package is a method-level balance-preserving perturbation audit. It
must not force the source-compatible datum to be stationary: that datum has
a real nonzero physical tangent.

The safe construction is:

```text
R_balanced(y) =
    R_baseline_high_order
    + [R_PLM(y) - R_PLM(y_baseline)]
```

Here `R_baseline_high_order` must be an independently converged,
conservative representation of the physical residual of the analytic
baseline. It must not be zeroed and must not merely reuse the same coarse
PLM residual. Exact face evaluation and high-order cell quadrature should
provide its transport, geometric, cooling, and stream terms.

WP10c7i must:

1. construct and converge the nonzero baseline residual oracle at
   N16/N32/N64/N128;
2. preserve exact conservative face telescoping and source moments;
3. use one-sided second-order baseline/perturbation traces with the unchanged
   physical boundary maps;
4. reduce exactly to the PLM operator when its baseline oracle is replaced
   by the discrete PLM baseline residual;
5. recover the oracle exactly at zero perturbation;
6. demonstrate at least order `1.8` for smooth perturbation transport and
   the full tangent in both the full domain and `15-60 rg`;
7. preserve positivity, causal characteristics, algebraic closure,
   descriptor/consistency rank, and dense/colored Jacobian parity;
8. expose the baseline correction explicitly in the physical ledger rather
   than hiding it as an untracked source;
9. reduce the remaining PLM N32/N64 full-domain tangent discrepancy by at
   least a factor of `10`;
10. reduce the remaining PLM `15-60 rg` tangent discrepancy by at least a
    factor of `5`.

Only if every method-level gate passes may WP10c7j repeat the same bounded
N32/N64 S32/S64 trajectory. A failed balance audit should stop before any
trajectory and move to a separately scoped local-refinement design.

## Evidence

WP10c7g machine summary:

```text
outputs/tables/causal_spatial_reconstruction_wp10c7g.json
SHA-256  fd6f08b39a292ea87a6dda63e6fe7c6131b6b5246bb6ea90d5e160c97b6c5e0d
```

WP10c7g compact arrays:

```text
outputs/tables/causal_spatial_reconstruction_wp10c7g_arrays.npz
SHA-256  63df124bddb6791e4c4265c61ada0bf3173e68ee46eca7d4dee137f63ea7db02
```

WP10c7h machine summary:

```text
outputs/tables/causal_reconstructed_flux_trajectory_wp10c7h.json
SHA-256  a3e945ced1ee5edc219d892dcad59b4a1f9410cf2bde11bc77879b86e87a8662
```

WP10c7h compact arrays:

```text
outputs/tables/causal_reconstructed_flux_trajectory_wp10c7h_arrays.npz
SHA-256  b4b3881d2a8b1f1fbf723fee9be7815eded73d161aa9299113d060c60d654eff
```

WP10c7h fixed checkpoints:

```text
N32 S32  02191b7703299639bbc47fc6f46c75f83c084f04b092f53413b760b25ba77917
N32 S64  ecb0e1e44682e6311f2a2dbd39f7a50b566372d7dcad4f6e981f89175d025985
N64 S32  6f83a9cf077672ce5d926b04064c7a0325b1c326e4642894b98f80cf874fb09e
N64 S64  29c02516f339960e18e2d1ca4036fea8bdd07059ae97740f8941da3cad979ebe
```

Runtime artifacts remain ignored under the repository artifact policy.

## Verification

Before the atomic commit:

```text
reconstruction/DAE/BDF focused tests             50 passed
full repository suite                            540 passed, 4 subtests passed
WP10c7g manufactured/operator audit              passed
WP10c7g dense/colored Jacobian audit              passed
WP10c7g common-state tangent audit                passed
WP10c7h N32/N64 S32/S64 trajectories             completed
WP10c7h temporal/source/state/ledger gates        passed
WP10c7h spatial gate                             rejected as reported
checkpoint round trips                           bitwise
```

## Reproduction

Run the method-level audit:

```text
PYTHONPATH=src python3 \
  scripts/run_causal_spatial_reconstruction_audit_wp10c7g.py
```

Run or resume the four fixed campaigns and aggregate:

```text
PYTHONPATH=src python3 \
  scripts/run_causal_reconstructed_flux_trajectory_wp10c7h.py --force
```

Rebuild only the aggregate from existing checkpoints:

```text
PYTHONPATH=src python3 \
  scripts/run_causal_reconstructed_flux_trajectory_wp10c7h.py
```
