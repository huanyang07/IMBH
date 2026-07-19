# Causal N64 Contraction WP10c7f Results

Date: 2026-07-19

## Verdict

WP10c7f completes the single authorized N64 fixed-BDF2 contraction
diagnostic. The N64 trajectories are temporally accurate and satisfy every
solver, state, and ledger contract, but the N32/N64 spatial response remains
far outside the declared gate:

```text
N64 raw S32/S64 Delta log(H/R) uncertainty    1.53598e-4
locked maximum temporal uncertainty           5.00000e-4
preferred temporal uncertainty                2.50000e-4

N16/N32 Delta log(H/R) difference              0.613215
N32/N64 Delta log(H/R) difference              0.134682
observed spatial order                         2.18684
declared spatial gate                          0.005
N32/N64 gate fraction                          26.9363
```

The response contracts by a factor of `4.5531`, which is much faster than
the minimum `p=0.75` stop rule. It is nevertheless nowhere near spatial
certification. If the measured order persisted, the expected N64/N128
difference would be about `0.02958`, or `5.92` times the gate. N128 would
need one-level order `4.75` to pass directly.

Therefore:

```text
WP10c7f diagnostic                         certified
N64 temporal adequacy                     certified
N32/N64 spatial gate                      rejected
automatic N128 authorization              rejected
uniform-refinement path                   stopped
next work                                 WP10c7g spatial-operator upgrade
```

No N128 run, longer duration, tide, wind, stability calculation, hot-state
search, or cycle search is authorized.

## Locked Scope

The campaign uses:

```text
N32 initial checkpoint       WP10c5q
N64 initial checkpoint       WP10c5s duration
N32 fixed endpoint           WP10c7d S64
N64 fixed endpoints          WP10c7f S32 and S64
initial time                 8.484232672865630e-4 s
extension                    1.537457597966907e-2 s
final time                   1.622299924695563e-2 s
physics                      exact C2 regression stream, no tide, no wind
temporal method              one BDF1 startup step, then fixed BDF2
```

The N64 state is independently generated on the N64 mesh. It is not
interpolated from N32. N32 and N64 share the exact physical start and end
times.

WP10c7f does not:

- alter the finite-volume transport or source operator;
- change the BDF method, nonlinear tolerances, or state gates;
- run an adaptive trajectory;
- relax the `0.005` spatial gate;
- evolve N128;
- extend the physical horizon;
- add a physical source.

## Grid And Source Contract

The N32 and N64 logarithmic grids are exactly nested:

```text
refinement ratio                               2
N32 edges == N64 edges[::2]                    bitwise
maximum scaled stream restriction defect       1.72942e-16
stream restriction gate                        5.0e-13
```

The exact compact stream therefore cannot explain the N32/N64 response
difference.

## N64 Fixed Trajectories

Both fixed trajectories reach the exact target with all state gates passing:

| Quantity | S32 | S64 |
|---|---:|---:|
| Completed steps | `32` | `64` |
| BDF1/BDF2 steps | `1/31` | `1/63` |
| Timestep (s) | `4.80455e-4` | `2.40228e-4` |
| Maximum scaled residual | `3.00e-13` | `9.90e-12` |
| Maximum algebraic residual | `5.69e-15` | `1.15e-13` |
| Maximum discrete-ledger defect | `1.15e-12` | `3.73e-11` |
| Cumulative physical-ledger defect | `1.80e-4` | `4.48e-5` |
| Maximum Newton iterations | `5` | `5` |
| Jacobian evaluations | `160` | `318` |
| Function evaluations | `5,952` | `11,830` |

The S64 state retains:

```text
maximum H/R                              0.0994264
minimum scattering optical depth        18.7909
inner incoming characteristics          0
outer incoming characteristics          2
outer Roche channel                      closed
maximum inner light-cone excess          0
```

Both checkpoints reload bitwise.

## Temporal Gate

The raw S32/S64 endpoint differences are:

| Observable | Raw difference | Gate fraction |
|---|---:|---:|
| Total cooling | `5.73618e-5` | `0.05736` |
| Cooling outside `6 r_g` | `2.27594e-5` | `0.02276` |
| Inner accretion rate | `4.11550e-6` | `0.00412` |
| `Delta log(H/R)` profile | `1.53598e-4` | `0.30720` |
| Integrated conserved state | `6.65842e-8` | `6.66e-5` |
| Baseline-scaled full state | `3.86948e-5` | `0.01935` |

The controlling thickness uncertainty passes both:

```text
1.53598e-4 < 5.0e-4     locked maximum
1.53598e-4 < 2.5e-4     preferred target
```

Temporal uncertainty is about `876` times smaller than the measured
N32/N64 thickness difference. It cannot explain the spatial result.

## Exact Spatial Comparison

N64 cell responses are restricted onto N32 control volumes using exact
Kerr-Schild measures. The thickness-response norms are:

```text
maximum absolute difference          0.1346816
measure-weighted L1 difference       0.00747560
measure-weighted L2 difference       0.0218409
RMS difference                       0.0587690
peak radius                          19.2204 r_g
```

Excluding the first and last two N32 cells leaves the maximum unchanged.
The response is not a boundary artifact.

At the peak:

```text
N32 Delta log(H/R)                    -0.232039
restricted N64 Delta log(H/R)         -0.0973574
difference                            -0.134682
```

The associated thermodynamic differences remain dominant:

| Response | Maximum difference | Peak (`r_g`) |
|---|---:|---:|
| `Delta log T` | `0.0329176` | `19.2204` |
| `Delta log Sigma` | `0.00350136` | `16.3242` |
| `Delta log integrated pressure` | `0.265969` | `19.2204` |
| `Delta log specific energy` | `0.269554` | `19.2204` |

Direct comparisons on the exactly coincident native faces also retain large
central, Rusanov, and total numerical-flux response differences. This
agrees with the WP10c7e transport classification and is independent of cell
profile restriction.

## Contraction And N128 Decision

Using the exact WP10c7e N16/N32 difference:

```text
D_16,32 = 0.6132147678
D_32,64 = 0.1346815750

p = log2(D_16,32 / D_32,64) = 2.1868399433
```

The measured contraction is real and comfortably exceeds the `p=0.75`
minimum. It does not make N128 useful as a direct certification run:

```text
measured contraction factor                    4.55307
projected D_64,128 at measured order            0.0295804
projected N64/N128 gate fraction                5.91608
order required for N64/N128 to pass directly    4.75148
```

Only one coarse-pair order is available, so the projection is not a
continuum claim. It is a resource decision. Even optimistic persistence of
the measured order predicts that N128 fails the unchanged gate by almost a
factor of six. A costly N128 run would not be a plausible direct
certification step.

Uniform refinement stops here.

## Locked WP10c7g

The next package is an operator-level second-order spatial reconstruction
audit. It must remain separate from physical evolution.

1. Add an optional limited piecewise-linear reconstruction for the cell
   primitive states used by the causal Rusanov face flux.
2. Preserve the exact central flux, causal maximum-speed envelope,
   positivity, source moments, boundary characteristic maps, and
   increment-primary DAE structure.
3. Update the exact Jacobian sparsity and color groups for the widened
   reconstruction stencil.
4. Retain the present first-order operator as a frozen regression backend.
5. Demonstrate constant-state preservation and second-order convergence on
   smooth manufactured profiles.
6. Repeat the common-state N16/N32/N64 operator audit. Require the Rusanov
   and full face-transport terms to converge at order at least `1.8`, while
   central and source terms retain their existing contracts.
7. Recheck descriptor/consistency rank, algebraic closure, positivity,
   causal boundaries, and exact source restriction.
8. Do not launch a disk trajectory until all method-level gates pass.
9. If they pass, repeat the bounded fixed-S64 N32/N64 comparison with raw
   temporal uncertainty below `5e-4`.

If piecewise-linear reconstruction does not remove the large
transport-versus-geometry imbalance, a separate well-balanced
perturbation-from-baseline discretization should be considered. It must not
be combined with the first reconstruction commit.

## Evidence

Machine summary:

```text
outputs/tables/causal_n64_contraction_wp10c7f.json
SHA-256  58828f66dbb6343b496c7f65f6e58a50e16730d0e8ecfd437829e68cd9c86237
```

Compact arrays:

```text
outputs/tables/causal_n64_contraction_wp10c7f_arrays.npz
SHA-256  77d32ea4902918e2911a63ddaca6d51617ac1114756891dc1dae0d72814d2ea4
```

N64 fixed checkpoints:

```text
S32  19fff9e5f6c7a9af5a3913f33cf81f1a38ada3f25d22aed27830393113f541eb
S64  549aef7c2265890fc82265af79d32d7da06e16bfec5b53c285e90aaaaf8ba270
```

Runtime artifacts remain ignored under the repository artifact policy.

## Verification

Before the atomic commit:

```text
spatial/DAE/BDF focused tests         65 passed
full repository suite                 531 passed, 4 subtests passed
N64 fixed S32/S64 trajectories        completed
N64 temporal uncertainty gate         passed
exact restriction/source gates        passed
state and ledger contracts            passed
checkpoint round trips                bitwise
```

## Reproduction

Run each expensive fixed rung independently:

```text
PYTHONPATH=src python3 scripts/run_causal_n64_contraction_wp10c7f.py \
  --subdivisions 32 --force

PYTHONPATH=src python3 scripts/run_causal_n64_contraction_wp10c7f.py \
  --subdivisions 64 --force
```

Then aggregate without recomputation:

```text
PYTHONPATH=src python3 scripts/run_causal_n64_contraction_wp10c7f.py
```
