# Causal Matched BDF2 WP10c7d Results

Date: 2026-07-18

## Verdict

WP10c7d produces a split result:

```text
matched N32 adaptive BDF2 temporal controller    certified
N16/N32 bounded spatial response                 failed
```

The N32 temporal result is strong:

```text
fixed observable orders                 1.99503-2.00670
fixed physical-ledger orders            1.99480-2.00126
S32/S64 reference uncertainty/gate      0.07602 maximum
adaptive plus reference error/gate      0.11411 maximum
adaptive physical-ledger defect         1.18980e-4
adaptive/fixed-S64 Jacobian ratio        0.284375
adaptive split replay                    bitwise
```

The exact-common-time spatial response is not close:

```text
maximum N16/N32 Delta log(H/R) difference   0.612925
declared spatial gate                        0.005
gate factor                                122.585
maximum-difference radius                   20.856 r_g
```

Therefore the first-order temporal bottleneck is closed through N32, but the
current `0.0154 s` trajectory is not spatially certified. N64/N128 production,
longer evolution, tide, wind, stability, hot-state, and cycle work remain
blocked.

This is not a physical instability result. It is a numerical statement that
N16 and N32 follow materially different semidiscrete trajectories over this
horizon.

## Locked Problem

Every N32 run starts from:

```text
checkpoint       causal_wp10c5q_N032_final.npz
elapsed time     8.484232672865630e-4 s
stream           exact circularized C2 regression stream
physics          no tide, no wind
duration         1.537457597966907e-2 s
```

WP10c7d retains the WP10c7c:

- one-domain ingoing-Kerr-Schild five-field DAE;
- source, inner characteristic boundary, and Roche outer boundary;
- responsive-height gas+radiation storage;
- sparse equilibrated Newton solve;
- BDF1 startup and variable-step BDF2;
- quadratic three-state predictor and one ordinary corrector;
- predictor scale, local gates, timestep factors, and maximum timestep;
- periodic full-versus-two-half BDF2 audits;
- dual discrete/physical ledgers;
- complete restart contract;
- immutable v1 observable gates.

No controller parameter is retuned between N16 and N32.

## N32 Fixed Reference

The N32 reference uses one BDF1 startup step followed by fixed BDF2:

| Subdivisions | Timestep (s) | Max residual | Max discrete ledger | Physical ledger | Jacobians |
|---:|---:|---:|---:|---:|---:|
| 16 | `9.60911e-4` | `2.633e-12` | `2.754e-12` | `8.55680e-4` | 80 |
| 32 | `4.80455e-4` | `1.320e-12` | `1.355e-12` | `2.13472e-4` | 160 |
| 64 | `2.40228e-4` | `6.619e-13` | `1.124e-12` | `5.33214e-5` | 320 |

All 112 fixed steps pass every nonlinear, algebraic, causal, optical-depth,
Roche, state-change, and ledger gate. Every saved endpoint reloads bitwise.

### Observable order

| Observable | S16-to-S32 | S32-to-S64 | Order |
|---|---:|---:|---:|
| Total cooling | `2.47388e-4` | `6.18736e-5` | `1.99938` |
| Cooling outside `6 r_g` | `1.11119e-4` | `2.78030e-5` | `1.99879` |
| Inner accretion rate | `1.64212e-5` | `4.08627e-6` | `2.00670` |
| Maximum log `H/R` profile | `6.08309e-4` | `1.52030e-4` | `2.00044` |
| Integrated conserved state | `3.19620e-7` | `8.01805e-8` | `1.99503` |
| Baseline-scaled full state | `1.53190e-4` | `3.82879e-5` | `2.00037` |

All orders satisfy the locked `1.7-2.3` interval.

The raw S32-to-S64 difference is used as the selected-reference uncertainty.
Its largest fraction of a full immutable gate is:

```text
0.0760152
```

This is below the locked `0.25` allowance. Equivalently, it consumes `0.3041`
of that allowance.

### Physical-ledger order

The five component orders are:

```text
1.99667
1.99960
1.99955
1.99480
2.00126
```

The S64 maximum relative physical-ledger defect is `5.33214e-5`, below the
unchanged `1e-3` gate.

## N32 Adaptive Controller

The unchanged controller reaches the exact target in:

```text
accepted steps          15
accepted BDF2 steps     14
rejected attempts        0
independent audits       4
minimum timestep         1.50142e-5 s
maximum timestep         1.92182e-3 s
```

The first eight accepted timesteps match N16 exactly. N32 then remains at the
shared maximum timestep longer and lands in fewer total steps. This
mesh-dependent history is permitted because the endpoint is judged against
an independent N32 reference.

The largest accepted predictor-estimator normalized error is:

```text
0.0851307
```

The largest independent full-versus-two-half audit error is:

```text
0.00262877
```

Both remain far below one.

## Endpoint Accuracy

The adaptive endpoint is compared with N32 fixed S64. Raw S32-to-S64
uncertainty is then added without cancellation.

| Observable | Adaptive-to-S64 / gate | Reference / gate | Combined / gate |
|---|---:|---:|---:|
| Total cooling | `0.01741` | `0.06187` | `0.07928` |
| Cooling outside `6 r_g` | `0.00342` | `0.02780` | `0.03123` |
| Inner accretion rate | `0.00699` | `0.00409` | `0.01108` |
| Maximum log `H/R` profile | `0.03809` | `0.07602` | `0.11411` |
| Integrated conserved state | `0.00010` | `0.00008` | `0.00018` |
| Baseline-scaled full state | `0.02744` | `0.01914` | `0.04659` |

Every combined gate passes. The controlling `H/R` result consumes only
`0.11411` of its gate.

## Ledgers And Restart

The maximum discrete BDF defect over all adaptive trial solves is:

```text
1.02049e-11
```

The cumulative physical-ledger component defects are:

```text
1.05814e-5
4.45878e-6
1.32740e-6
7.02105e-6
1.18980e-4
```

The maximum remains below `1e-3`.

The trajectory is split after accepted step three. Split reload, final reload,
and continued endpoint replay are all bitwise identical.

## Work

The adaptive N32 campaign, including four independent two-half-step audits,
uses:

```text
23 implicit solves
3,390 residual evaluations
91 Jacobians/Newton iterations
```

Fixed N32 S64 uses:

```text
64 implicit solves
11,904 residual evaluations
320 Jacobians/Newton iterations
```

The adaptive Jacobian fraction is:

```text
91 / 320 = 0.284375
```

This is comfortably below the locked `0.75` usefulness gate.

## Spatial Hard Stop

N16 and N32 begin at the exact same physical time and land at:

```text
0.01622299924695563 s
```

For each mesh, `log(H/R)` is reconstructed from cell centers onto the same
129-point log-radius grid with one-cell linear edge extrapolation. The initial
profile is subtracted before the meshes are compared.

The result is:

```text
maximum response difference        0.6129252312
RMS response difference            0.2180897085
gate                               0.005
```

The maximum occurs near:

```text
R                                  20.8559 r_g
N16 Delta log(H/R)                 -0.843023
N32 Delta log(H/R)                 -0.230098
```

Both final states separately pass all physical-state gates:

```text
N16 max/min H/R                    0.100453 / 0.008490
N32 max/min H/R                    0.099739 / 0.008488
```

Thus the failure is not caused by either trajectory crossing the existing
thickness or positivity bounds. It is a large spatial response discrepancy
in the inner disk over the longer bounded horizon.

The earlier short-horizon work already classified the face transport as
ordinary first-order spatial truncation. WP10c7d does not justify changing
the operator, but it proves that N16 is not an adequate physical mesh for
this longer trajectory.

## Classification

WP10c7d establishes:

```text
N32 fixed BDF2 second-order convergence       certified
N32 fixed temporal reference                  certified
N32 adaptive BDF2 temporal accuracy           certified
N32 adaptive conservation and restart         certified
N16/N32 temporal-controller transfer          certified
N16/N32 spatial response at 0.0154 s          rejected
physical relaxation or stability              not tested
```

The temporal method is no longer the active blocker. Spatial resolution is.

## Evidence

Machine output:

```text
outputs/tables/causal_matched_bdf2_wp10c7d_N032.json

SHA-256
8bd2783678370073b15e43708a44097c7f1c526f253ce86be16c85c3718a4d96
```

N32 fixed checkpoints:

```text
S16  663e5653bfda6da0c6d792cf9c48701161b6f5e7553fb84219db880a11b06f60
S32  8c94d90b1e6ead91dc7fe55fc9f18762b234779095fcd13b6a3166f5ae85d45e
S64  fe07254152c37b32d39277edc50bcaad61c2b775c8f9e8784aa919664a95bbfb
```

Adaptive checkpoints:

```text
split a4f5953f5e96e4c6e9053586d31ba0f6f06f13c9d563c6c1b08c4865c0fc9805
final 2ae27d6dd683275471e1b1ffef93c752fc987f09e9f2a1c05421229125a62af7
```

Runtime artifacts remain ignored under the artifact policy.

## Recommended Next Gate

Do not extend physical duration or add physics.

The next bounded package should:

1. localize the N16/N32 response discrepancy by radius, equation block, face
   term, and characteristic family at the common endpoint;
2. verify that it is the expected amplification of the already identified
   first-order face truncation rather than a reconstruction or comparison
   defect;
3. predeclare one N64 adaptive run at the identical horizon only if that
   localization passes;
4. pair the N64 run with a tighter N64 temporal audit so temporal uncertainty
   stays well below the spatial gate;
5. compare N32/N64 response contraction and stop before N128 if it is not
   consistent with the established spatial order.

Do not change the flux operator, loosen `0.005`, or launch N128 merely because
N64 can run. A separate authorization is required after the localization
audit.

## Verification

Before the atomic commit:

```text
BDF/controller/evolution tests    20 passed
WP10c7d machine campaign          completed
N32 temporal controller gate      passed
N16/N32 spatial response gate     failed as reported
full repository suite             525 passed, 4 subtests passed
```

## Reproduction

```text
PYTHONPATH=src python3 scripts/run_causal_matched_bdf2_wp10c7d.py
```
