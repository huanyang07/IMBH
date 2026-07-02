# Codex Stream Source-Fraction Tangent Results

Date: 2026-07-02

## Scope

This update implements and tests GPT's source-fraction tangent predictor:

```text
J_z dz/df_s = -F_f_s
```

where `f_s` is the stream source fraction. The tests use the residual-remeshed
`N=768` stream-fed branch at:

```text
Mdot_inner/Edd = 2
Rout = 300 rg
Rinj/Rout = 0.80
stream_torque_delta_l_fraction = +0.005
no wind
no stream heating
```

## Code Added

New audit script:

```text
scripts/run_standard_slim_stream_source_tangent_audit.py
```

Updated continuation script:

```text
scripts/run_standard_slim_stream_mass_annulus_scan.py
```

The tangent predictor is opt-in:

```text
IMBH_STANDARD_SLIM_STREAM_MASS_USE_TANGENT_PREDICTOR=1
```

It can be used together with the guarded secant predictor. The continuation
script evaluates current, secant, and tangent seeds and chooses the seed with
the lowest full residual.

## Predictor Audit

Output:

```text
outputs/tables/standard_slim_stream_source_tangent_initial_N768.md
outputs/tables/standard_slim_stream_source_tangent_tangent_polish_N768.md
```

Initial residual comparison:

| step | current seed | secant seed | tangent seed |
|---|---:|---:|---:|
| 0.800 -> 0.805 | 9.994e-2 | 3.063e-2 | 4.281e-3 |
| 0.805 -> 0.808585 | 2.252e-1 | 1.336e-1 | 1.068e-2 |
| 0.808585 -> 0.810 | 1.033e-1 | 7.438e-2 | 2.701e-3 |

The tangent predictor is decisively better than both current-state and secant
seeds in all three audited transitions.

Tangent-only polish results:

| step | tangent initial | final residual | dominant | nfev | result |
|---|---:|---:|---|---:|---|
| 0.800 -> 0.805 | 4.281e-3 | 1.397e-6 | outer_omega | 10 | accepted |
| 0.805 -> 0.808585 | 1.068e-2 | 9.901e-7 | outer_omega | 4 | accepted |
| 0.808585 -> 0.810 | 2.701e-3 | 4.152e-7 | outer_omega | 3 | accepted |

## Continuation With Tangent Predictor

The integrated adaptive continuation now reaches beyond the previous
`f_s≈0.8086` front.

### 0.808585 -> 0.810

Output:

```text
outputs/tables/high_mdot_stream_source_tangent_adaptive_0p808585_to0p81_N768.md
outputs/checkpoints/high_mdot_stream_source_tangent_adaptive_0p808585_to0p81_N768/
```

Result:

```text
f_s = 0.810
initial residual = 2.701e-3
final residual = 4.152e-7
dominant = outer_omega
nfev = 3
Mdot_outer/Mdot_inner = 0.193049
f_adv_global = 0.2039
f_adv_inner = 0.09517
Lrad/LEdd = 0.8673
max H/R = 0.2269
Rson = 4.66 rg
```

### 0.810 -> 0.820

Output:

```text
outputs/tables/high_mdot_stream_source_tangent_adaptive_0p81_to0p82_N768.md
outputs/checkpoints/high_mdot_stream_source_tangent_adaptive_0p81_to0p82_N768/
```

All 0.002 steps were strict anchors:

| f_s | initial residual | final residual | nfev | Mdot_outer/Mdot_inner |
|---:|---:|---:|---:|---:|
| 0.812 | 5.277e-3 | 4.944e-7 | 3 | 0.191056 |
| 0.814 | 5.274e-3 | 4.681e-7 | 3 | 0.189064 |
| 0.816 | 5.319e-3 | 4.628e-7 | 3 | 0.187071 |
| 0.818 | 5.378e-3 | 4.529e-7 | 3 | 0.185079 |
| 0.820 | 5.467e-3 | 4.440e-7 | 3 | 0.183086 |

### 0.820 -> 0.830

Output:

```text
outputs/tables/high_mdot_stream_source_tangent_adaptive_0p82_to0p85_N768.md
outputs/checkpoints/high_mdot_stream_source_tangent_adaptive_0p82_to0p85_N768/
```

Two 0.005 steps succeeded:

| f_s | initial residual | final residual | nfev | Mdot_outer/Mdot_inner |
|---:|---:|---:|---:|---:|
| 0.825 | 3.487e-2 | 8.566e-7 | 4 | 0.178105 |
| 0.830 | 3.771e-2 | 3.028e-7 | 5 | 0.173124 |

The next 0.005 step to `f_s=0.835` had a larger seed residual,
`5.985e-2`, and the Newton corrector became too slow. The run was interrupted
manually rather than treated as a clean physical failure.

### 0.830 -> 0.83225

Output:

```text
outputs/tables/high_mdot_stream_source_tangent_adaptive_0p83_to0p84_smallstep_N768.md
outputs/tables/high_mdot_stream_source_tangent_adaptive_0p832_to0p833_tinystep_N768.md
outputs/checkpoints/high_mdot_stream_source_tangent_adaptive_0p83_to0p84_smallstep_N768/
outputs/checkpoints/high_mdot_stream_source_tangent_adaptive_0p832_to0p833_tinystep_N768/
```

Results:

| f_s | initial residual | final residual | nfev | Mdot_outer/Mdot_inner | note |
|---:|---:|---:|---:|---:|---|
| 0.831 | 2.409e-3 | 2.571e-7 | 4 | 0.17213 | strict |
| 0.832 | 3.221e-3 | 2.546e-7 | 10 | 0.17113 | strict |
| 0.83225 | 5.542e-4 | 1.652e-6 | 25 | 0.17088 | strict, but max iterations reached |

The next tiny step toward `f_s=0.8325` had an even smaller initial residual
of about `5e-5`, but the corrector again became slow and was interrupted.

## Interpretation

The tangent predictor solves the predictor problem. It moves the continuation
front from:

```text
f_s ≈ 0.8086
```

to:

```text
f_s ≈ 0.83225
```

on the residual-remeshed `N=768` branch.

The old `f_s≈0.8086` barrier was therefore not a physical endpoint. It was a
combination of source-tail remeshing and predictor failure.

The new bottleneck appears near:

```text
f_s ≈ 0.832--0.833
Mdot_outer/Mdot_inner ≈ 0.171
```

This bottleneck is different. The tangent seed can be excellent
(`initial residual ~5e-5`), but the Newton/Jacobian corrector still becomes
very slow. The limiting residual is again associated with the outer/source-tail
energy block, not the sonic point.

So the current best diagnosis is:

```text
The predictor problem is largely fixed.
The remaining wall is a corrector/Jacobian/outer-boundary closure problem.
```

The branch remains mildly advective:

```text
f_adv_global ≈ 0.204
f_adv_inner ≈ 0.095
max H/R ≈ 0.227
Rson ≈ 4.66 rg
```

This is still not the hot Mdot=3--5 stream-fed branch.

## Recommended Next Step

Implement the next two GPT items in this order:

1. Add cost-aware adaptive control:
   - shrink the step when `nfev` is high even if accepted;
   - record tangent seed residual, corrector iterations, and final dominant residual;
   - stop or shrink when a tiny initial residual still produces a slow corrector.

2. Add soft/Robin outer angular closure:
   - test at `f_s=0.82`, `0.83`, and `0.83225`;
   - determine whether the new corrector wall is hard-boundary closure rather than a physical branch feature.

Pseudo-arclength should wait until the soft/Robin closure and corrector-cost
diagnostics are in place.
