# Codex Cost-Aware Continuation and Robin Closure Results

Date: 2026-07-02

## Scope

This pass follows the next step after the source-fraction tangent predictor:

1. Add cost-aware adaptive source-fraction continuation.
2. Add an opt-in soft/Robin angular outer closure.
3. Test both near the current stream front.

The branch is the residual-remeshed `N=768` stream-fed no-wind branch:

```text
Mdot_inner/Edd = 2
Rout = 300 rg
Rinj/Rout = 0.80
stream_torque_delta_l_fraction = +0.005
no wind
no stream heating
```

## Code Changes

Updated:

```text
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
scripts/run_standard_slim_stream_mass_annulus_scan.py
scripts/run_standard_slim_stream_anchor_regression.py
scripts/run_standard_slim_stream_residual_remesh.py
```

### Cost-Aware Adaptive Control

New environment controls:

```text
IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_SHRINK_NFEV
IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_HARD_SHRINK_NFEV
IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_GROW_NFEV
IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_SHRINK
IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_COST_HARD_SHRINK
```

The output table now records:

```text
predictor
attempt step
next step
cost action
```

### Robin Angular Closure

New closure:

```text
outer_closure = "pressure_supported_robin_energy"
```

It keeps the thin-energy outer boundary but replaces the angular residual with:

```text
R_ang = (1 - chi) * R_pressure_supported_omega
      + chi * ( d ln(Omega/OmegaK) / d ln R - target ) / scale
```

New `TransonicSlimParams` fields:

```text
outer_robin_chi
outer_robin_slope_target
outer_robin_slope_scale
```

Stream checkpoint loading/saving now preserves these fields.

## Cost-Aware Test

Output:

```text
outputs/tables/high_mdot_stream_source_cost_aware_0p83_to0p83225_N768.md
outputs/checkpoints/high_mdot_stream_source_cost_aware_0p83_to0p83225_N768/
```

Run settings:

```text
start f_s = 0.830
target f_s = 0.83225
initial step = 0.001
min step = 0.00025
cost shrink threshold = nfev >= 8
hard shrink threshold = nfev >= 20
```

Results:

| f_s | predictor | step | final residual | dominant | nfev | next step | action |
|---:|---|---:|---:|---|---:|---:|---|
| 0.831 | tangent:1 | 0.001 | 2.571e-7 | outer_omega | 4 | 0.001 | grow/hold |
| 0.832 | tangent:1 | 0.001 | 2.546e-7 | outer_omega | 10 | 0.0005 | shrink |
| 0.83225 | tangent:1 | 0.00025 | 1.654e-6 | interval_E | 25 | 0.00025 | hard shrink |

Interpretation:

```text
The controller now detects the rising corrector cost and shrinks the next step
even when the solve is accepted.
```

This does not remove the `f_s≈0.832--0.833` bottleneck, but it prevents the
continuation from blindly growing or holding large steps after expensive
accepted solves.

## Robin Closure Tests

At the hard-closure `f_s=0.83225` anchor, the measured outer slope residual is:

```text
d ln(Omega/OmegaK) / d ln R = -2.6525e-3
```

### Zero-slope Robin target

Tested:

```text
outer_closure = pressure_supported_robin_energy
outer_robin_slope_target = 0
outer_robin_slope_scale = 1
chi = 0.1, 0.5
```

Outputs:

```text
outputs/tables/high_mdot_stream_source_robin_chi01_fs083225_N768.md
outputs/tables/high_mdot_stream_source_robin_chi05_fs083225_N768.md
```

Results:

| chi | initial residual | final residual | dominant | nfev | result |
|---:|---:|---:|---|---:|---|
| 0.1 | 2.436e-4 | 2.435e-4 | outer_omega | 91 | failed |
| 0.5 | 9.826e-4 | 9.822e-4 | outer_omega | 102 | failed |

The solver could not reduce the Robin angular residual. A similar `chi=0.1`
spot check at `f_s=0.82` was also slow and was interrupted.

### Local-slope Robin target

Also tested using:

```text
outer_robin_slope_target = -2.6525e-3
chi = 1
```

for a tiny continuation step from `f_s=0.83225` toward `0.833`. The seed was
worse than the hard-boundary tangent seed and the corrector was interrupted
after becoming slow.

## Interpretation

Cost-aware stepping is useful and should stay.

The first Robin closure form is not a quick fix. The naive zero-slope target is
incompatible with the current branch, and a local-slope target did not improve
the continuation front. This means the next outer-boundary fix should not be a
simple slope residual imposed at the same finite outer edge.

The current bottleneck remains:

```text
f_s ≈ 0.832--0.833
Mdot_outer/Mdot_inner ≈ 0.171
dominant problem: interval_E / outer source-tail corrector stiffness
not sonic failure
```

## Recommended Next Step

Do not spend more time on the current Robin slope target as-is. The better next
move is one of:

1. Add a true outer reservoir/tail domain beyond `Rout=300 rg`, so the stream
   annulus is not pressed directly against a hard finite boundary.
2. Implement a compact source shape with no tail at `Rout` and rerun
   `f_s=0.82--0.84`.
3. Add a finite-volume/integral energy residual in the outer source-tail cells
   to reduce interval_E stiffness.

Among these, the compact no-tail source-shape test is the fastest diagnostic.
