# Targeted Remesh and Higher-N Refinement Results

Date: 2026-07-03

## Purpose

Test whether the physical-gate wall near

```text
0.89806 < f_s_clean < 0.898125
```

is just an unresolved local physical-energy defect near the raw physical `interval_E` peak at `R ~ 259.18 rg`.

The acceptance criterion remains:

```text
weighted solver residual <= 1e-5
raw physical-domain differential energy residual <= 3e-5
```

## Code Changes

Updated `scripts/run_standard_slim_stream_mass_annulus_scan.py`:

1. Increased checkpoint mass-fraction precision from `.4g` to `.9g`.
   - This prevents `f_s=0.8980625` and `f_s=0.898125` from overwriting each other as `0.8981`.

2. Tightened residual-remesh adoption under the physical gate.
   - If the original state is physically clean, a remeshed state cannot replace it unless it is also physically clean.
   - If both original and remeshed states fail the physical gate, the remeshed state is only adopted when it lowers the raw physical energy residual.

Regression tests:

```text
146 passed
```

## Runs

### 1. Aggressive target remesh

Output:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_target_remesh_0898_to0898125.md
outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_target_remesh_0898_to0898125/
```

Settings:

```text
N = 896
target R = 259.18426 rg
remesh strength = 24
blend = 0.85
W_physical_E = 8
W_target = 12
target log width = 0.012
```

Results:

| f_s | final full | physical_E | remesh adopted | remesh physical_E | result |
|---:|---:|---:|---|---:|---|
| 0.8980625 | 5.738e-08 | 2.934e-05 | false | 5.464e-03 | accepted, kept original grid |
| 0.898125 | 4.324e-06 | 5.850e-03 | true before adoption-rule fix | 5.850e-03 | rejected |

Interpretation:

The aggressive target remesh creates a large raw physical energy defect. This is not a useful route to a clean anchor.

### 2. Gentle target remesh

Output:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_gentle_remesh_08980625_to0898125.md
outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_gentle_remesh_08980625_to0898125/
```

Settings:

```text
N = 896
target R = 259.18426 rg
remesh strength = 8
blend = 0.35
W_physical_E = 4
W_target = 2
target log width = 0.025
```

Result:

| f_s | final full | physical_E | remesh adopted | remesh physical_E | result |
|---:|---:|---:|---|---:|---|
| 0.898125 | 6.155e-08 | 3.147e-05 | false | 3.669e-02 | rejected |

Interpretation:

The corrected adoption rule kept the original state. The remeshed state was much worse physically. The clean frontier did not move.

### 3. N1024 focused target grid

Output:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_N1024_target_grid.md
outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_N1024_target_grid/
```

Settings:

```text
N = 1024
source-grid mode = annulus_peak
target fraction = 259.18426 / 335 = 0.77368436
source-grid blend with current = 0.45
```

Result:

| f_s | initial full | final full | physical_E | result |
|---:|---:|---:|---:|---|
| 0.8980625 | 1.287e-04 | 4.971e-05 | 2.256e-02 | rejected |

Interpretation:

The focused N1024 grid cannot preserve the already-clean `f_s=0.8980625` anchor.

### 4. N1024 resampled current grid

Output:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_N1024_resample_grid.md
outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_N1024_resample_grid/
```

Settings:

```text
N = 1024
grid transfer = resample current custom grid
no source-grid retargeting
```

Result:

| f_s | initial full | final full | physical_E | result |
|---:|---:|---:|---:|---|
| 0.8980625 | 6.954e-05 | 8.833e-07 | 1.098e-03 | rejected |

Interpretation:

Even plain N1024 resampling reaches a good weighted residual but fails the raw physical differential energy audit. This strongly suggests that direct grid remapping is not physically consistent near the wall.

## Conclusion

The wall near `f_s ~ 0.8981` is not fixed by simple local node concentration or by a direct jump to N1024.

Key evidence:

- tangent seeds are already tiny;
- weighted residuals can be excellent;
- raw physical energy residual remains above the gate;
- direct remeshing often makes raw physical `interval_E` much worse;
- N1024 remaps fail to preserve the clean `f_s=0.8980625` anchor.

This is therefore not simply "use more N." It is a grid-continuation / residual-objective consistency problem.

## Recommended Next Step

Implement fixed-physics grid continuation before further source-fraction continuation:

1. Start from the clean N896 `f_s=0.8980625` anchor.
2. Introduce a grid homotopy parameter `eta` from current grid to target/refined grid.
3. Continue at fixed `f_s=0.8980625` in small `eta` steps, with the physical gate active.
4. Only after the refined grid preserves the clean anchor, retry `f_s=0.898125`.
5. If fixed-physics grid continuation cannot preserve the clean anchor, change the Newton objective so it directly includes the raw physical differential energy residual, not just the weighted integrated residual.

The immediate robust claim remains:

```text
last clean source fraction = f_s = 0.8980625
next attempted f_s = 0.898125 fails physical_E gate
```
