# Codex Mdot=5 Local-Mdot Eta_E=100 Grid-Homotopy Results

Date: 2026-07-06

This sprint follows the recommendation from the mesh-certification note:
avoid one-shot N160 -> N164 remapping, speed up the local block correction, and
test whether controlled mesh movement opens a Newton basin.

## Code changes

Updated driver:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

New fast block residual control:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_FAST_LOCAL_RESIDUAL
```

This evaluates only the selected interval_R, interval_E, mass, and optional
outer rows during block least-squares.  A consistency check on a saved N164
checkpoint gave:

```text
max_abs_diff = 0.0
```

relative to the old full-residual row selection.

New block centering control:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_PEAK_KIND = radial | energy | mass | auto
```

New grid-homotopy controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GRID_HOMOTOPY_STEPS
IMBH_MDOT5_LOCAL_MDOT_ETA_GRID_HOMOTOPY_COLLAPSE_FRACTION
IMBH_MDOT5_LOCAL_MDOT_ETA_GRID_HOMOTOPY_BLOCK_CORRECT
```

When the start checkpoint has fewer nodes than the target, the new nodes are
first placed close to existing old-grid nodes, then moved to the final target
grid over several homotopy steps.  Optional block correction can be applied at
each mesh step.

## Baseline

Starting strict checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta100_cert_N160_global_polish_from_seed/stage_00_etaE_100_N160.npz
```

N160 strict values:

```text
final_full = 3.896e-06
local_interval_R = 2.654e-06
local_interval_E = 3.896e-06
mass_residual_max = 2.256e-07
```

## N160 -> N164 grid-homotopy tests

Pure mesh homotopy:

```text
outputs/tables/m5_local_mdot_eta100_N160_to_N164_grid_homotopy_seed.md
```

Result:

```text
collapsed-grid full = 1.044e+00
final_full = 2.848e-02
local_interval_R = 2.848e-02
local_interval_E = 7.117e-03
mass_residual_max = 7.512e-04
```

Grid homotopy plus block correction at each mesh step:

```text
outputs/tables/m5_local_mdot_eta100_N160_to_N164_grid_homotopy_block_seed.md
outputs/tables/m5_local_mdot_eta100_N160_to_N164_grid_homotopy_auto_block_seed.md
```

Result:

```text
final_full = 9.014e-03
local_interval_R = 6.904e-03
local_interval_E = 1.298e-03
mass_residual_max = 9.014e-03
peak mass/radial location = R~204.19 rg
```

Interpretation:

```text
Grid homotopy avoids the order-unity source-annulus blow-up at the final grid,
but it still lands in the same coupled radial/mass floor.  Auto-centering did
not improve the result because the block correction quickly moves between
radial and mass dominated intervals.
```

## Additional N164 block passes

Starting from the homotopy+block N164 seed, repeated mass/radial-centered
block passes were tested:

| run | final_full | local_R | local_E | mass | note |
|---|---:|---:|---:|---:|---|
| q8, mass, edge anchor 0.1 | 6.632e-03 | 6.520e-03 | 1.298e-03 | 6.632e-03 | modest improvement |
| q12, mass, edge anchor 0.1 | 6.435e-03 | 6.435e-03 | 2.585e-03 | 5.558e-03 | modest improvement |
| q18, auto/radial pass | 5.615e-03 | 5.615e-03 | 2.585e-03 | 4.190e-03 | broader block helps |
| q24, auto/radial pass | 4.893e-03 | 4.893e-03 | 2.585e-03 | 4.777e-03 | best so far |

Best checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta100_N164_massblock_q24_a0p1_pass3/stage_00_etaE_100_N164.npz
```

Best residual localization:

```text
interval_R = 4.893e-03 at R = 106.95 rg
interval_E = 2.585e-03 at R = 240.30 rg
mass      = 4.777e-03 at R = 189.41 rg
```

The floor is now distributed over a broad radial/mass region rather than a
single source-annulus cell.

## Global polish attempt

A capped global polish from the best q24 seed was launched:

```text
outputs/checkpoints/m5_local_mdot_eta100_N164_massblock_q24_a0p1_pass3/stage_00_etaE_100_N164.npz
```

It was interrupted after several minutes with no stage result; it was again
inside sparse finite-difference Jacobian evaluations.  This matches the earlier
N164 global-polish behavior.

## Interpretation

The new infrastructure improves the N164 transfer:

```text
one-shot N164 seed from N160 polished:        2.532e-02
grid homotopy without correction:            2.848e-02
grid homotopy + stepwise block correction:   9.014e-03
repeated wider block passes:                 4.893e-03
```

This is useful progress, but still not mesh certification.  The remaining
problem is no longer a single source-tail interval.  It is a broad coupled
state/mass correction that the local block can only reduce slowly, while the
global finite-difference Newton step is too expensive and does not promptly
enter a convergent basin.

## Recommended next step

Do not lower eta_E and do not attempt N168 yet.

Best next move:

```text
1. Implement a true reduced Newton solve for the N164 repair region:
   include all nodes from roughly R=90--270 rg plus logR_son/lambda0 if needed.
2. Use the fast selected-row evaluator, but solve a contiguous multi-block
   system containing radial, energy, and mass rows over the whole defect band.
3. Add a banded/sparse Jacobian for this reduced system instead of relying on
   full global finite-difference Jacobian construction.
4. Only when N164 reaches final_full <= 1e-5 should N164 -> N168 be retried.
```

## Verification

```text
python -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
PYTHONPATH=src python -m pytest tests/test_winds.py tests/test_transonic_local.py
```

Result:

```text
48 passed
```
