# Codex Results: Mdot=5 Coupled Inner-Window Eta Continuation

Date: 2026-07-08

## Context

Target branch:

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- compact stream-fed, local-Mdot wind branch
- N164 compatible source-band replacement formulation
- continuation parameter: `eta_E`

Starting strict checkpoint for this sprint:

```text
outputs/checkpoints/m5_eta_two_pass_sonic12_halfstep_from98p1875_N164/
  stage_00_etaE_98p171875_N164.npz
```

Acceptance metric remains the compatible source-band replacement score, not the
legacy midpoint `final_full`.

## Code Change

Added a bounded linearized mode to the coupled inner-window corrector in
`scripts/run_mdot5_local_mdot_eta_continuation.py`.

New flags:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_COUPLED_INNER_WINDOW_LINEARIZED
IMBH_MDOT5_LOCAL_MDOT_ETA_COUPLED_INNER_WINDOW_LINEAR_FD_STEP
IMBH_MDOT5_LOCAL_MDOT_ETA_COUPLED_INNER_WINDOW_LINEAR_RIDGE
IMBH_MDOT5_LOCAL_MDOT_ETA_COUPLED_INNER_WINDOW_LINEAR_MAX_STEP
```

The mode builds one local finite-difference Jacobian for the selected coupled
rows, solves a damped linear least-squares step, and reuses the existing
source-band-score line search. This avoids repeated dense nonlinear optimizer
iterations when the residual floor is at the `~1e-5` level.

## Reproduction Trap Fixed

Manual reruns initially produced order-unity source-band defects because I was
overriding physical stream parameters inconsistently. The correct recipe for
re-evaluating these checkpoints is to avoid manual physical overrides and use
only the formulation flags:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_HALO_INTERVALS=32
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_MASS_INCREMENT=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_GLOBAL_REPLACEMENT=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_GLOBAL_REPLACEMENT_EVALUATE_ONLY=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_GLOBAL_REPLACEMENT_FREEZE_AUX=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_GLOBAL_REPLACEMENT_SKIP_ACCEPTED=0
```

With those flags, the narrow-corrected `eta_E=98.15625` checkpoint reproduces:

```text
source_band_global_replacement_n_intervals = 54
source_band_global_replacement_n_rows      = 446
source_band_global_replacement_final_score = 1.0043409549e-05
peak old_mass row                          = R~12.706 rg
mass_increment_int/link                    = 9.527088928e-06
```

## Runs

### Half-step diagnostic from eta_E=98.25

Table:

```text
outputs/tables/m5_eta_two_pass_sonic12_halfstep_from98p25_N164.json
```

Results:

```text
eta_E=98.234375: score = 9.950611477e-06 strict
eta_E=98.218750: score = 9.941824543e-06 strict
```

The old `98.21875` wall was crossed by inserting `98.234375`.

### Reduced continuation to eta_E=98.1875

Table:

```text
outputs/tables/m5_eta_two_pass_sonic12_halfstep_from98p21875_N164.json
```

Results:

```text
eta_E=98.203125: score = 9.852540025e-06 strict
eta_E=98.187500: score = 9.781105897e-06 strict
```

### eta_E=98.15625 remains slightly non-strict

Table:

```text
outputs/tables/m5_eta_two_pass_sonic12_halfstep_from98p1875_N164.json
```

Results:

```text
eta_E=98.171875: score = 9.945842959e-06 strict
eta_E=98.156250: score = 1.015836792e-05 non-strict
```

The dominant rows are old mass/sonic rows near:

```text
R~5.93 rg, old_mass
R~5.30 rg, old_sonic_pivot
R~12.71 rg, old_mass
```

Mass-increment rows remain strict.

### Nonlinear coupled inner-window diagnostic

Table:

```text
outputs/tables/m5_eta_coupled_inner_window_98p15625_N164.json
```

This used the 4.8--7.2 rg coupled window and did improve the selected inner
sonic/mass rows, but the global max moved outward:

```text
initial source score   = 1.015836792e-05
candidate selected row = 2.417924871e-10
final source score     = 1.004340955e-05
final peak             = old_mass at R~12.706 rg
```

### Linear 10--15 rg patch

Table:

```text
outputs/tables/m5_eta_coupled_window10_15_linear_bare_from_narrow_98p15625_N164.json
```

Result:

```text
initial score = 1.004340955e-05
final score   = 1.003932476e-05
alpha         = 4.8828125e-04
```

The 12.7 rg selected row improved slightly, but the peak moved back to
`R~5.93 rg`.

### Alternating 4.8--7.2 rg linear patch

Table:

```text
outputs/tables/m5_eta_coupled_window4p8_7p2_linear_after10_15_98p15625_N164.json
```

Result:

```text
initial score = 1.003932476e-05
final score   = 1.003852144e-05
alpha         = 3.662109375e-04
```

The peak moved back to `R~12.706 rg`.

### Combined 4.8--15 rg linear patch

Table:

```text
outputs/tables/m5_eta_coupled_window4p8_15_linear_after_alt_98p15625_N164.json
```

Result:

```text
initial score = 1.003852144e-05
final score   = 1.003852144e-05
accepted step = none
candidate score = 1.365830366e-03
```

The combined linear direction was not useful under the current scaling.

### Micro eta staging from 98.171875

Table:

```text
outputs/tables/m5_eta_two_pass_sonic12_micro_from98p171875_N164.json
```

Results:

```text
eta_E=98.1640625: score = 1.015003345e-05 non-strict
eta_E=98.1562500: score = 1.043299543e-05 non-strict
```

Extra eta staging did not fix the floor and actually made the next target
worse under this setup.

## Interpretation

The current blocker is not source-band mass-increment bookkeeping. The
mass-increment rows remain below `1e-5`.

The blocker is a distributed old-row closure floor in `active_outside_old`,
mostly old-mass rows, with coupled sonic-pivot participation near the inner
boundary. Local patches can reduce one row family, but the max immediately
moves to another old-mass row.

The finite-difference global source-band polish is the right conceptual next
operation, but the current finite-difference implementation is too expensive:
with roughly 494 optimizer variables, even `max_nfev=20` implies a very large
number of source-band residual evaluations. I stopped that run before it wrote
output.

## Recommended Next Step

Implement a sparse/linearized source-band global replacement polish:

1. Reuse the existing source-band sparsity from
   `_source_band_global_replacement_sparsity`.
2. Build a row-local finite-difference Jacobian only for rows near the active
   residual floor, or use graph coloring if available.
3. Solve a damped least-squares step for the distributed `active_outside_old`
   mass/sonic floor plus mass-increment guard rows.
4. Use the compatible source-band score for line-search acceptance.
5. Only resume eta continuation below `98.15625` after this score is strictly
   below `1e-5`.

## Verification

```text
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache \
PYTHONPATH=src \
/Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest
```

Result:

```text
160 passed in 3.07s
```
