# Source-Band Replacement Homotopy Results

Date: 2026-07-07

## Goal

Implement and test the next source-band formulation for the
`Mdot_inner/Edd = 5`, `Rout = 335 rg`, `Rinj = 240 rg`, `f_s = 0.80`,
compact-C2, local-Mdot mass-loaded wind branch at `eta_E = 100`.

The old midpoint source-band rows are no longer used as the strict production
target inside the source-plus-buffer band. They remain guard/audit rows.

## Code Changes

Primary file:

- `scripts/run_mdot5_local_mdot_eta_continuation.py`

New source-band replacement flags:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT=1`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACE_MASS=1`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACE_IMPLICIT_RE=1`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_CHI_MASS=...`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_CHI_IMPL=...`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_OLD_ROWS_AUDIT_ONLY=1`

New formulation:

- Old production residual rows outside the source-plus-buffer band remain
  active.
- Old radial/energy/mass midpoint rows inside the source-plus-buffer band are
  excluded from the active residual by default and reported as audits.
- Source-band active rows now include:
  - finite-volume mass rows with homotopy `chi_mass`;
  - implicit-slope node and midpoint ODE rows with homotopy `chi_impl`;
  - Hermite-Simpson endpoint compatibility rows;
  - source-band interface continuity rows.
- FV energy rows are currently audit-only unless
  `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_FV_ENERGY_WEIGHT > 0`.
- FV angular-momentum audit is explicitly reported as `not_implemented`.

The local replacement polish writes compact columns:

- active residual groups: outside old, FV mass, implicit ODE, Simpson,
  interface;
- guard/audit groups: old source rows, old full residual, FV energy interface
  and element audits, ODE conditioning;
- line-search trials with active/guard pass flags.

## Verification

Syntax:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache \
  /Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
```

Tests:

```bash
PYTHONPATH=src \
  /Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m pytest
```

Result:

- `160 passed in 2.84s`

## Input Checkpoint

Initial eta_E=100 N164 checkpoint:

- `outputs/checkpoints/m5_source_plus_buffer_production_eta100_N164_bandonly_nfev8/stage_00_etaE_100_N164.npz`

This checkpoint is strict under the old midpoint production residual:

- old `final_full = 9.354e-6`

but has the known source-band identity mismatch:

- ODE-integral max `= 14.484`
- max `|A g_old + c| = 10.044`
- max `|g_direct - g_old| = 1.025e3`

## Homotopy Results

Mass replacement was run first with `chi_impl = 0`.

| run | chi_m | chi_impl | active | FV mass | implicit ODE | Simpson | outside old | old source | alpha | nfev | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `m5_source_band_replacement_chi015_eta100_N164` | 0.15 | 0.0 | 1.357e-03 | 1.357e-03 | nan | nan | 5.780e-06 | 8.793e-03 | 0.25 | 5 | True |
| `m5_source_band_replacement_chi0175_eta100_N164` | 0.175 | 0.0 | 5.780e-06 | 2.113e-08 | nan | nan | 5.780e-06 | 1.973e-02 | 1.0 | 6 | True |
| `m5_source_band_replacement_chi020_eta100_N164` | 0.2 | 0.0 | 5.780e-06 | 2.197e-08 | nan | nan | 5.780e-06 | 1.997e-02 | 1.0 | 3 | True |
| `m5_source_band_replacement_chi025_eta100_N164` | 0.25 | 0.0 | 5.780e-06 | 1.124e-06 | nan | nan | 5.780e-06 | 1.948e-02 | 1.0 | 4 | True |
| `m5_source_band_replacement_chi040_eta100_N164` | 0.4 | 0.0 | 5.780e-06 | 1.578e-07 | nan | nan | 5.780e-06 | 2.028e-02 | 1.0 | 10 | True |
| `m5_source_band_replacement_chi050_eta100_N164` | 0.5 | 0.0 | 5.780e-06 | 4.391e-09 | nan | nan | 5.780e-06 | 2.567e-02 | 1.0 | 10 | True |
| `m5_source_band_replacement_chi060_eta100_N164` | 0.6 | 0.0 | 7.881e-05 | 7.881e-05 | nan | nan | 5.780e-06 | 3.479e-02 | 1.0 | 24 | True |
| `m5_source_band_replacement_chi065_eta100_N164` | 0.65 | 0.0 | 7.228e-04 | 7.228e-04 | nan | nan | 8.763e-06 | 3.676e-02 | 0.5 | 13 | True |
| `m5_source_band_replacement_chi070_eta100_N164` | 0.7 | 0.0 | 5.205e-03 | 5.205e-03 | nan | nan | 6.592e-06 | 2.028e-02 | 0.125 | 20 | False |

Small implicit radial/energy replacement was then tested from the clean
`chi_mass=0.50` checkpoint.

| run | chi_m | chi_impl | active | FV mass | implicit ODE | Simpson | outside old | old source | alpha | nfev | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `m5_source_band_replacement_chi050_impl0005_eta100_N164` | 0.5 | 0.005 | 2.405e-02 | 5.562e-05 | 2.405e-02 | 5.407e-04 | 5.820e-06 | 2.340e-02 | 0.0625 | 26 | True |
| `m5_source_band_replacement_chi050_impl0005_relaxed_eta100_N164` | 0.5 | 0.005 | 4.673e-04 | 4.673e-04 | 3.290e-04 | 4.407e-04 | 9.311e-05 | 1.023e-01 | 1.0 | 26 | True |

## Interpretation

The new source-band replacement machinery works and produces the intended
two-objective split:

- active rows can be optimized independently of old source midpoint rows;
- outside old rows remain a strict guard;
- old source midpoint rows expose how far the replacement formulation moves
  away from the suspect old source-band manifold.

Mass replacement is smooth and strict through `chi_mass = 0.50`:

- active residual is limited by outside old rows at `5.78e-6`;
- FV mass rows are `<= 1.1e-6`, and usually much smaller;
- old source rows rise to `O(10^-2)`, confirming that the old midpoint rows are
  not equivalent to the finite-volume source-band view.

Mass replacement becomes exploratory at `chi_mass = 0.60`:

- active residual is `7.88e-5`, dominated by FV mass;
- the solve still accepts full alpha, but it is no longer strict.

Mass replacement is not robust at `chi_mass >= 0.65` with the current grid and
guard:

- `chi_mass=0.65` accepts only `alpha=0.5`, with active `7.23e-4`;
- `chi_mass=0.70` fails under the current budget and line-search guard;
- the rejected full-alpha `chi_mass=0.70` candidate had better active mass but
  pushed outside old rows above the strict `1e-5` guard.

The first implicit radial/energy replacement pilot shows the same structural
conflict as earlier local implicit-slope experiments:

- strict-guard run accepts only `alpha=0.0625`, leaving implicit ODE at
  `2.405e-2`;
- relaxed-guard scout accepts `alpha=1`, improves implicit ODE to
  `3.29e-4`, but violates guards:
  - outside old `= 9.31e-5`;
  - old source `= 1.02e-1`.

So a nearby implicit-compatible state exists, but it is not compatible with the
current outside-band guard and old midpoint source-band manifold.

## Current Bottleneck

The next bottleneck is not merely the mass finite-volume row. It is the
interface between the replacement source-band manifold and the outside old
production manifold.

Evidence:

- mass replacement strict to `chi_mass=0.50`;
- high-chi mass candidates fail mainly when outside old rows or source-band
  guardrails rise;
- implicit replacement can greatly improve active ODE/Simpson rows only by
  moving outside rows to `~1e-4`.

This suggests the source-plus-buffer band is too narrow or the interface
continuity is too weak for the new polynomial/implicit source-band state to
attach cleanly to the old midpoint outside discretization.

## Recommended Next Step

Do not lower `eta_E` yet.

Before continuing `chi_mass -> 1` or increasing `chi_impl`, implement one of:

1. A wider/two-layer source-plus-buffer replacement band, so the transition
   from implicit/FV source rows to old midpoint outside rows is gradual.
2. A dedicated interface compatibility block that adds one or two halo
   intervals on each side with mixed old/new residual rows and stronger
   endpoint slope compatibility.
3. Local analytic/sparse Jacobian support for the replacement rows, especially
   FV mass and implicit ODE/Simpson rows. The small `chi_impl=0.005` pilot took
   roughly minutes rather than seconds.
4. A controlled relaxed-outside scout ladder, clearly labeled exploratory, to
   see whether the replacement manifold can reach `chi_mass=1` if outside old
   rows are allowed up to `1e-4`.

Acceptance for a robust eta_E=100 source-band replacement should require:

- active residual `<= 1e-5`;
- outside old rows `<= 1e-5`;
- FV mass strict at `chi_mass=1`;
- implicit ODE and Simpson rows strict for nonzero `chi_impl`;
- old source rows treated only as audit, but not exploding without bound;
- stable physical diagnostics after a full global polish under the new
  formulation.
