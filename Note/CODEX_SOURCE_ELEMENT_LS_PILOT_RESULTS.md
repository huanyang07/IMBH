# Codex Source-Element LS Pilot Results

Date: 2026-07-06

## Context

This sprint started from GPT's `eta_E=90` source-annulus formulation note:

- `Note/CODEX_SOURCE_ELEMENT_PRODUCTION_FORMULATION_PLAN.md`

Target branch:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
Rinj = 240 rg
f_s = 0.80
eta_E = 90
compact source annulus
local-Mdot mass-loaded wind
```

The implementation added a new disabled-by-default local source-element LS mode in:

- `scripts/run_mdot5_local_mdot_eta_continuation.py`

Key flags:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_LS=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_LS_GAMMAS=...
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_LS_FV_MASS=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_LS_FV_ENERGY=1
```

## Implemented

The new LS mode adds:

- polynomial source-element state/slope evaluation from local 5-node Lagrange stencils;
- pointwise local-Mdot parameters so the physics routines see the polynomial `logMdot` value and derivative at collocation points;
- source radial and energy collocation rows using the polynomial state;
- finite-volume mass rows using polynomial wind quadrature and exact stream-source integrals;
- finite-volume energy rows using the same `Qvisc`, `Qstream`, `Qrad`, `Qadv`, and `Qwind` convention as the differential residual;
- a sparse finite-difference Jacobian pattern for local source-element variables;
- gamma continuation and filter acceptance;
- table output for LS group norms.

The strict default filter is now:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_LS_FILTER_TOL=0.0
```

so the local LS step is not allowed to reduce one residual group by worsening another, unless this is explicitly relaxed for diagnostics.

## Runs

### N201 strict pilot

Run:

```text
outputs/tables/m5_local_mdot_eta90_source_element_ls_poly_strict_N201.json
```

Checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta90_Nsrc32_source_domain_qm_halo4_seed/stage_00_etaE_90_N201.npz
```

Result:

```text
source_element_ls_applied      False
production final_full          3.9291186e-2
old source_band_extra          3.0880927e-2
poly radial defect             5.9573522e-2
poly energy defect             2.5540674e-1
poly FV mass defect            2.3627608e-2
poly FV energy defect          3.8012403e-2
```

Interpretation:

The polynomial source-element energy defect is much larger than the old endpoint-linear source-band audit. This confirms that the previous audit was under-resolving the hidden source-annulus defect.

### N201 relaxed diagnostic

Run:

```text
outputs/tables/m5_local_mdot_eta90_source_element_ls_poly_pilot_N201.json
```

This used `SOURCE_ELEMENT_LS_FILTER_TOL=0.02`.

Result:

```text
source_element_ls_applied      True
poly energy defect             2.5540674e-1 -> 2.3792226e-1
poly FV mass defect            2.3627608e-2 -> 2.2149053e-2
poly FV energy defect          3.8012403e-2 -> 3.7588393e-2
old source_band_extra          3.0880927e-2 -> 3.1208403e-2
production full                unchanged at 3.9291186e-2
```

Interpretation:

The relaxed local LS direction reduces the new polynomial energy/FV defects, but it worsens the old endpoint source-band audit. This is exactly the tradeoff GPT warned about, so the relaxed result is diagnostic only.

### N251 strict pilot

Run:

```text
outputs/tables/m5_local_mdot_eta90_source_element_ls_poly_strict_N251.json
```

Checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta90_source_element_refine2_global_domain2_eta90/stage_00_etaE_90_N251.npz
```

Result:

```text
source_element_ls_applied      False
production final_full          3.7518660e-2
old source_band_extra          3.6121368e-2
poly radial defect             5.0695160e-2
poly energy defect             1.8723939e-1
poly FV mass defect            1.2499953e-2
poly FV energy defect          7.9317703e-2
```

Interpretation:

N251 does reduce the new polynomial radial, energy, and FV mass defects relative to N201. However it worsens FV energy and the old endpoint audit. Therefore the earlier conclusion still holds: N251 is informative but not certified.

### N251 relaxed diagnostic

Run:

```text
outputs/tables/m5_local_mdot_eta90_source_element_ls_poly_relaxed_N251.json
```

This used `SOURCE_ELEMENT_LS_FILTER_TOL=0.02`.

Result:

```text
source_element_ls_applied      True
poly energy defect             1.8723939e-1 -> 1.8059725e-1
poly FV mass defect            1.2499953e-2 -> 1.2394751e-2
poly FV energy defect          7.9317703e-2 -> 7.2188748e-2
old source_band_extra          3.6121368e-2 -> 3.6775257e-2
production full                unchanged at 3.7518660e-2
```

Interpretation:

The relaxed N251 direction again improves the polynomial LS groups but worsens the old endpoint audit. It is not acceptable as a certified source-annulus step.

## Main finding

The new polynomial LS audit is doing what it should: it exposes a larger hidden source-annulus energy defect than the old endpoint-linear source-band audit.

However, with the current variable layout, the local LS descent direction still trades residual groups:

```text
polynomial energy/FV defects improve
old endpoint source-band audit worsens
```

Therefore the current implementation is a useful diagnostic advance, but not yet the certified source-annulus production formulation.

## Next move

Do not lower `eta_E`.

The next implementation step should add stronger source-element compatibility rather than just more LS iterations:

1. Add explicit source-block boundary/interface rows.
2. Add a production option that replaces old endpoint source-band rows with polynomial source-element rows, while keeping the old endpoint rows as a separate audit.
3. Add true source-element internal or flux variables if the current nodal stencil layout keeps forcing the same tradeoff.
4. Only after the strict filter accepts a step that improves both the polynomial groups and old audits should we release to global polish.

Current acceptance status:

```text
eta_E=90 source annulus remains uncertified.
```
