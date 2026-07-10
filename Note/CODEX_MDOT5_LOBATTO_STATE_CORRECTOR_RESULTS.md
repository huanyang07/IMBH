# Mdot=5 True Lobatto State Corrector Results

Date: 2026-07-09

Target:

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact source
- local-Mdot wind
- `eta_E = 98.125`
- `N = 164`

Baseline checkpoint:

```text
outputs/checkpoints/m5_eta_lobatto_homotopy_chi0_local_refslopes_98p125_N164/stage_00_etaE_98p125_N164.npz
```

## Implementation

Added an opt-in true Lobatto state reconstruction/corrector:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_STATE_CORRECTOR=1
```

The corrector directly minimizes the true Lobatto finite-element residual rather than the HS/Lobatto blended homotopy rows.

For each source interval it uses the shared 3-point Lobatto polynomial state:

```text
U_L, U_M, U_R
Theta_L, Theta_M, Theta_R
F_L, F_M, F_R
```

and computes derivatives from the polynomial:

```text
z'_L = (-3 z_L + 4 z_M - z_R) / dx
z'_M = (z_R - z_L) / dx
z'_R = (z_L - 4 z_M + 3 z_R) / dx
```

Production rows in the corrector:

- true Lobatto radial ODE: `A_R(z) z' + c_R(z)`
- true Lobatto energy ODE: `A_E(z) z' + c_E(z)`
- Lobatto/Simpson finite-volume mass conservation
- optional pointwise `F'` consistency rows
- endpoint anchors, optional all-node anchors, optional midpoint anchors

New environment knobs:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_STATE_CORRECTOR_RADIAL_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_STATE_CORRECTOR_ENERGY_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_STATE_CORRECTOR_MASS_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_STATE_CORRECTOR_FPRIME_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_STATE_CORRECTOR_EDGE_ANCHOR_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_STATE_CORRECTOR_NODE_ANCHOR_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_STATE_CORRECTOR_MIDPOINT_ANCHOR_WEIGHT
```

Added row-local diagnostics:

- true corrector ODE/radial/energy/FV maxima;
- peak radius;
- peak interval;
- peak Lobatto point label (`left`, `mid`, `right`);
- peak component (`radial`, `energy`, `mass`, `Fprime`).

## Results

Initial true Lobatto audit:

```text
ODE max    = 6.631
energy max = 6.631
radial max = 4.603e-2
FV mass    = 9.905e-4
```

Run comparison:

| run | nfev | final_full | true ODE | energy | radial | FV mass | peak |
|---|---:|---:|---:|---:|---:|---:|---|
| `m5_eta_lobatto_state_corrector_energy3_trust003_98p125_N164` | 12 | 1.352e+00 | 1.808e-01 | 8.851e-03 | 1.808e-01 | 4.205e-03 | R=203.1, left/radial |
| `m5_eta_lobatto_state_corrector_mass100_node1e3_98p125_N164` | 27 | 5.980e-01 | 1.973e-01 | 6.221e-03 | 1.973e-01 | 1.538e-04 | R=203.1, left/radial |
| `m5_eta_lobatto_state_corrector_balanced_mass100_98p125_N164` | 16 | 9.348e-01 | 1.809e-01 | 5.001e-02 | 1.809e-01 | 4.373e-05 | R=203.1, left/radial |
| `m5_eta_lobatto_state_corrector_r2_e3_mass100_98p125_N164` | 14 | 9.285e-01 | 1.844e-01 | 2.566e-02 | 1.844e-01 | 2.095e-04 | R=203.1, left/radial |
| `m5_eta_lobatto_state_corrector_mass100_halo16_98p125_N164` | 100 | 1.572e+00 | 2.760e-01 | 2.118e-02 | 2.760e-01 | 1.177e-03 | R=186.4, left/radial |

Best current seed:

```text
m5_eta_lobatto_state_corrector_mass100_node1e3_98p125_N164
```

Why:

- `final_full` improves from `7.059e-1` to `5.980e-1`.
- true Lobatto energy drops from `6.631` to `6.221e-3`.
- true Lobatto FV mass improves from `9.905e-4` to `1.538e-4`.
- remaining ODE defect is now radial, localized at the left Lobatto point near `R = 203.1 rg`.

## Interpretation

The true Lobatto state corrector works for the requested target:

- It directly reduces the true Lobatto ODE residual instead of hiding it in a blended residual.
- It collapses the energy component by roughly three orders of magnitude.
- With strong mass weighting and weak node/midpoint anchors, it also improves the global full residual and finite-volume mass defect.

The remaining obstruction is no longer the energy Lobatto mismatch. It is a radial interface/boundary defect at the left edge of the source/buffer element.

Halo16 did not help; it moved the defect inward and worsened both FV mass and ODE. That suggests the current issue is not simply insufficient halo width. It is more likely an interface compatibility problem at the left source edge, where the true Lobatto derivative wants a different radial slope than the adjacent old/global formulation permits.

## Next Plan

1. Add a source-edge radial compatibility row:
   - match the left-edge Lobatto radial derivative to the adjacent outer/global finite-volume radial residual;
   - start as an audit row, then promote if it explains the `R~203 rg` peak.

2. Add a one-sided interface buffer element:
   - one Lobatto element immediately inside the left source edge;
   - shared endpoint state with the source element;
   - radial ODE active, energy optional/guarded.

3. Add an acceptance guard for the corrector:
   - accept a corrector checkpoint only if true Lobatto energy, FV mass, and global full residual do not regress beyond set factors;
   - keep the target ODE residual as a diagnostic when radial interface rows are intentionally relaxed.

4. Use `m5_eta_lobatto_state_corrector_mass100_node1e3_98p125_N164` as the next seed for interface tests, not the halo16 run.
