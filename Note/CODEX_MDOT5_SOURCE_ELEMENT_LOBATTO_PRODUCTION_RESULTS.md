# Mdot=5 source-element Lobatto production results

Date: 2026-07-09

## Target

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact stream source
- local-Mdot wind formulation
- `eta_E = 98.125`
- `N = 164`

This note follows:

- `CODEX_MDOT5_SOURCE_ELEMENT_LOBATTO_SCALING_RESULTS.md`
- `CODEX_MDOT5_SOURCE_ELEMENT_SCALED_LOCAL_SOLVE_RESULTS.md`

## Implementation

Added a new explicit source mode:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_SOURCE_MODE=lobatto_source_element
```

The new mode keeps the global conservative flux coordinate

```text
F = Mdot / Mdot_inner
```

and replaces the old independent source-element slopes with a true 3-point
Lobatto polynomial inside each source interval.

Old conservative source element:

```text
U_L,U_M,U_R
Theta_L,Theta_M,Theta_R
F_L,F_M,F_R
U'_L,U'_M,U'_R
Theta'_L,Theta'_M,Theta'_R
F'_L,F'_M,F'_R
```

New Lobatto source element:

```text
U_L,U_M,U_R
Theta_L,Theta_M,Theta_R
F_L,F_M,F_R
```

Derivatives are computed from the common L/M/R polynomial:

```text
z'_L = (-3 z_L + 4 z_M - z_R) / dx
z'_M = (z_R - z_L) / dx
z'_R = (z_L - 4 z_M + 3 z_R) / dx
```

Production rows inside the source element:

- radial ODE at left/mid/right;
- energy ODE at left/mid/right;
- conservative FV mass row:

```text
F_R - F_L
  - dx/6 * [(Mdot_wind' - Mdot_stream')_L
          + 4 (Mdot_wind' - Mdot_stream')_M
          +   (Mdot_wind' - Mdot_stream')_R] / Mdot_inner = 0
```

Simpson compatibility is no longer an active constraint because it is exact by
construction for the shared Lobatto polynomial. The existing Lobatto audit now
reports zero interpolation slope mismatch in this mode.

Also added:

- Lobatto aux checkpoint keys:
  - `source_lobatto_element_aux_interval_indices`
  - `source_lobatto_element_aux_midpoint_y`
  - `source_lobatto_element_aux_F_mid`
- Lobatto-aware physics audits using the same Lobatto derivatives as production;
- Lobatto-aware local Jacobian audit and local source solve scaling.

## Runs

All runs started from:

```text
outputs/checkpoints/m5_eta_source_element_colscaled_local_pass2_98p125_N164/stage_00_etaE_98p125_N164.npz
```

except pass 2, which started from the very-wide Lobatto checkpoint.

| run | final_full | source max | ODE | radial | energy | FV mass weighted | FV mass raw | energy FV audit | angular FV audit | nfev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen Lobatto eval | `7.249e-01` | `6.612e+00` | `6.612e+00` | `4.600e-02` | `6.612e+00` | `1.082e-02` | `1.082e-03` | `7.867e-01` | `1.610e-01` | 1 |
| local halo8, column-scaled | `1.199e+00` | `1.828e-01` | `1.828e-01` | `1.828e-01` | `4.488e-02` | `4.313e-02` | `4.313e-03` | `3.787e-03` | `1.698e-01` | 14 |
| local halo8, unscaled | `1.030e+00` | `1.811e-01` | `1.811e-01` | `1.811e-01` | `4.368e-02` | `4.117e-02` | `4.117e-03` | `4.494e-03` | `1.611e-01` | 15 |
| local halo8, wide trust | `6.952e-01` | `2.298e-01` | `2.298e-01` | `2.298e-01` | `2.148e-02` | `4.189e-02` | `4.189e-03` | `2.133e-03` | `1.712e-01` | 15 |
| local halo8, very wide trust | `5.571e-01` | `2.173e-01` | `2.173e-01` | `2.173e-01` | `2.156e-02` | `4.061e-02` | `4.061e-03` | `1.901e-03` | `1.717e-01` | 21 |
| very-wide pass 2 | `7.881e-01` | `1.897e-01` | `1.897e-01` | `1.897e-01` | `1.737e-02` | `3.202e-02` | `3.202e-03` | `2.179e-03` | `1.719e-01` | 30 |

Jacobian audit:

| run | rows | vars | rank | raw condition | scaled condition |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen Lobatto eval | 260 | 159 | 159 | `4.214e+07` | `9.589e+06` |
| local halo8, column-scaled | 260 | 159 | 159 | `3.797e+07` | `8.622e+06` |
| local halo8, unscaled | 260 | 159 | 159 | `3.422e+07` | `7.771e+06` |
| local halo8, wide trust | 260 | 159 | 159 | `4.300e+07` | `9.768e+06` |
| local halo8, very wide trust | 260 | 159 | 159 | `4.989e+07` | `1.134e+07` |
| very-wide pass 2 | 260 | 159 | 159 | `7.190e+07` | `1.636e+07` |

## Interpretation

The true Lobatto source element is now implemented and behaves as intended
at the representation level:

- Simpson/interpolation compatibility is exact by construction;
- `U`, `Theta`, and `F` share one polynomial basis;
- interpolation slope mismatch reports zero in Lobatto production mode.

But the local halo8 production solve does not meet the acceptance criteria.

Important findings:

1. The frozen HS-to-Lobatto projection exposes the expected large ODE defect:
   `ODE ~ 6.61`, dominated by the energy equation.

2. A local Lobatto solve reduces this dramatically to `ODE ~ 0.18-0.23`, so
   the new variables are moving in the right direction.

3. The solve then stalls far above the required `1e-4` source tolerance.
   The remaining active source defect is radial ODE, with raw FV mass still
   `~3e-3` to `4e-3`.

4. Wider trust improves the old full residual down to `0.557`, but does not
   solve the Lobatto source element. A second pass reduces source ODE slightly
   to `0.190` but worsens old full residual to `0.788`.

5. The local Lobatto Jacobian remains badly conditioned:
   `cond ~ 3e7-7e7`, scaled `~8e6-2e7`. This is much worse than the old
   family-scaled HS/Fprime audit and explains the `xtol` stalls.

6. Halo16 and halo32 were not attempted because halo8 failed the acceptance
   gate:

```text
required exploratory: ODE < 1e-4, FV mass < 1e-4, energy < 1e-4, no edge export
best achieved:        ODE ~ 1.90e-1, FV mass ~ 3.20e-3 raw
```

## Current conclusion

The mixed HS-slope representation has been replaced by a true Lobatto source
production mode, but the current local solve is not yet a certified source
element.

This is progress relative to the previous audit because the representation
problem is fixed. The new bottleneck is the nonlinear Lobatto source solve and
its local conditioning.

## Recommended next step

Do not continue `eta_E`, do not expand to halo16/halo32, and do not add wind
or angular momentum production yet.

Next implementation should focus on the Lobatto local solve itself:

1. Add interval-local sparsity/Jacobian for the Lobatto source block instead
   of the current dense local finite-difference pattern.
2. Add a formulation homotopy:

```text
g = (1 - chi) * g_HS_aux + chi * g_Lobatto
chi = 0, 0.1, 0.25, 0.5, 0.75, 1
```

   using the homotopy only as a seed path, and certify only `chi=1`.
3. Add row-family diagnostics for the Lobatto local solve at each accepted
   homotopy step:
   - radial ODE;
   - energy ODE;
   - FV mass;
   - energy FV audit;
   - angular FV audit.
4. If the homotopy stalls above `1e-4`, add row/column scaled Newton on the
   local Lobatto block with the physical residual still reported unscaled.

## Verification

- `python3 -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py`
- `git diff --check -- scripts/run_mdot5_local_mdot_eta_continuation.py Note/CODEX_MDOT5_SOURCE_ELEMENT_LOBATTO_PRODUCTION_RESULTS.md`
- `PYTHONPATH=src python3 -m pytest -q`
  - result: `160 passed, 2 subtests passed`
