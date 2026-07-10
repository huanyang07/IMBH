# Mdot=5 Source-Element Lobatto/Scaling Audit Results

Date: 2026-07-09

## Target

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact source, local-Mdot wind
- `eta_E = 98.125`
- `N = 164`
- source mode: `conservative_source_element`
- source window: `SOURCE_BAND_HS_CORE_ONLY=1`, `SOURCE_BAND_HS_RELEASE_HALO=8`

## Code Added

`scripts/run_mdot5_local_mdot_eta_continuation.py` now includes audit-only diagnostics:

- family-scaled local source-block Jacobian:
  - row families: ODE, Simpson, FV mass, midpoint, Fprime, F midpoint;
  - column families: `U`, `Theta`, `F`, `Uprime`, `Thetaprime`, `Fprime`;
  - reports raw and scaled condition numbers and singular-vector localization.
- 3-point Lobatto diagnostic for `U`, `Theta`, and `F`:
  - common L/M/R interpolation basis;
  - Lobatto derivative-based ODE residuals;
  - Lobatto finite-volume mass residual;
  - current HS/Fprime slope mismatch against Lobatto derivatives;
  - per-source-element profile written to row JSON.
- expanded physics audit:
  - pointwise and FV energy-balance norms;
  - `Qvisc`, `Qrad`, `Qadv`, `Qwind`, `Qstream` maxima;
  - explicit angular audit:
    `[Mdot*l-G]_R - [Mdot*l-G]_L - integral(wind*l_w - stream*l_s + tau_s)`.

Verification:

- `py_compile`: passed.
- `git diff --check`: passed.
- `PYTHONPATH=src python -m pytest -q`: `160 passed, 2 subtests passed`.

## Runs

Outputs:

- seed audit:
  - `outputs/tables/m5_eta_source_element_lobatto_scaling_audit_seed_98p125_N164.*`
- local nfev80 checkpoint audit:
  - `outputs/tables/m5_eta_source_element_lobatto_scaling_audit_local_nfev80_98p125_N164.*`

## Source Residuals

| audit | source max | ODE | Simpson | Fprime | FV mass | F midpoint |
|---|---:|---:|---:|---:|---:|---:|
| seed | `6.524` | `6.524` | `1.902e-3` | `3.105e-2` | `3.523e-3` | `1.208e-2` |
| local nfev80 | `2.211e-2` | `1.142e-3` | `2.211e-2` | `2.988e-3` | `2.187e-2` | `1.711e-2` |

## Jacobian Scaling

| audit | raw cond | raw smin | scaled cond | scaled smin | improvement |
|---|---:|---:|---:|---:|---:|
| seed | `8.80e4` | `9.84e-3` | `7.76e2` | `2.92e-3` | `1.13e2` |
| local nfev80 | `2.33e5` | `3.65e-3` | `1.05e3` | `1.99e-3` | `2.21e2` |

Scaled smallest right singular vector, local nfev80 RMS:

- `U`: `1.36e-1`
- `Theta`: `1.58e-2`
- `F`: `1.22e-2`
- `Uprime`: `1.67e-5`
- `Thetaprime`: `3.36e-4`
- `Fprime`: `3.68e-5`

Scaled smallest left singular vector, local nfev80 RMS:

- FV mass: `1.07e-1`
- Simpson: `6.41e-2`
- ODE: `3.53e-2`
- F midpoint: `2.70e-2`
- Fprime: `8.23e-3`
- midpoint: `4.62e-3`

Interpretation: scaling removes most of the apparent conditioning problem, but the weakest scaled direction still lives in `U` and the residual tension is still FV mass + Simpson. This supports GPT's diagnosis: the issue is finite-element consistency, not missing `Fprime`.

## Lobatto Diagnostic

| audit | Lobatto ODE | radial | energy | Fprime defect | FV mass | Simpson | U slope mismatch | Theta slope mismatch | F slope mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed | `5.156` | `8.64e-2` | `5.156` | `2.372` | `4.14e-4` | `1.19e-15` | `10.74` | `1.19` | `2.37` |
| local nfev80 | `6.329` | `4.48e-2` | `6.329` | `3.538` | `2.57e-3` | `1.78e-15` | `10.51` | `2.32` | `3.51` |

Peak radii for local nfev80:

- Lobatto ODE peak: `R ~250.43 rg`
- Lobatto `Fprime` peak: `R ~235.37 rg`
- Lobatto FV mass peak: `R ~255.63 rg`
- `U` slope mismatch peak: `R ~245.32 rg`
- `Theta` slope mismatch peak: `R ~250.43 rg`
- `F` slope mismatch peak: `R ~235.37 rg`

The Lobatto Simpson residual is machine-zero by construction, as expected. But the Lobatto ODE residual is huge because the current optimized HS slopes are not the derivatives of a shared L/M/R polynomial. A true Lobatto production formulation would be a substantive formulation change, not a drop-in diagnostic.

## Energy And Angular Audits

| audit | point energy norm | FV energy norm | max `Qwind/Qvisc` | max `Qadv/Qvisc` | angular FV norm | angular wind | angular stream | angular torque |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed | `1.0` | `0.973` | `0.874` | `1.58` | `0.158` | `2.07e-4` | `0.211` | `5.05e-4` |
| local nfev80 | `2.74e-3` | `2.73e-3` | `1.06e-2` | `0.977` | `0.161` | `7.83e-7` | `0.214` | `5.16e-4` |

The local nfev80 state is thermally consistent under both pointwise and FV energy audits. The large angular residual is dominated by the stream mass angular term, not wind or the explicit stream torque term.

## Interpretation

The current source element is not underdetermined.

The scaled audit says:

- raw condition looked severe (`~2e5`);
- family row/column scaling improves it to `~1e3`;
- the remaining weak direction is still `U`/midpoint interpolation coupled to FV mass and Simpson rows.

The Lobatto audit says:

- a common L/M/R basis makes Simpson compatibility exact;
- but the current HS slope variables are far from Lobatto derivatives;
- forcing Lobatto derivatives into the existing state gives large energy/ODE defects.

The energy audit says:

- the local nfev80 checkpoint is not mainly failing because of thermal/wind-energy bookkeeping.

The angular audit says:

- angular consistency is not yet solved;
- the discrepancy is dominated by stream-carried angular momentum.

## Recommended Next Step

Do not continue `eta_E` and do not expand halo yet.

The best next move is a scaled local solve using the existing equations:

1. Apply row/column family scaling inside the local source optimizer, not just in the audit.
2. Keep current variables and residuals unchanged.
3. Re-run core+halo8 from the local nfev80 checkpoint.
4. Accept only if ODE, Simpson, FV mass, and F midpoint all fall below `1e-4`.

If scaled local solve still stalls, then implement a true Lobatto production source element:

- remove independent `g_node/g_mid` slope variables;
- derive all `Uprime`, `Thetaprime`, and `Fprime` from L/M/R states;
- solve with ODE residuals at L/M/R plus FV mass and angular audit rows.

Before making angular momentum a production row, clarify or implement a physical stream-carried `l_s`; the current audit uses local disk `l` for the stream mass term and therefore may be exposing a closure assumption rather than just a numerical defect.
