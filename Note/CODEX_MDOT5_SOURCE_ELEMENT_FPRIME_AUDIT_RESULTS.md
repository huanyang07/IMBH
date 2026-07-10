# Mdot=5 Source-Element Fprime Audit Results

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

`scripts/run_mdot5_local_mdot_eta_continuation.py` now exports opt-in conservative source-element diagnostics:

- finite-difference local source-block Jacobian audit:
  - `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_CONSERVATIVE_SOURCE_ELEMENT_JACOBIAN_AUDIT=1`
  - rank, singular values, condition number;
  - smallest right-singular-vector localization by variable family;
  - smallest left-singular-vector localization by residual family.
- source quadrature physics audit:
  - normalized local energy balance;
  - `Qwind/Qvisc`, `Qadv/Qvisc`, `Qrad/Qvisc`, `Qstream/Qvisc`;
  - provisional angular flux/FV residual diagnostic.

Verification:

- `py_compile`: passed.
- `PYTHONPATH=src python -m pytest -q`: `160 passed, 2 subtests passed`.

## Audit Runs

Outputs:

- seed audit:
  - `outputs/tables/m5_eta_source_element_fprime_corehalo8_audit_seed_98p125_N164.*`
- previous local-source checkpoint audit:
  - `outputs/tables/m5_eta_source_element_fprime_corehalo8_audit_local_nfev80_98p125_N164.*`

| audit | source max | ODE | Simpson | Fprime | FV mass | F midpoint | rank/vars | smin | cond |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed | `6.524` | `6.524` | `1.902e-3` | `3.105e-2` | `3.523e-3` | `1.208e-2` | `318/318` | `9.84e-3` | `8.80e4` |
| local nfev80 checkpoint | `2.211e-2` | `1.142e-3` | `2.211e-2` | `2.988e-3` | `2.187e-2` | `1.711e-2` | `318/318` | `3.65e-3` | `2.33e5` |

The source-block audit dimensions are:

- rows: `496`
- variables: `318`

## Singular-Vector Localization

Smallest right singular vector, local checkpoint RMS by variable family:

- `logu`: `1.36e-1`
- midpoint `logu/logT`: `9.67e-2`
- `logT`: `1.57e-2`
- `g_mid`: `1.01e-2`
- `g_node`: `8.70e-3`
- `F`: `1.21e-3`
- `F_mid`: `1.21e-3`
- `Fprime_node`: `6.65e-5`
- `Fprime_mid`: `7.05e-5`

Smallest left singular vector, local checkpoint RMS by residual family:

- Simpson compatibility: `6.77e-2`
- FV mass: `3.89e-2`
- midpoint compatibility: `1.85e-2`
- `F_midpoint`: `4.91e-3`
- Fprime ODE: `3.47e-3`
- ODE rows: `2.18e-3`

Interpretation: the stalled direction is not an `Fprime` null mode and not an ODE-only problem. It is mainly a `logu`/midpoint-state compatibility direction, with residual tension concentrated in Simpson and FV-mass rows.

## Physics Audit

| audit | energy balance norm | max `Qwind/Qvisc` | max `Qadv/Qvisc` | angular FV norm |
|---|---:|---:|---:|---:|
| seed | `1.0` | `0.874` | `1.58` | `0.158` |
| local nfev80 checkpoint | `2.74e-3` | `1.06e-2` | `0.977` | `0.161` |

The local-source solve greatly improves the explicit thermal balance and reduces wind dominance, but the angular FV diagnostic remains at `~0.16`. This angular diagnostic is provisional, not a production row, but it is a warning that the conservative source element is being asked to satisfy mass/energy closure without an equally compatible angular-momentum source-element closure.

## Current Interpretation

The conservative `F+Fprime` source element is not rank deficient: the local source Jacobian is full column rank.

The real issue is poor conditioning plus row-family incompatibility:

- the optimizer can reduce ODE rows to `~1e-3`;
- it then stalls with Simpson and FV-mass rows at `~2e-2`;
- the weakest Jacobian direction is dominated by `logu` and midpoint states, not by the mass-flux variables;
- the energy audit is acceptable after the local solve, so the immediate bottleneck is not simply wind-energy thermodynamics.

## Recommended Next Move

Do not continue `eta_E` and do not expand halo yet.

The next formulation change should add a consistent source-element angular momentum closure and then retest the same core+halo8 block:

1. Promote angular flux/source consistency to an audit first, using the same L/M/R source element and Simpson quadrature.
2. Add explicit angular source-element production rows only if the audit confirms the residual is not a harmless diagnostic convention.
3. Re-run the local source solve with ODE + Simpson + FV mass + angular closure measured together.
4. If conditioning remains `>1e5`, implement a row/column scaled local block Jacobian or a bordered/local SVD step rather than more weight tuning.

This fits GPT's guidance: the current barrier is no longer missing `F`; it is a coupled conservative finite-element closure problem.
