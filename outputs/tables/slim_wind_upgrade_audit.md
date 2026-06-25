# Slim/Wind Upgrade Audit

This audit updates the interpretation of the current IMRI minidisk model after
reading `Note/CODEX_SLIM_WIND_UPGRADE.md`.

## Revised Conclusion

The previous local S-curve result remains a useful target, but the upgraded
global calculation does **not** yet demonstrate a faithful physical slim/wind
hot branch.

The current local Layer-2 calculation with

```text
Q_adv = xi * Mdot * P / (2 pi R^2 rho)
```

shows that an advective sink of plausible magnitude can stabilize a hot branch.
However, the addendum correctly points out that `xi` is imposed. The new
global Layer-3 diagnostic computes `v_R` from angular-momentum transport and
computes `Q_adv = Sigma v_R T ds/dR` from radial entropy gradients. When the
local hot-branch roots are evaluated this way, the inferred `xi_eff` is not the
imposed `xi = 0.3`, and the energy budget is over-cooled by advection.

A fixed-Sigma temperature relaxation can close the formal energy residual, but
the relaxed state is still not a clean steady hot branch: `xi_eff(R)` is broad,
the integrated advective term is not a stable hot-branch cooling reservoir, and
one radial zone has outward mass flux.

## What Changed In Code

Implemented Step 1 from the upgrade note:

- `specific_internal_energy(rho, T)` for gas+radiation mixture.
- `entropy_temperature_gradient(R, rho, T, P, e)`:

  ```text
  T ds/dR = de/dR - P/rho^2 d rho/dR
  ```

- `q_advective(Sigma, v_R, TdsdR)`:

  ```text
  Q_adv = Sigma v_R T ds/dR
  ```

- `xi_eff(R, rho, P, TdsdR)`:

  ```text
  xi_eff = - R rho/P * T ds/dR
  ```

- `mdot_from_vr(R, Sigma, v_R)`:

  ```text
  Mdot = -2 pi R Sigma v_R
  ```

Also added energy-limited wind helpers:

```text
Q_Edd,z = 2 c Omega_K^2 H / kappa
Q_avail = Q_visc + Q_stream + Q_tide - Q_adv
Q_wind = epsilon_w [Q_avail - Q_Edd,z]_+
dotSigma_w = Q_wind / E_w
```

Implemented the near-term global slim/wind diagnostic:

- `GlobalSlimParams`, `GlobalSlimProfile`, and
  `evaluate_global_slim_profile(...)` in
  `src/imri_qpe/layer3_minidisk_1d/global_slim.py`.
- Nearly Keplerian `Omega_K`, `nu = alpha H^2 Omega_K`, stress
  `W = 1.5 nu Sigma Omega_K`.
- Radial velocity from the reduced angular-momentum equation.
- Global energy terms:

  ```text
  Q_visc
  Q_rad
  Q_adv = Sigma v_R T ds/dR
  Q_wind
  energy_residual = Q_visc - Q_rad - Q_adv - Q_wind
  ```

- `relax_temperature_energy_balance(...)`, a diagnostic fixed-Sigma relaxation
  that asks whether a supplied radial surface-density profile can close the
  global energy equation.
- The wind helper now forbids negative radiative cooling when
  `Q_visc - Q_adv < 0`; that deficit is left as an actual energy residual.

## Current Computed Figures

The following figure remains useful as a local diagnostic:

```text
outputs/figures/layer2_physical_advective_scurve.png
```

But the figure should now be captioned as:

```text
Local xi-parameterized advective S-curve, not yet a global slim-disk solution.
```

The new global audit figure is:

```text
outputs/figures/global_slim_wind_audit.png
```

Sprint-A audit hardening output is:

```text
outputs/figures/global_slim_audit_hardened.png
outputs/tables/global_slim_audit_hardened.md
```

Sprint-B isolated no-wind benchmark output is:

```text
outputs/figures/isolated_slim_branch_continuation.png
outputs/tables/isolated_slim_branch_summary.md
```

The repaired Sprint-B solver uses a two-face viscous-heating convention,
physical ISCO angular-momentum constant, direct diffusion cooling in no-wind
cases, and a simultaneous solve for `ln Sigma(R)` and `ln T(R)`. It now
recovers the thin disk and a smooth reduced advective sequence through
moderate super-Eddington rates.

Representative continuation points:

| `Mdot/Mdot_Edd` | Converged | L1 residual | Integrated `Q_adv/Q_visc` | Max `H/R` |
|---:|:---:|---:|---:|---:|
| 0.01 | yes | 2.1e-5 | 1.1e-5 | 0.0053 |
| 1 | yes | 9.7e-4 | 0.186 | 0.263 |
| 3 | yes | 5.4e-4 | 0.474 | 0.436 |
| 10 | yes | 5.1e-5 | 0.767 | 0.603 |
| 100 | no | 6.6 | 7.62 | 5.97 |

Numerical summary from the fiducial run:

| State | Global residual | Median `xi_eff` | Integrated `Q_adv/Q_visc` | `Mdot(R)` range | Inward cells | Max `H/R` |
|---|---:|---:|---:|---:|---:|---:|
| local hot roots evaluated globally | -4.8 | 2.2 | 5.8 | 0.0237-0.12 Msun/yr | 100% | 1.5 |
| fixed-Sigma energy-relaxed profile | 7.4e-19 | 2.7 | -0.09 | -0.0293-0.0248 Msun/yr | 94% | 0.67 |

## Updated Pass/Fail Status

| Requirement | Status |
|---|---|
| Fiducial scales reproduce note | pass |
| Cool stable branch | pass |
| Radiation-pressure unstable branch for `mu_stress < 4/7` | pass |
| Local advective hot branch with imposed `xi` | pass |
| Entropy-gradient advection routines | pass |
| Energy-limited wind helper routines | pass |
| Independent entropy-gradient formula and manufactured tests | pass |
| Boundary-aware L1/L2/max global audit metrics | pass |
| Resolution/stencil comparison of local hot-root audit | pass |
| Global `Q_adv = Sigma v_R T ds/dR` evaluator | pass |
| Fixed-Sigma global energy relaxation | diagnostic pass |
| Isolated low-rate thin-disk benchmark | pass |
| Isolated reduced advective branch through moderate rates | pass |
| Isolated reduced branch at QPE target | fail/outside Keplerian validity |
| Transonic low-rate sonic eigenvalue smoke test | pass through `Mdot/Mdot_Edd = 0.03` |
| Transonic continuation to moderate/high rates | not robust yet |
| Faithful steady advective/wind hot branch | not yet |
| `xi_eff(R)` matching the imposed local hot branch | fail |
| Global conservation residual diagnostic | pass |
| Perturbative stability test of branches | not yet |
| Time-dependent limit cycle without manual switching | not yet |

## Next Analysis Step

The next meaningful calculation is not another local S-curve with tuned `xi`.
The first true transonic solver scaffold now exists and passes low-rate
sonic-eigenvalue smoke tests:

```text
outputs/figures/transonic_branch_summary.png
outputs/tables/transonic_solver_audit.md
```

After replacing local finite-difference partials with analytic derivatives and
replacing SciPy's full-residual finite-difference Jacobian with a block-local
sparse Jacobian, the solver now satisfies the current smoke-test tolerance
through `Mdot/Mdot_Edd = 0.03`. However, continuation is not robust yet at
`Mdot/Mdot_Edd = 0.05`, `0.1`, or `1`. The next numerical bottlenecks are:

1. Replace the block-local finite-difference Jacobian with a true analytic
   global Jacobian or staged Newton solve.
2. Improve residual scaling and continuation/remapping.
3. Re-run grid convergence at `Mdot/Mdot_Edd = 1e-3`, `0.003`, `0.01`, and
   `0.03`.
4. Continue to `0.05`, `0.1`, `1`, and then toward the QPE target only after
   moderate-rate convergence is robust.

Only after this succeeds should the hot branch be called physical rather than
parameterized. The current diagnostic is useful because it says the local
`xi`-hot branch does **not** automatically survive this test.

## Tests

Current test result:

```text
Ran 93 tests
OK
```
