# Project Status

- Updated: 2026-07-11
- Pre-cleanup scientific tag: `pre-cleanup-p0-2026-07-11`
- Legacy phase classification tag: `legacy-steady-positive-flux-dae-2026-07-10`

This is the canonical project handoff. Status labels mean:

- **CERTIFIED:** passes the stated numerical and physical gates for its scope.
- **SUPPORTED BUT NOT FULLY CERTIFIED:** strong numerical evidence with an
  identified unresolved robustness or closure condition.
- **DIAGNOSTIC ONLY:** useful mathematical or numerical evidence that must not
  be promoted to a physical branch claim.
- **REJECTED:** tested formulation or composite fails its acceptance gates.
- **PLANNED:** not implemented or not yet evaluated.

## Result Matrix

| Result | Status | Decisive evidence | Limitation |
|---|---|---|---|
| Standard no-wind slim disk through `Mdot/Edd=5` | **CERTIFIED** | N768 accepted canonical state; high-rate ladder and mesh checks | Does not include stream, heating, or wind |
| Compact stream-fed no-wind `Mdot_inner/Edd=2`, `f_s=0.80` | **SUPPORTED BUT NOT FULLY CERTIFIED** | N896 residual-remeshed canonical state and N640/768/896 diagnostics | Relies on residual-aware remeshing; naive remaps fail |
| N164 global phase-DAE entry at `Mdot_inner/Edd=5`, `eta_E=98.125` | **SUPPORTED BUT NOT FULLY CERTIFIED** | Local phase radial/energy/FV equations can be solved accurately | Global far-side attachment is unresolved |
| Formal low-velocity endpoint near `225.52125 rg` | **DIAGNOSTIC ONLY** | Two step sizes, bordered continuation, source-shape scans, homogeneous residual audit | `L_u/H<1` first at `223.23643 rg`; endpoint is outside 1D validity |
| Local annulus-mass integrability | **SUPPORTED BUT NOT FULLY CERTIFIED** | Common-window exponent gives positive mass power `0.435-0.559` | Applies to the mathematical asymptote under current equations |
| Existing global phase-plus-ordinary-tail composite | **REJECTED** | Phase rows remain small while outside radial/energy defects become large | Rejection is not global nonexistence |
| Independent outer-manifold connection | **DIAGNOSTIC ONLY** | Best flux mismatch `1.04e-5`; best state mismatch `1.77e-3` | Misses strict `1e-3` state gate; shooting map condition `8.74e5` |
| Algebraic angular representation ledger | **SUPPORTED BUT NOT FULLY CERTIFIED** | Point closure at machine precision; phase FV floor `9.75e-6` | Representation identity, not physical `l_s`, `l_w`, `tau_ext` closure |
| Unified conservative mass/angular/energy formulation | **SUPPORTED BUT NOT FULLY CERTIFIED** | No-wind `Mdot/Edd=5` regression and physical compact-stream roots pass raw ledgers | Stream-fed wind tests currently start from `Mdot_inner/Edd=2` |
| Physical mass-loaded-wind steady branch at `Mdot_inner/Edd=2` | **SUPPORTED BUT NOT FULLY CERTIFIED** | Exploratory roots through `epsilon_w=0.54`; scouts through `0.90` | Wind loss is `<0.5%`; this is not a strong hot branch |
| Unified compact-stream branch at `Mdot_inner/Edd=5`, `Rout=335 rg` | **SUPPORTED BUT NOT FULLY CERTIFIED** | `f_s=0.05,0.10,0.30` pass at `N=192,256,384`; conservative mass budget closes | Outer compatibility convergence is only exploratory |
| Unified Mdot=5 energy-limited wind | **SUPPORTED BUT NOT FULLY CERTIFIED** | Exact-source eta=8 roots pass at N426/N512/N640; power/carried residuals agree below `9e-20`; essentially all wind mass is Bernoulli-unbound | Wind loss is only `1.71%`; no hot transition; launch closure remains artificial and gives a tiny formally superluminal inner tail |
| Terminal-Bernoulli Mdot=5 wind | **SUPPORTED BUT NOT FULLY CERTIFIED** | Physical `B_infinity=0.02 c2` roots pass at N426/N512/N640 with wind loss `6.87-6.88%` and no mass-cap activation | Still no new hot topology; inner rate remains imposed and stream supply is fractional |
| Angularly closed prescribed-viscosity signed-flux disk | **SUPPORTED BUT NOT FULLY CERTIFIED** | Exact stream `S_M,S_J,S_E`; open split `0.170064596`; wall torque `0.768986584`; unnamed angular defect below `9e-16` | Fixed-Keplerian steady reservoir; source-bearing coupled time evolution is pending |
| Angularly closed thermoviscous signed-flux disk | **SUPPORTED BUT NOT FULLY CERTIFIED** | N64-N512 roots; wall internal-energy export `0.548`, `H/R=0.341`, `Lrad=1.323 LEdd`; angular defect at roundoff | Internal-energy ledger is not total energy; ideal wall power, inner transonic coupling, and stability are pending |
| Fully time-dependent total-energy signed-flux disk | **PLANNED** | Required to test accumulation, fronts, and limit cycles under physical feeding | Coupled mass+total-energy IMEX evolution not implemented |

## Frozen Target Under Review

```text
Mdot_inner/Mdot_Edd = 5
Rout                 = 335 rg
Rinj                 = 240 rg
stream fraction      = 0.80
source shape         = compact C2
wind formulation     = local Mdot
eta_E                = 98.125
N                    = 164
```

## Most Important Findings

1. The standard no-wind high-rate benchmark is solid; the present obstruction
   is not failure of the underlying slim-disk solver.
2. A phase-space DAE representation is required in the stiff source/transition
   layer; ordinary `ln R` polynomial derivatives are incompatible there.
3. The accepted positive phase branch approaches a closure-dependent formal
   low-velocity singular limit, but the 1D radial/vertical separation fails
   before the limit.
4. Independent outer branches reach the physical validity boundary, including
   a conservative near-match, but no strict state-and-flux connection has been
   certified at fixed `lambda0`.
5. A new unified conservative solver explicitly specifies `l_s`, `B_s`,
   `l_w`, and `B_w`, and separates external torque from external power. Its
   first stream-fed wind branch is physical but only weakly mass loaded.
6. Exact compact-source moments and 8/16-point transport audits remove source
   quadrature as the low-`eta_E` explanation. An interval-local Jacobian and
   bordered/direct continuation recover mesh-supported roots through `eta_E=8`.
7. Power-primary wind energy transport is algebraically equivalent to the old
   carried-energy ledger. At eta=8 the mass-weighted wind Bernoulli is about
   `0.102 c^2`, so lower eta is not needed to make the wind escape.
8. A target-terminal-Bernoulli closure reaches `B_infinity=0.02 c^2` with
   mesh-stable `6.9%` wind loss. `f_adv_global` rises only from about `0.400` to
   `0.414`, while `H/R`, luminosity, and sonic radius remain nearly fixed.
9. The independent-Sigma signed-flux core now includes the physical stream
   angular moment in the steady solution. The ideal wall accretes all supply
   and requires torque `0.7689866` of the stream angular flux; the open edge
   accretes `17.006%`, overflows `82.994%`, and stagnates near `222.35 rg`.
10. The angularly closed tidal-wall hot-reservoir candidate survives with
    internal-energy export `0.548`, `H/R=0.341`, and `Lrad=1.323 LEdd`. At
    `10 rg`, radial pressure support is `0.116`, so the Keplerian reservoir
    cannot be interpreted down to the current inner boundary.

## Claims That Are Not Allowed Yet

- “A strong advective/hot mass-loaded-wind branch has been recovered.”
- “The branch ends physically at `225.52125 rg`.”
- “No global far-side steady solution exists.”
- “The current internal-energy wall state is a certified physical hot branch.”

## Next Scientific Work

1. Replace internal-energy transport with a total-energy/enthalpy ledger and
   recover the pressure-work identity used by the slim entropy equation.
2. Couple `(Mdot,J,E)` fluxes to the existing inner no-wind slim solver.
3. Implement coupled mass+energy IMEX evolution, allowing
   accumulation when no steady state exists.
4. Repeat validity-surface matching with a bordered global eigenvalue solve if
   a mesh-stable critical layer appears.

## Review Entry Points

- Equations: [`MODEL_EQUATIONS.md`](MODEL_EQUATIONS.md)
- Reproduction and archive recovery: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- Compact evidence: [`../results/README.md`](../results/README.md)
- P0 synthesis: `reports/current/CODEX_IMBH_PROJECT_REVIEW_P0_RESULTS_2026-07-10.md`
- Detailed current reports: `reports/current/`
- Historical development sequence: [`history/MILESTONES.md`](history/MILESTONES.md)
- Cleanup verification: `reports/current/CODEX_REPOSITORY_CLEANUP_RESULTS_2026-07-11.md`
- Unified conservative transport: `reports/current/CODEX_UNIFIED_CONSERVATIVE_TRANSPORT_RESULTS_2026-07-11.md`
- Exact-source certification: `reports/current/CODEX_UNIFIED_SOURCE_BAND_CERTIFICATION_RESULTS_2026-07-11.md`
- Block Jacobian and bordered eta continuation: `reports/current/CODEX_UNIFIED_BLOCK_JACOBIAN_CONTINUATION_RESULTS_2026-07-11.md`
- Wind-power and terminal-Bernoulli audit: `reports/current/CODEX_UNIFIED_WIND_POWER_ESCAPE_AUDIT_RESULTS_2026-07-11.md`
- Physical terminal-Bernoulli wind: `reports/current/CODEX_UNIFIED_TERMINAL_BERNOULLI_WIND_RESULTS_2026-07-11.md`
- Signed-flux absolute-stream baseline: `reports/current/CODEX_SIGNED_FLUX_ABSOLUTE_STREAM_RESULTS_2026-07-11.md`
- Signed-flux thermoviscous baseline: `reports/current/CODEX_SIGNED_FLUX_THERMOVISCOUS_RESULTS_2026-07-11.md`
- Signed-flux angular closure: `reports/current/CODEX_SIGNED_FLUX_ANGULAR_CLOSURE_RESULTS_2026-07-11.md`
