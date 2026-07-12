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
| Signed-flux total-energy ledger | **SUPPORTED BUT NOT FULLY CERTIFIED** | Enthalpy-compatible `W_H=Mdot(P/rho)dlnH`; direct transonic identity mismatch `6.32e-15`; four-level source-bearing FV convergence; equation residual below `3e-11` | Physical use requires a valid inner interface and calibrated tidal power |
| Total-energy reservoir to `6.1 rg` | **REJECTED** | Corrected N512 wall/open viscosity mismatches `3.72` and `1.70`, localized to near-ISCO cells | Fixed-Keplerian diffusion is invalid where `dln l_K/dlnR` is small |
| Total-energy `Rin=10 rg` interface control | **DIAGNOSTIC ONLY** | Corrected N512 wall/open converge; wall `H/R=0.311`, `Lrad=1.025 LEdd` | Pressure force exceeds `0.1` inside about `15 rg`; zero-torque interface is artificial |
| Prescribed conserved-flux inner boundary | **SUPPORTED BUT NOT FULLY CERTIFIED** | Shared `(Mdot,J,F_E)` object; wall/open transport and energy round-trip tests; incompatible wall flux rejected | No physical overlap band or coupled two-domain root yet |
| Inner/reservoir overlap audit | **DIAGNOSTIC ONLY** | Open reservoir passes `14.73-59.69 rg`; a 10% pressure sensitivity gives common wall/transonic `29.45-59.69 rg` | No common band passes the primary 5% pressure gate; absorption opacity is only bracketed |
| One-way transonic/reservoir composite | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Flux mismatch `<2.1e-16`; N128/N256 composite luminosity varies only `0.20-0.24%` across `30-60 rg` | Primitive continuity fails: integrated-pressure log mismatch remains `0.327-0.334` |
| Projected pressure-supported reservoir | **DIAGNOSTIC ONLY** | Full-support N64 roots reduce the rotation mismatch to `0.37%` with closed fluxes | No N128 case passes; pressure mismatch worsens to `0.356`; projection leaves `1.38%` force mismatch |
| Common-stress fixed-Keplerian reservoir | **DIAGNOSTIC ONLY** | Corrected `Rout=335 rg`; all `30-60 rg`, N64/N128/N256 roots close stress, energy, and conserved fluxes | Maximum primitive mismatch remains `0.20-0.30`, dominated by surface density |
| Simultaneous non-Keplerian reservoir | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Corrected `Rout=335 rg`, `R_I=40 rg`, N256 root seeds the full coupled solve | The older multi-interface canonical sweep inherited a `10000 rg` numerical buffer and is superseded for physical interpretation; inner solution remains frozen |
| Fully coupled inner/outer ideal-wall control | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Corrected finite `Rout=335 rg`; chained `96/64 -> 144/96 -> 192/128` roots; full-rank `772x772` systems at `34.97-50.05 rg`; luminosity spread `5.22e-5`; fixed-band `H/R` spread `0.284%` | Ideal zero-mass-flux wall remains a limiting control; stability, time evolution, and wind are pending |
| Binary pattern-power wall continuation | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Paired torque/power identity closes to `1.7e-16`; 25%, 50%, and 75% stages solve numerically | Tidal-band `H/R` exceeds `0.3` at 25% power and reaches `0.61`; this is a candidate transition to a thick inflow-outflow regime outside the one-zone validity domain |
| Coupled open-overflow eigenvalue | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Full-rank `96/64` and `144/96` roots; inner/stream `0.16894`, overflow `0.83106`, stagnation near `222.18 rg`, Hill-band `H/R=0.0383` | Controlled `168/112` refinement fails in outer stress/energy endpoint cells; steady branch is not mesh certified |
| Flux-primary time-dependent DAE | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Direct inner response, exact stream moments, cooling, resolved timestep comparison, eight accepted `16/8` steps, and bitwise restart | `24/16` evolved refinement fails the fixed gate; all interface remedies are closed; tide and wind remain blocked |
| Global signed conservative descriptor | **DIAGNOSTIC ONLY** | Sparse local Jacobians have zero off-pattern defect; conserved-donor outer flux closes radial/J/E consistency exactly; donor N96/N128 mapping differs by `0.00782` supply | Donor evolved N64/N96 outer flux still differs by `0.01150` supply and narrowly fails the fixed gate; reconstruction tuning is closed; characteristic/Roche boundary, column energy, and inner absorption remain blocked |

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
11. The first total-energy prototype mixed an enthalpy flux with the work term
    for internal-energy transport. The corrected identity closes at `6.32e-15`
    and moves the near-ISCO wall profile to `H/R~0.327`, `Lrad~1.280 LEdd`.
    Its N256/N512 fixed point still fails only in the invalid innermost cells.
12. Moving the corrected numerical interface to `10 rg` restores N512
    convergence, with wall `H/R=0.311` and `Lrad=1.025 LEdd`, but
    radial pressure support remains above the production gate at `12 rg` and
    falls below it by roughly `15 rg`. Inner transonic matching is mandatory.
13. The old reservoir and inner disk used different alpha-stress laws. Sharing
    `W=alpha Pi` removes most of the pressure discontinuity but leaves a
    `0.20-0.30` fixed-Keplerian density mismatch.
14. A simultaneous `(Sigma,T,Omega)` solve closes common stress, radial
    momentum, and total energy at `40-60 rg`. At `40 rg`, N256 pressure and
    rotation match to about `0.13%` and `0.05%`, while density misses by
    `5.7%`. This is the best splice candidate so far, but not a certified
    physical branch.
15. Releasing the inner entropy/eigenvalue/sonic state closes the interface.
    After correcting the reservoir edge from the inherited `10000 rg`
    numerical buffer to the physical `335 rg` minidisk, chained prolongation
    reaches Ninner192/Nouter128 with a full-rank `772x772` Jacobian. Roots at
    `34.97,40.04,44.78,50.05 rg` have luminosity spread `5.22e-5` and
    common-band `H/R` spread `0.284%`.
16. Pairing the wall torque with binary pattern-speed power makes the outer
    tidal band thick before full power is reached. At 25% differential work,
    the band reaches `H/R=0.432`; at 75% it reaches `0.609`. The full-power
    solve fails after the one-zone validity boundary has been crossed. This
    is a candidate transition to a thick inflow-outflow state, not a physical
    nonexistence result.
17. The augmented open-boundary system is square and full rank. Its open root
    processes about `16.9%` of the stream inward and overflows `83.1%`, while
    remaining thin in the Hill band. It converges through `144/96` but fails
    the controlled `168/112` endpoint refinement, activating the declared
    conservative time-evolution fallback.
18. The single endpoint-asymptotic retry also fails, closing steady open-edge
    development. The selected time-dependent architecture is the
    repository-compatible flux-primary DAE with explicit face `Mdot` and
    angular flux. Small physical prototypes are full rank and conserve all
    three global ledgers under accepted backward-Euler steps.
19. The flux-primary DAE is now coupled directly to the re-solved inner
    transonic core. It includes the absolute stream mass/angular/energy moments
    and radiative cooling. A corrected, numerically complete colored sparsity
    pattern and cross-interface first radial stencil support accepted `16/8`
    and `24/16` steps with maximum residual below `3.2e-8` and independent
    mass/angular/energy ledgers below `9e-9`.
    A resolved three-level timestep comparison converges monotonically; long
    evolution and evolved-mesh convergence are not yet certified.
20. The `16/8` control now passes eight repeated steps through
    `t/t_load=1e-7`, including a bitwise-identical restart continuation. The
    `24/16` mesh accepts two subcycled steps but rejects the third at
    `1.0466e-7`, localized in interface continuity and flux extraction while
    all three global ledgers remain below `7.5e-9`. The radial row is no longer
    controlling.
21. The final boundary-eliminated interface test makes `Sigma,T` continuity
    exact but fails the first `24/16` subcycled step at `1.271e-7` in the inner
    transonic core. It is rejected and reverted. This closes further splice
    conditioning and blocks tide and long evolution in the present hybrid DAE.

## Claims That Are Not Allowed Yet

- “A strong advective/hot mass-loaded-wind branch has been recovered.”
- “The branch ends physically at `225.52125 rg`.”
- “No global far-side steady solution exists.”
- “The current internal-energy wall state is a certified physical hot branch.”
- “The total-energy `Rin=10 rg` control is already a physical inner match.”

## Next Scientific Work

1. Stop further hybrid-interface conditioning. Do not scan residual weights or
   relax the `1e-7` gate.
2. Keep the split IMEX, grouped colored Jacobian, and iterative LSMR pilots
   closed. Use certified local columns with exact factorization.
3. Close open-face reconstruction tuning. The characteristic audit finds one
   incoming acoustic mode at the deeply subsonic `335 rg` edge, but that edge
   is only `0.4485 R_H`, not a Roche saddle, and no exterior thermodynamic
   state is declared. Obtain one exterior invariant from Layer 1 or implement
   a Hill/Roche overflow layer before repeating the bounded N64/N96 comparison.
   Do not substitute a mesh-dependent pressure target or vacuum ghost state.
4. Treat WP2 column energy as complete: the enthalpy-compatible radial and
   temporal work terms pass manufactured, identity, physical tiny-step, and
   independent-ledger gates.
5. Treat WP3 inner absorption as complete for the reference-state preflight.
   The actual edge is subsonic with one incoming acoustic mode; the new
   characteristic projection removes only that mode, preserves outgoing
   perturbations and all four flux ledgers, and leaves the physical accretion
   fraction effectively unchanged.
6. Treat WP4 finite-volume energy conditioning as complete. A fixed
   mass-weighted mechanical reference removes the cell-average/center-point
   contamination without floors; all N16-N128 mappings recover positive
   internal energy, 32/64-point quadrature agrees below `1.2e-3`, and the
   selected N64/N96 evolved outer-flux difference passes at `0.00635` supply.
   The remaining blocker is the physical exterior invariant or Hill/Roche
   overflow layer, not numerical remapping.
7. Continue one physical distributed tide only after the global no-tide model
   passes; search for accumulation, fronts, hot phases, and limit cycles.
8. Add wind only after the tidal and stability gates pass.

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
- Signed-flux total energy: `reports/current/CODEX_SIGNED_FLUX_TOTAL_ENERGY_RESULTS_2026-07-11.md`
- Enthalpy identity correction: `reports/current/CODEX_SIGNED_FLUX_TOTAL_ENERGY_IDENTITY_CORRECTION_RESULTS_2026-07-11.md`
- Prescribed inner flux interface: `reports/current/CODEX_PRESCRIBED_INNER_FLUX_INTERFACE_RESULTS_2026-07-11.md`
- Inner/outer overlap audit: `reports/current/CODEX_INNER_OUTER_OVERLAP_AUDIT_RESULTS_2026-07-11.md`
- Two-domain interface sweep: `reports/current/CODEX_TWO_DOMAIN_INTERFACE_SWEEP_RESULTS_2026-07-11.md`
- Pressure-supported interface pilot: `reports/current/CODEX_PRESSURE_SUPPORTED_INTERFACE_PILOT_RESULTS_2026-07-11.md`
- Common stress and simultaneous reservoir: `reports/current/CODEX_COMMON_STRESS_AND_SIMULTANEOUS_RESERVOIR_RESULTS_2026-07-11.md`
- Fully coupled rank prototype: `reports/current/CODEX_COUPLED_INNER_OUTER_RANK_PROTOTYPE_RESULTS_2026-07-11.md`
- Coupled mesh/interface certification: `reports/current/CODEX_COUPLED_MESH_INTERFACE_CERTIFICATION_RESULTS_2026-07-11.md`
- Coupled wall pattern-power gate: `reports/current/CODEX_COUPLED_WALL_PATTERN_POWER_RESULTS_2026-07-11.md`
- Coupled open-overflow eigenvalue: `reports/current/CODEX_COUPLED_OPEN_OVERFLOW_RESULTS_2026-07-11.md`
- Flux-primary time DAE selection: `reports/current/CODEX_TIME_DAE_BOUNDARY_AND_FLUX_PRIMARY_RESULTS_2026-07-12.md`
