# Project Status

- Updated: 2026-07-18
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
| Global signed conservative descriptor | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | N64/N96/N128 reach shared `1e-6 t_load`; inner-flux spread is `0.56%` of supply, `H/R` spread `0.048%`, accepted residuals `<1.1e-11`, and overflow remains zero | The duration remains far below loading/thermal/viscous times; tide and wind remain blocked |
| Gas-radiation Hill/Roche boundary | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Shared EOS entropy/enthalpy/acoustic closure; exact `335 rg` reconstruction; continuous closed/choked flux; disk/Jacobi/pattern-power ledgers; N64/N96/N128 and filling-factor preflight | All mapped states are energetically closed; the reduced symmetric local-Hill channel is not a multidimensional L1/L2 solution and its open-state filling remains uncertain |
| Causally outgoing inner plunge boundary | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Same-equation continuation from `5.210237` to `4.5 rg`; stationary Mach `-9.45`; zero incoming modes; N64/N96/N128 pass mapping, tiny-step, restart, and shared `1.001e-6 t_load` gates | N64 remains regular through `1.430993e-6`, but the `2.1e-6` duration gate is blocked by certified-Jacobian/nonlinear cost; long no-tide evolution remains incomplete |
| Global evolution diagnostic/checkpoint contract | **CERTIFIED** for its diagnostic scope | Full inner/outer `M/J/E`, fixed-radius and sonic diagnostics, controller-cell characteristics, normalized Roche active-set audit, shared time metadata, immutable checksummed milestones; 380 tests plus 4 subtests pass | Long-duration physics remains pending |
| Global solver-efficiency WP2 | **CERTIFIED** as a bounded negative result | N64 serial reference: 198 s, 600 nfev, 596 Jacobians, 153773 residual calls; gate stop gives only 1.18x; independent blocked columns only 1.11x | Both candidates were removed; production remains serial sparse-forward; N128 candidate runs were skipped after the coarse adoption gate failed |
| Exact-common-time global snapshot | **SUPPORTED BUT NOT FULLY CERTIFIED** | N64/N96/N128 land at exactly `0.15200886034168773 s`; inner M/J/E differ by at most `0.109%/0.636%/0.981%` on the selected scales; Roche flux stays zero; immutable milestones are checksummed | Fixed-radius Mach and primitives converge slowly at `4.65-4.75 rg`; files predate the WP4 converged-rate mapper correction and are not restart-compatible with it |
| Sonic-gradient and plunge-mapping WP4 | **SUPPORTED BUT NOT FULLY CERTIFIED** | Accepted `96/64 -> 144/96` gradient mismatch halves `0.3763 -> 0.1883`; offset/tolerance changes stay below `5.2e-8`; matched exact-time N64 evolution differs by only `2.53e-4` of supply in inner mass flux | Only two accepted coupled-root resolutions exist; the failed `168/112` steady root cannot provide a third stationary level |
| Source-on/source-off WP5 control | **CERTIFIED** for its diagnostic scope | Exact source tendency closes below `1.21e-16`; matched N64 runs use identical 15-step histories; inner M/J/E and `H/R` agree at numerical precision while source-on minus source-off mass equals `1.0000000049` injected increments | Duration is `1e-7 t_load`; source-off is a counterfactual trajectory, not a relaxed physical equilibrium |
| Source-free N64 relaxation WP6a | **DIAGNOSTIC ONLY** | Exact `2e-7`, `5e-7`, and `1e-6 t_load` milestones retain ledgers below `6.1e-16`, causal inner outflow, and a closed Roche edge | Relaxation gate fails; `L_v/Delta R_cell` falls to `0.900` at `1e-6`, activating the named plunge-resolution stop |
| Source-free N96 refinement and N128 remap WP6a-R | **DIAGNOSTIC ONLY** | Exact N96 milestones retain ledgers below `4.6e-16`; `L_v/Delta R_cell=1.698` at `1e-6`; the N128 remap preserves all totals below `1.2e-16` | Temporal relaxation worsens, N64/N96 fixed-radius primitives disagree, and the remap changes inner Mach by `10.34`; no source-free reference may be frozen |
| Local inner-plunge projection WP6b | **DIAGNOSTIC ONLY** | The exact supersonic prefix gives full-rank N64/N96 roots below `1.2e-14`; the source-on hold reaches exact `2e-7 t_load` without nonlinear rejection and passes all flux/thermal gates | N64 passes the complete hold gate but N96 fails the predeclared Mach-drift gate (`0.225>0.1`); the projected state is not a mesh-certified initializer |
| Low-throughput remnant WP6c | **SUPPORTED BUT NOT FULLY CERTIFIED** as an initial state; **DIAGNOSTIC ONLY** for evolution | Fresh `0.025 Mdot_Edd` transonic state maps conservatively at N64/N96 with only `0.475-0.482%` of the physical stream throughput; disk mass/loading time agree to `1.1e-6`; Roche edge is closed | Coupled half-supply correction fails; subsonic N64 characteristic hold spends 30 CPU-minutes in its first Jacobian without completing a timestep, so source-on and N96 holds remain unverified |
| Characteristic-response efficiency WP7 | **CERTIFIED** for local equivalence; **REJECTED** as an initializer unlock | Exact trace caching cuts one-Jacobian pressure roots `260 -> 5`, accelerates characteristic work `23.8x`, and reproduces the 20-evaluation trajectory exactly; 378 tests plus 4 subtests pass | Cached N64 coarse retry still ends at residual `9.83e-7 > 1e-8`; source-on, N96, tide, and wind were not launched |
| Fresh low-mass global startup WP8 | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **REJECTED** as a production initializer | Constant-`Pi` N64/N96 states close radial balance below `3.3e-13`; predicted N64 first step takes 5 evaluations; all matched N64/N96 equations and ledgers pass at exact common time | N64 retains one incoming inner mode but N96 reverses to weak outflow and requires three; the fixed one-mode boundary is not mesh invariant |
| Fresh-loading inner-boundary architecture WP9 | **CERTIFIED** as a stop decision; **REJECTED** for production evolution | Exact counts completed; the low-rate branch retains one incoming acoustic mode from `4.5` through `2.0001 rg`; the prior quasi-steady hybrid fails its refined repeated-step gate | Neither current candidate supplies a mesh-invariant causal low-throughput boundary; inner causal physics must be repaired before evolution |
| Causal inner thermodynamics WP10a | **SUPPORTED BUT NOT FULLY CERTIFIED** for diagnostic thermodynamics; **REJECTED** as a standalone production boundary | Relativistic enthalpy derivative keeps all audited sound speeds subluminal and approaches `c/sqrt(3)`; radial-only SR characteristics reach zero incoming modes at `2.0001 rg` | The PW profile has `v_phi>c` by `3 rg` and no subluminal full-state excision; its stationary equations and global flux remain Newtonian |
| Horizon-penetrating Valencia core WP10b | **SUPPORTED BUT NOT FULLY CERTIFIED** as an architecture; **DIAGNOSTIC ONLY** physically | Ingoing-Kerr-Schild flux includes transverse rotation; analytic/numerical eigenvalues agree to `9.71e-11`; stationary rank changes `4 -> 3` at one acoustic critical mode; all 342 sampled inside-horizon states have zero incoming modes | Local ideal-gas column prototype only; gas+radiation recovery, geometric sources, stress, radiation, stream/Roche migration, and stationary roots are pending |
| Valencia gas+radiation primitive recovery WP10c1 | **SUPPORTED BUT NOT FULLY CERTIFIED** for the local map; **DIAGNOSTIC ONLY** physically | Nine rotating gas-to-radiation states give primitive/conserved defects `7.42e-11/6.46e-15`, characteristic defect `1.94e-8`, causal sound speed, and zero inside-horizon incoming modes | Fixed-height EOS chart only; geometric sources, vertical closure, stress, radiation, stream/Roche migration, stationary roots, and evolution remain pending |
| Kerr-Schild geometric finite volume WP10c2 | **SUPPORTED BUT NOT FULLY CERTIFIED** for source-free geometry; **DIAGNOSTIC ONLY** physically | Direct/3+1 source identities close below `4.85e-15`; flat pressure balance closes at `9.54e-16`; horizon-crossing free fall converges at order `1.997-2.000` with mass/Killing-energy flux spreads below `1.9e-15` | Equatorial `2+1` geometry only; supplemented by WP10c3a stress and WP10c3b thermal sources, while stream/Roche migration, stationary roots, and evolution remain pending |
| Causal relativistic alpha shear WP10c3a | **SUPPORTED BUT NOT FULLY CERTIFIED** for the local stress/flux contract; **DIAGNOSTIC ONLY** physically | Nine gas/radiation states have real acoustic/contact/shear spectra with zero light-cone excess; all inside-horizon modes leave the domain; tensor and paired torque/Killing-work identities close below `8.67e-16` | Its fixed-height thermodynamic chart is superseded by WP10c3b; `c_nu=sqrt(alpha)a` and the final nonlinear coupled characteristic structure remain unvalidated |
| Responsive-height thermal ledger WP10c3b | **SUPPORTED BUT NOT FULLY CERTIFIED** for the local thermal/source contract; **DIAGNOSTIC ONLY** physically | Nine bounded dynamic-height states recover below `5.35e-13`; the vertical-work acoustic matrix closes below `8.33e-17`; comoving cooling/work identities close below `1.16e-15`; source integration is second order | `Omega_perp` is a supplied provisional closure; no full nonlinear characteristic proof, stream/Roche migration, stationary root, or timestep |
| Kerr-Schild stream/Roche migration WP10c4 | **SUPPORTED BUT NOT FULLY CERTIFIED** for the source/boundary adapter; **DIAGNOSTIC ONLY** physically | Exact C2/C4 four-state moments close below `2.06e-16`; source-per-`ct` conversion below `1.80e-16`; the closed/choked Roche edge has one incoming acoustic mode and exact angular/Killing/pattern-power ledgers; outer face-row rank is four | Circular stream state is a regression fixture, local Hill force remains reduced PW+Hill, and the final causal-stress augmented nonlinear count, stationary root, and timestep remain pending |
| Five-field causal DAE preflight WP10c5 | **SUPPORTED BUT NOT FULLY CERTIFIED** for local count/rank; **DIAGNOSTIC ONLY** physically | Exact count `15N+5`; covariant shear recovers `-R dOmega/dR`; five real responsive acoustic/contact/shear modes; zero inner incoming modes; two independent outer incoming responses; temporal Killing-storage transform closes at `2.74e-16` | Superseded operationally by the assembled WP10c5b gate; no root was authorized by this local preflight alone |
| Five-field causal DAE assembly WP10c5b | **SUPPORTED BUT NOT FULLY CERTIFIED** for the time-dependent descriptor; **REJECTED** as a stationary-root unlock | Complete five-field residual closes primitive/face maps exactly and telescopes at `1.78e-16`; descriptor rank is `80/80`; backward-Euler rank is `245/245` at `0.1-10 s` | Stationary N16 rank is stably `244/245`; its outermost thermal/stress mode blocks N64/N96 roots and the tiny step under the locked gate |
| Reduced primitive null audit WP10c5c | **CERTIFIED** for algebraic reduction and rank classification; **DIAGNOSTIC ONLY** physically | The algebraic block is `165/165` with condition approximately one; direct and Schur reduced operators agree to `2.99e-11`; the primitive response is `80/80`, the outer thermal/stress response is `2/2`, and the same provider remains full rank when the Roche channel opens | The audited seed is not a stationary root; the weak outer mode is nonzero and cannot establish equilibrium, marginality, or stability |
| Consistent initial step WP10c5d | **SUPPORTED BUT NOT FULLY CERTIFIED** for index-one initialization; **REJECTED** as an evolution unlock | The consistency matrix is `245/245`, descriptor rows are `80/80`, and storage/algebraic tangent defects are below `9.1e-15` | Both bounded N16 steps stop above the unchanged `1e-8` gate (`4.79e-6`, `1.40e-6`), localized to outer-cell temporal storage; N32, roots, tide, wind, and physical evolution remain blocked |
| Temporal-storage increment WP10c5e | **CERTIFIED** for the exact-map path identity; **REJECTED** as an evolution unlock | Endpoint/path rate defects reach `7.05e-6`; the path converges below `2.53e-9` and telescopes below `5.60e-17`; tiny `Delta lnSigma` recovery improves by over six orders of magnitude | Path-integrated N16 steps still stop at `3.77e-6` and `1.42e-6` with reduced-Newton condition near `1.03e10`; N32 and all physical searches remain blocked |
| Reduced linear precision WP10c5f | **CERTIFIED** for the frozen linear-solve audit; **REJECTED** as an evolution unlock | LAPACK equilibration reduces condition `1.03e10 -> 27.5`; direct/refined corrections agree to `2.62e-14`; linear residuals are `1.49e-16`; second/fourth-order Jacobians agree to `4.47e-12` | Every correction gives nonlinear residual `3.35e-6`; no recoverable precision is demonstrated, so N16 is not repeated and N32 remains blocked |
| Residual directional consistency WP10c5g | **CERTIFIED** for component diagnosis; **REJECTED** as an evolution unlock | Flux, source, and height-work directional defects are at most `2.09e-13`; residual/component identities close below `2.0e-16`; path conserved storage is the unique failing block | One coordinate-Jacobian storage repair still leaves N16 at `1.42473e-6`; no second repair or N32 attempt is authorized |
| Increment-primary causal startup WP10c5h | **SUPPORTED BUT NOT FULLY CERTIFIED** for source-free startup; **DIAGNOSTIC ONLY** physically | Direct `Delta U/(c Delta t)` storage gives full equilibrated rank `245/245` and `485/485`; N16/N32 residuals are `8.80e-9/3.64e-9`; relative full/two-half errors are `2.76e-6/1.01e-6` | Only one bounded step from a nonstationary source-free seed; repeated source-on evolution, stability, tide, wind, and hot-state claims remain blocked |
| Exact-stream causal startup WP10c5i | **SUPPORTED BUT NOT FULLY CERTIFIED** for source-on startup; **DIAGNOSTIC ONLY** physically | Exact C2 moments normalize at roundoff; N16/N32 residuals are `8.80e-9/3.64e-9`; full/two-half errors are `5.63e-6/1.21e-6` | Circularized regression stream, not ballistic Layer-1 input; only one tiny step from a nonstationary seed |
| Colored sparse causal backend WP10c5j | **CERTIFIED** for numerical parity | Exact 18-color pattern; omitted derivatives and colored/dense matrix defects are zero; directional defects `<7.1e-16`; root defects `<9.3e-18`; evaluations fall `490/970 -> 36` | Finite-difference sparse direct backend, not an analytic Jacobian or long-duration performance result |
| Repeated causal source-on startup WP10c5k | **SUPPORTED BUT NOT FULLY CERTIFIED** for short no-tide startup; **DIAGNOSTIC ONLY** physically | N16/N32 reach exact `3.39278e-7 s`; no rejected steps; mass defects `6.32e-13/8.14e-12`; bitwise restart; common-radius `Delta ln(H/R)` mesh difference `2.05e-3` | Only about `2e-13 t_load`; arbitrary seed relaxation has inner flux about `9.2e4` times supply; long evolution, tide, wind, stability, hot state, and cycle remain blocked |
| Matched causal source control WP10c5l | **CERTIFIED** for differential source isolation; **DIAGNOSTIC ONLY** physically | Lockstep N16/N32 exact-time controls recover the four prescribed stream moments to `3.25e-6/1.08e-6`; isolated mass and `H/R` response mesh defects are at most `1.38e-9/1.03e-9` | Regression stream and only `1.73e-7 s`; the stress field has no direct stream moment; no physical relaxation is established |
| Source-compatible causal startup WP10c5m | **SUPPORTED BUT NOT FULLY CERTIFIED** for bounded no-tide startup; **DIAGNOSTIC ONLY** physically | Exact unit inner throughput, `H/R=0.1`, `tau=18.5`, zero inner incoming modes, closed Roche edge, and full rank; equal-time N16/N32 startup reaches `5.542e-5 s` with mass defects `6.59e-13/1.57e-11` and response mismatch `1.00e-3` | Constructed datum, circularized source, and only about `1e-10 t_load`; duration, stability, tide, wind, hot-state, and cycle claims remain blocked |

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
22. The global one-domain WP0 audit separates conserved cell-average energy
    from physical face Bernoulli energy. Nonzero mechanical-offset boundary
    corrections are continuous, the numerical physical-flux eigensystem
    agrees with the analytic acoustic rule, and restart files now preserve the
    full offset, mesh, hashes, and provenance. Layer 1 has no exterior
    thermodynamic invariant at `335 rg`, so the selected next boundary is one
    adiabatic Hill/Roche overflow side channel ending at a real saddle.
23. The Hill/Roche side-channel now uses the exact shared gas+radiation
    entropy, enthalpy, and adiabatic acoustic derivative. It is coupled at
    exactly `335 rg` through a continuous closed-to-choked conservative edge,
    with one incoming acoustic condition, pressure traction, zero outer
    viscous torque, and explicit disk/Jacobi/binary ledgers. The mapped
    N64/N96/N128 deficits are `8.57e16`, `8.76e16`, and `8.93e16 erg/g`.
    Filling factors `0.25-1.0` cannot change the closed classification. The
    no-tide evolution must therefore begin with accumulation rather than an
    imposed donor overflow.
24. The first physical-edge loading step passes at N64/N96/N128. About
    `16.7-16.9%` of the supply initially accretes inward, zero mass crosses the
    closed Roche edge, and the remaining `83.1-83.3%` accumulates exactly as
    required by the mass ledger. Full/half-step differences are negligible.
    N64 accepts direct steps through `1e-7 t_load` but rejects `1e-6 t_load`,
    so the next implementation is adaptive stepping plus restart, not tide.
25. Adaptive backward-Euler continuation now rejects both unconverged roots
    and accepted roots that exceed declared `Sigma`, temperature, or `H/R`
    change gates. An N64 run reaches `5e-7 t_load` in eight accepted steps and
    two recovered retries while reloading a checksum-verified checkpoint after
    every step. The Roche deficit remains strongly negative. This validates
    the controller, not a long-time physical state.
26. At shared `1e-7 t_load`, N64/N96/N128 inner fluxes agree within `0.26%`
    of supply and maximum `H/R` within `0.05%`. N128 rejects one full jump and
    reaches the target through two accepted half steps. The resumed N64 state
    reaches `1e-6 t_load`; its inner fraction rises to `0.1918`, the remainder
    accumulates, and the Roche edge stays closed. The next gate is N96/N128 at
    this same time, not a longer single-mesh claim.
27. N64/N96/N128 now also reach shared `1e-6 t_load`. The inner fractions are
    `0.1918`, `0.1880`, and `0.1863`; maximum `H/R` agrees within `0.048%`,
    accumulated mass within `0.3%`, and all Roche fluxes remain zero. N96 and
    N128 recover their rejected trials through adaptive halving. This passes
    the preliminary mesh gate but remains many orders of magnitude short of a
    physical loading time.
28. The N64 continuation reaches `2e-6 t_load` cleanly, but a bounded
    `5e-6` request stops after 60 new accepted states at
    `3.9166e-6 t_load`. Retained residuals remain below `5.5e-12`, all ledgers
    close, and the Roche edge stays closed. The timestep nevertheless falls to
    `5.27e-9 t_load` because the first three cells at `5.38-6.13 rg` approach
    the fixed 2% thickness-change gate. The inner Mach number changes from
    `-0.654` to `-0.148`, and the fixed-reference incoming acoustic correction
    reaches `1.388` times the reference sound speed. This is outside the
    absorber's certified linear regime. The checkpoint is retained as a
    boundary-validity witness, not a long-time physical state.
29. The accepted transonic solution now continues inward with the same local
    equations from `5.210237` to `4.5 rg`. The stationary face has Mach
    `-9.452` and zero incoming radial characteristics. Conservative
    N64/N96/N128 mappings pass full/half tiny steps and a shared
    `1e-7 t_load` adaptive gate without any inner projection. All three meshes
    now pass a shared persisted `1.001e-6 t_load` gate. N96 and N128 differ by
    `1.43e-4` of supply in inner flux and `1.27e-4` in maximum `H/R`; N128 is
    causally outgoing at Mach `-50.89`. The old absorber breakdown is removed.
30. The target-time controller now distinguishes roundoff remainders from a
    physically finite last step and permits exact final landing below the
    ordinary minimum timestep. A bounded N64 extension remains regular through
    `1.430993e-6 t_load`, with Mach `-52.07`, zero incoming modes, and zero
    Roche flux. It does not reach `2.1e-6`: accepted evolution requires the
    600-evaluation certified sparse-forward solve, while 100/200-evaluation
    probes fail the unchanged residual gate. Jacobian/nonlinear efficiency is
    therefore the next numerical prerequisite.
31. Global evolution reports now separate physical seconds, mesh loading time,
    and a shared N128 reference loading time. They expose complete inner and
    outer conserved fluxes, fixed-radius plunge states, the emergent sonic
    crossing, controller cell/characteristics, and a normalized Roche
    active-set residual. Immutable milestones include Git, state, reference,
    and mechanical-offset hashes. A zero-step N96 production audit finds the
    sonic crossing at `5.23284 rg`, three cells inside it, `L_v/dR=1.70`,
    `L_v/H=2.02`, and a safely closed normalized Roche margin of `-0.0864`.
32. The bounded solver-efficiency audit profiles one immutable N64 next step at
    198 seconds, 600 nonlinear evaluations, 596 Jacobian assemblies, and
    153773 residual evaluations. Gate-aware stopping reaches the unchanged
    residual and ledger gates but gives only `1.18x`; independent blocked
    columns give only `1.11x`. Both candidates are removed. Production remains
    serial sparse-forward, now with persistent nonlinear work telemetry.
33. A regenerated N64/N96/N128 snapshot lands at the exactly shared physical
    time `0.15200886034168773 s`. Global inner fluxes, accumulated mass, and
    maximum thickness remain mesh supported, and the Roche edge is safely
    closed. Fixed-radius diagnostics expose slower local convergence at
    `4.65-4.75 rg`; N96/N128 Mach differs by `0.38` at `4.65 rg` but only
    `0.027` at `5 rg`. Sonic-gradient claims therefore remain a WP4 question.
34. WP4 finds that the sonic-gradient mismatch halves from `0.3763` to
    `0.1883` between the accepted `96/64` and `144/96` roots while the plunge
    trajectory is already stable: at `4.5 rg`, Mach differs by `0.0263` and
    `Delta ln Sigma=-0.00314`. A matched `1e-9 t_load` N64 step changes the
    inner mass flux by only `2.53e-4` of supply and selects the same
    causally-disconnected first-cell thickness controller. The audit also
    corrects the mapper to use the accepted trial accretion rate and centers
    the regular-root scan on the physical outer gradient.
35. WP5 shows that the early inner evolution is mapping/operator relaxation.
    Source-on and source-off N64 trajectories choose identical 15-step and
    four-retry histories through exact `1e-7 t_load`; inner M/J/E, sonic
    position, maximum thickness, and controller location agree at numerical
    precision. The stream nevertheless adds exactly its prescribed mass,
    angular momentum, and energy in the distant source annulus. A relaxed
    source-free reference and controlled stream ramp must precede the physical
    loading clock.
36. The bounded N64 source-free relaxation reaches `1e-6 t_load` but cannot be
    frozen as a reference. Inner mass flux moves toward `-0.1955` of supply,
    while fixed-radius Mach changes by as much as `2.34` between the final two
    milestones and `L_v/Delta R_cell` drops to `0.900`. Conservation and Roche
    gates remain excellent. This is a named local-resolution stop requiring
    one N96 refinement, not a physical instability claim.
37. N96 resolves the local layer at `1e-6 t_load` with
    `L_v/Delta R_cell=1.698`, but it does not relax: the final interval changes
    fixed-radius Mach by `8.37`, temperature by `0.172` in log space, and the
    angular/energy fluxes by about `5.5%`. A conservative N128 remap preserves
    every total below `1.2e-16` yet shifts the innermost Mach by `10.34`.
    More fundamentally, a source-free state with inward throughput and a
    closed Roche edge must drain and cannot be a nontrivial global equilibrium.
    The next baseline must be physically source compatible.
38. A global source-balanced steady projection is impossible under the
    selected no-tide contract: the Roche edge is closed, only `15-16%` of the
    supply initially accretes, and neither tide nor wind removes the remaining
    mass and angular momentum. Projecting only the causally outgoing
    supersonic prefix gives machine-accurate full-rank roots at N64/N96. Its
    source-on hold passes at N64, but N96 fixed-radius Mach drifts by `0.225`
    against the declared `0.1` gate. The local projection is retained as a
    diagnostic, not adopted as the production initializer.
39. WP6c closes the coupled low-supply deformation after its fixed first
    factor-of-two stage fails at residual `1.1826`. A fresh standard
    transonic sequence nevertheless yields cross-mesh low-throughput remnants:
    N64/N96 carry `0.00475-0.00482` of the eventual stream rate, retain a
    closed Roche edge, and agree in disk mass and loading time to `1.1e-6`.
    Their inner edge is subsonic, so the one-incoming-mode characteristic
    boundary is required. Its first N64 source-off implicit step does not
    complete within a fixed 30 CPU-minute ceiling because sparse Jacobian
    perturbations repeatedly solve the boundary thermodynamic root. The state
    is physically plausible but not yet an evolution-certified initializer.
40. WP7 confirms that repeated characteristic pressure roots consume `25.23%`
    of one reference sparse-forward Jacobian. An exact bounded trace cache
    reduces pressure-root calls from `260` to `5`, makes the local map `23.8x`
    faster, and reproduces the 20-evaluation nonlinear trajectory exactly.
    The complete coarse retry nevertheless stops after 600 evaluations at
    residual `9.8312e-7`, above the unchanged `1e-8` gate. Characteristic work
    is then only `1.406%` of Jacobian time. The local optimization is retained,
    but the low-throughput remnant is rejected as an implicit initializer under
    the present complete global solve.
41. WP8 constructs a fresh constant-pressure, optically thick global datum
    directly on N64/N96. A conservative explicit predictor used only as the
    implicit initial guess reduces the N64 first step to five evaluations with
    residual `9.27e-14`. N64 source-on/source-off holds pass at exact
    `2e-7 t_load`; N96 equations and ledgers also pass. The N96 first cell,
    however, reaches Mach `+1.05e-4` while N64 remains at `-1.27e-4`.
    The incoming characteristic count changes from one to three. Source-on and
    source-off are identical at the inner edge, so this is a mesh-dependent
    boundary/initial relaxation. The startup is not adopted and no longer run,
    tide, or wind is authorized.
42. WP9 completes the required inner-boundary rank decision. The one-domain
    global system has `4N` backward-Euler unknowns and rows and needs no inner
    boundary row only when zero characteristics enter. The accepted
    `0.025 Mdot_Edd` stationary branch never reaches that regime: it retains
    one incoming acoustic mode from `4.5` through `2.0001 rg`, while its
    Newtonian radiation sound speed becomes superluminal near the
    Paczynski-Wiita singularity. The alternative `2Ni+5No+5` quasi-steady
    hybrid is ADR 0012 and already fails refined repeated evolution. Neither
    candidate is selected; tide, wind, and longer loading remain blocked.
43. WP10a replaces the acausal Newtonian high-temperature acoustic derivative
    with the relativistic enthalpy derivative
    `a^2=c^2(dP/d rho)_s/(c^2+e+P/rho)`. The low-rate audit remains subsonic
    from `4.5` through `2.001 rg` and first reaches zero incoming modes at
    `2.0001 rg`, where `v_R/c=-0.8615` and `a/c=0.57735`. The old PW
    azimuthal speed exceeds `c` by `3 rg` and reaches `357c` at that radial
    crossing. This repairs the local EOS causality defect without a cap, but
    it confirms that a complete conservative causal system is required before
    another trajectory.
44. WP10b selects one ingoing-Kerr-Schild Schwarzschild Valencia column
    system from an inner excision inside `2 rg` to the Hill/Roche edge. Its
    rotating analytic characteristics match the numerical conservative-flux
    Jacobian to `9.71e-11`; the stationary matrix loses exactly one rank when
    one acoustic speed vanishes. At `1.9` and `1.5 rg`, all 342 sampled
    physical states have zero incoming inner modes. The exact flux-primary
    count is `12N+4` unknowns and rows with zero physical inner boundary rows.
    This locks the architecture, not a production disk solution.
45. WP10c1 implements the gravity-independent fixed-height gas+radiation
    column EOS and pressure-root Valencia primitive recovery. Across nine
    rotating states at `20`, `4.5`, and `1.8 rg`, the maximum primitive,
    conserved, and characteristic defects are `7.42e-11`, `6.46e-15`, and
    `1.94e-8`; all three inside-horizon states have zero incoming modes. The
    fixed height is only a thermodynamic chart. It does not select vertical
    equilibrium or authorize a stationary/evolution run.
46. WP10c2 selects the stationary Killing-energy chart
    `E_K=alpha(tau+D)-beta^R S_R`, exact proper column measures, and the
    covariant radial geometric source. Twelve gas/radiation identity states
    close below `4.85e-15`; flat cylindrical pressure closes at `9.54e-16`;
    and radial dust free fall through the horizon converges at second order
    while preserving mass and Killing-energy fluxes below `1.9e-15`. This is
    a source-free `2+1` column result, not a stationary disk.
47. WP10c3a transforms one rest-frame `R-phi` stress through the same
    Kerr-Schild tensor and pairs its angular flux with Killing power. A
    Maxwell-Cattaneo shear law recovers `W=alpha Pi` at the reference shear
    while propagating transverse modes at finite `c_nu=sqrt(alpha)a`. Nine
    states have real causal spectra, zero inside-horizon incoming modes, and
    tensor/work defects below `8.67e-16`. The rejected pressure-amplitude-only
    control retains a step-stable complex pair with
    `max |Im lambda| >= 6.66e-5`; instantaneous `alpha Pi` is therefore only
    an equilibrium calibration, not a causal time-dependent closure.
48. WP10c3b replaces the fixed-height chart with a responsive
    `H(Sigma,T,Omega_perp)` gas+radiation column and includes vertical pressure
    work in the acoustic principal matrix. Nine bounded states recover below
    `5.35e-13`; acoustic/shear defects stay below `8.33e-17`; inside-horizon
    states retain zero incoming modes; and comoving cooling/vertical-work
    identities close below `1.16e-15`. Midpoint Killing-source integration is
    second order with N128 error `1.16e-5`. The supplied vertical frequency
    remains a physical closure input, and no disk root or timestep has run.
49. WP10c4 migrates one immutable stream four-state and the physical
    closed/choked Hill/Roche contract into the Kerr-Schild Killing chart.
    Exact compact C2/C4 cell moments close below `2.06e-16`; conversion from
    physical rates to the `x^0=ct` source closes below `1.80e-16`; and the
    fiducial edge retains exactly one incoming acoustic mode. The nozzle uses
    the relativistic edge Killing energy and flux angular momentum while a
    constant potential shift preserves the existing reduced Hill force and
    opening gate. The four base face rows remain full rank at N16-N128. This
    is an adapter/rank result, not a ballistic stream calibration, stationary
    disk, or final causal-stress characteristic proof.
50. WP10c5h reparameterizes the same complete `15N+5` DAE in
    `(Delta U,Delta p,Delta F)`, so conserved storage enters backward Euler
    directly. Equilibrated N16 and N32 systems are `245/245` and `485/485`;
    both bounded steps pass below `8.8e-9`, and one-full-step versus
    two-half-step differences are only `2.76e-6` and `1.01e-6` of the full
    change. This unlocks short source-on no-tide startup work, not a physical
    relaxation, hot-state, or stability claim.
51. WP10c5i adds one exact compact-C2 circularized regression stream at
    `240 rg` and `5 Mdot_Edd`. Its mass, radial momentum, angular momentum,
    and Killing-energy moments normalize exactly. N16/N32 source-on steps pass
    below `8.8e-9`, and full/two-half temporal errors remain
    `5.63e-6/1.21e-6`. This is not yet a ballistic Layer-1 source.
52. WP10c5j certifies the complete DAE's exact nearest-neighbor sparsity and an
    18-color central Jacobian. It reproduces every dense derivative and root
    while reducing one Jacobian from `490/970` residual evaluations at
    N16/N32 to `36`. Max-norm-equilibrated sparse LU preserves the accepted
    decisions and full equilibrated ranks.
53. WP10c5k reaches the exact shared time `3.392784696e-7 s` in eight N16 and
    seven N32 accepted steps with no retries, cancellation-safe mass defects
    `6.32e-13/8.14e-12`, and bitwise restart replay. The common-radius
    baseline-subtracted `Delta ln(H/R)` response differs by `2.05e-3`, below
    its `5e-3` gate. The raw seed maxima differ by `6.8%` because the physical
    nozzle-compatible endpoint datum is anchored at moving cell centers; that
    remains an explicit non-gating diagnostic. The duration is only about
    `2e-13 t_load`, and the seed's inner flux is about `9.2e4` times the
    stream supply, so longer physical evolution remains blocked pending a
    source-compatible causal initializer.
54. WP10c5l evolves bitwise-identical source-on/source-off pairs through the
    same accepted timestep histories. The four prescribed stream moments are
    recovered to `3.25e-6/1.08e-6` at N16/N32, while stored-mass and
    baseline-subtracted thickness response differences across the meshes stay
    below `1.39e-9`. The relaxing-stress field remains an explicit residual
    audit but is not falsely counted as a fifth injected stream moment.
55. WP10c5m replaces the arbitrary high-throughput preflight seed with a
    constraint-consistent datum satisfying
    `|Mdot_inner|/Mdot_stream=1` and `H_inner/R_inner=0.1`. Both meshes retain
    zero inner incoming modes, a closed Roche channel, scattering depth above
    `18.5`, exact maps, and full scaled/equilibrated rank. Seven or eight
    accepted equal-time steps reach `5.542012666e-5 s`; the N16/N32
    baseline-subtracted thickness response differs by `1.00e-3` and the
    aggregate mass defects remain below `1.57e-11`. This authorizes one
    bounded geometric duration extension, not a hot, stable, or cyclic
    physical interpretation.

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
3. Treat the physical Roche edge as complete for the no-tide preflight. It
   starts closed, retains pressure traction, opens only when the Jacobi gate
   becomes positive, and rejects inconsistent angular/energy ledgers. Do not
   restore donor overflow, a vacuum ghost state, or a fitted pressure target.
4. Treat WP2 column energy as complete: the enthalpy-compatible radial and
   temporal work terms pass manufactured, identity, physical tiny-step, and
   independent-ledger gates.
5. Treat WP3 inner absorption as complete only for the reference-state
   preflight.
   The actual edge is subsonic with one incoming acoustic mode; the new
   characteristic projection removes only that mode, preserves outgoing
   perturbations and all four flux ledgers, and leaves the physical accretion
   fraction effectively unchanged. Long continuation has now left its linear
   regime, so the fixed-reference absorber is closed for production evolution.
6. Treat WP4 finite-volume energy conditioning and the WP0 energy convention
   as complete for the mapped-state preflight. A fixed
   mass-weighted mechanical reference removes the cell-average/center-point
   contamination without floors; all N16-N128 mappings recover positive
   internal energy, 32/64-point quadrature agrees below `1.2e-3`, and the
   pre-WP0 selected N64/N96 evolved outer-flux difference passed at `0.00635`
   supply; that production pair must now be regenerated with the physical
   nozzle boundary.
   Physical face Bernoulli fluxes exclude the quadrature offset, while stored
   conservative energy and numerical dissipation retain their declared roles.
   The Hill/Roche layer now passes its preflight; the remaining gate is the
   no-tide loading evolution, not another boundary reconstruction.
7. Treat the causally outgoing plunge architecture, shared
   `1.001e-6 t_load` mesh gate, WP1 diagnostic contract, and bounded WP2
   solver audit as complete. Keep the serial sparse-forward production
   backend and its new work telemetry; do not start a third optimization
   architecture.
8. Treat WP9, WP10a, WP10b, WP10c1, source-free WP10c2 geometry, the local
   WP10c3a/WP10c3b stress and thermal contracts, the WP10c4 stream/Roche
   adapters, the bounded WP10c5 count/rank preflight, the WP10c5b
   assembled-residual stop decision, the WP10c5c reduced audit, the
   WP10c5d consistent-data gate, the WP10c5e storage audit, the WP10c5f
   frozen linear-precision audit, the WP10c5g component audit, and the
   WP10c5h increment-primary startup, WP10c5i exact-stream startup, WP10c5j
   sparse parity backend, WP10c5k repeated startup, WP10c5l matched source
   control, and WP10c5m source-compatible startup as complete. The old PW
   plunge has superluminal transverse rotation and must not be mapped into the
   new variables. Continue only the selected one-domain ingoing-Kerr-Schild
   Valencia path. Restart N16 from WP10c5m and extend geometrically to about
   `1e-9 t_load` under the unchanged `1e-10` nonlinear and aggregate
   conservation gates. Stop at the first failed physical or numerical gate.
   Only after N16 passes may N32 be evolved to exactly the same physical time.
   Do not launch N64/N96, distributed tide, wind, stability, or a hot/cycle
   search in this duration package.
9. Continue one physical distributed tide only after the global no-tide
   duration gate is computationally practical and passes; search for
   accumulation, fronts, hot phases, and limit cycles.
10. Add wind only after the tidal and stability gates pass.

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
- Causal relativistic alpha shear: `reports/current/CODEX_CAUSAL_RELATIVISTIC_ALPHA_SHEAR_WP10C3A_RESULTS_2026-07-17.md`
- Responsive-height thermal ledger: `reports/current/CODEX_RESPONSIVE_HEIGHT_THERMAL_LEDGER_WP10C3B_RESULTS_2026-07-17.md`
- Kerr-Schild stream/Roche migration: `reports/current/CODEX_KERR_SCHILD_STREAM_ROCHE_MIGRATION_WP10C4_RESULTS_2026-07-17.md`
- Five-field causal DAE preflight: `reports/current/CODEX_CAUSAL_FIVE_FIELD_DAE_PREFLIGHT_WP10C5_RESULTS_2026-07-17.md`
- Five-field causal DAE assembly: `reports/current/CODEX_CAUSAL_FIVE_FIELD_DAE_ASSEMBLY_WP10C5B_RESULTS_2026-07-17.md`
- Reduced primitive null audit: `reports/current/CODEX_CAUSAL_FIVE_FIELD_REDUCED_NULL_AUDIT_WP10C5C_RESULTS_2026-07-17.md`
- Consistent initial step: `reports/current/CODEX_CAUSAL_FIVE_FIELD_CONSISTENT_INITIAL_STEP_WP10C5D_RESULTS_2026-07-17.md`
- Temporal-storage increment audit: `reports/current/CODEX_CAUSAL_FIVE_FIELD_TEMPORAL_STORAGE_INCREMENT_WP10C5E_RESULTS_2026-07-17.md`
- Reduced linear-precision audit: `reports/current/CODEX_CAUSAL_FIVE_FIELD_LINEAR_PRECISION_WP10C5F_RESULTS_2026-07-17.md`
- Residual directional-consistency audit: `reports/current/CODEX_CAUSAL_FIVE_FIELD_DIRECTIONAL_CONSISTENCY_WP10C5G_RESULTS_2026-07-17.md`
- Increment-primary startup audit: `reports/current/CODEX_CAUSAL_FIVE_FIELD_INCREMENT_PRIMARY_WP10C5H_RESULTS_2026-07-17.md`
- Exact-stream sparse repeated startup: `reports/current/CODEX_CAUSAL_SOURCE_ON_SPARSE_REPEATED_STARTUP_WP10C5I_K_RESULTS_2026-07-18.md`
- Matched source and source-compatible startup: `reports/current/CODEX_CAUSAL_MATCHED_SOURCE_AND_COMPATIBLE_STARTUP_WP10C5L_M_RESULTS_2026-07-18.md`
- Fully coupled rank prototype: `reports/current/CODEX_COUPLED_INNER_OUTER_RANK_PROTOTYPE_RESULTS_2026-07-11.md`
- Coupled mesh/interface certification: `reports/current/CODEX_COUPLED_MESH_INTERFACE_CERTIFICATION_RESULTS_2026-07-11.md`
- Coupled wall pattern-power gate: `reports/current/CODEX_COUPLED_WALL_PATTERN_POWER_RESULTS_2026-07-11.md`
- Coupled open-overflow eigenvalue: `reports/current/CODEX_COUPLED_OPEN_OVERFLOW_RESULTS_2026-07-11.md`
- Flux-primary time DAE selection: `reports/current/CODEX_TIME_DAE_BOUNDARY_AND_FLUX_PRIMARY_RESULTS_2026-07-12.md`
- Energy semantics and Roche contract: `reports/current/CODEX_GLOBAL_ENERGY_SEMANTICS_AND_ROCHE_BOUNDARY_CONTRACT_2026-07-13.md`
- Standalone Roche nozzle: `reports/current/CODEX_HILL_ROCHE_NOZZLE_PROTOTYPE_RESULTS_2026-07-13.md`
- Gas-radiation production Roche edge: `reports/current/CODEX_GAS_RADIATION_ROCHE_BOUNDARY_RESULTS_2026-07-13.md`
- Physical-edge loading preflight: `reports/current/CODEX_GLOBAL_ROCHE_LOADING_PREFLIGHT_RESULTS_2026-07-13.md`
- Adaptive/restart preflight: `reports/current/CODEX_GLOBAL_ROCHE_ADAPTIVE_RESTART_RESULTS_2026-07-13.md`
- Shared-time mesh and N64 extension: `reports/current/CODEX_GLOBAL_ROCHE_SHARED_TIME_EXTENSION_RESULTS_2026-07-13.md`
- Shared `1e-6 t_load` mesh gate: `reports/current/CODEX_GLOBAL_ROCHE_SHARED_MILLIONTH_RESULTS_2026-07-13.md`
- N64 long-extension boundary stop: `reports/current/CODEX_GLOBAL_ROCHE_N64_LONG_EXTENSION_STOP_RESULTS_2026-07-13.md`
- Causally outgoing inner plunge: `reports/current/CODEX_GLOBAL_SUPERSONIC_PLUNGE_RESULTS_2026-07-13.md`
- Global evolution WP1 diagnostics: `reports/current/CODEX_GLOBAL_EVOLUTION_DIAGNOSTICS_WP1_RESULTS_2026-07-14.md`
- Global solver-efficiency WP2: `reports/current/CODEX_GLOBAL_SOLVER_EFFICIENCY_WP2_RESULTS_2026-07-14.md`
- Exact-common-time global snapshot WP3: `reports/current/CODEX_GLOBAL_EXACT_COMMON_TIME_WP3_RESULTS_2026-07-14.md`
- Sonic-gradient and plunge-mapping WP4: `reports/current/CODEX_TRANSONIC_SONIC_GRADIENT_WP4_RESULTS_2026-07-14.md`
- Source-on/source-off WP5: `reports/current/CODEX_GLOBAL_SOURCE_ON_OFF_WP5_RESULTS_2026-07-14.md`
- Source-free N64 relaxation WP6a: `reports/current/CODEX_GLOBAL_SOURCE_FREE_RELAXATION_WP6A_RESULTS_2026-07-14.md`
- Source-free N96 refinement and N128 remap WP6a-R: `reports/current/CODEX_GLOBAL_SOURCE_FREE_REFINEMENT_WP6AR_RESULTS_2026-07-15.md`
- Local inner-plunge projection WP6b: `reports/current/CODEX_GLOBAL_INNER_PLUNGE_PROJECTION_WP6B_RESULTS_2026-07-15.md`
- Low-throughput remnant WP6c: `reports/current/CODEX_GLOBAL_LOW_THROUGHPUT_REMNANT_WP6C_RESULTS_2026-07-15.md`
- Characteristic-response efficiency WP7: `reports/current/CODEX_GLOBAL_CHARACTERISTIC_RESPONSE_WP7_RESULTS_2026-07-17.md`
- Fresh low-mass global startup WP8: `reports/current/CODEX_GLOBAL_FRESH_LOW_MASS_STARTUP_WP8_RESULTS_2026-07-17.md`
- Fresh-loading inner-boundary architecture WP9: `reports/current/CODEX_GLOBAL_INNER_BOUNDARY_ARCHITECTURE_WP9_RESULTS_2026-07-17.md`
- Causal inner thermodynamics WP10a: `reports/current/CODEX_CAUSAL_INNER_THERMODYNAMICS_WP10A_RESULTS_2026-07-17.md`
- Horizon-penetrating Valencia core WP10b: `reports/current/CODEX_HORIZON_PENETRATING_VALENCIA_WP10B_RESULTS_2026-07-17.md`
- Valencia gas+radiation primitive recovery WP10c1: `reports/current/CODEX_VALENCIA_GAS_RADIATION_PRIMITIVE_RECOVERY_WP10C1_RESULTS_2026-07-17.md`
- Kerr-Schild geometric finite volume WP10c2: `reports/current/CODEX_KERR_SCHILD_GEOMETRIC_FINITE_VOLUME_WP10C2_RESULTS_2026-07-17.md`
