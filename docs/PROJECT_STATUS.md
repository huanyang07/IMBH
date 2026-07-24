# Project Status

- Updated: 2026-07-24
- Pre-cleanup scientific tag: `pre-cleanup-p0-2026-07-11`
- Legacy phase classification tag: `legacy-steady-positive-flux-dae-2026-07-10`

This is the canonical project handoff. Status labels mean:

- **CERTIFIED:** passes the stated numerical and physical gates for its scope.
- **SUPPORTED BUT NOT FULLY CERTIFIED:** strong numerical evidence with an
  identified unresolved robustness or closure condition.
- **DIAGNOSTIC ONLY:** useful mathematical or numerical evidence that must not
  be promoted to a physical branch claim.
- **IMPLEMENTED AND METHOD-TESTED:** the software and local identities pass
  their tests, but the scientific campaign contract is not certified.
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
| Bounded causal duration WP10c5n | **CERTIFIED** as a bounded negative mesh result; **DIAGNOSTIC ONLY** physically | N16/N32 separately pass at exact `6.78172e-4 s` with full rank, no retries, closed Roche flow, `tau>18.67`, and mass/five-field defects `<4.5e-12` | Thickness-response mesh mismatch is `1.2557e-2 > 5e-3`; independently tuned moving-cell initial profiles confound continuum interpretation; no further duration or physics is authorized |
| Mesh-common startup and temporal parity WP10c5o-q | **CERTIFIED** for common-data short startup and temporal-confound exclusion; **CERTIFIED** as a bounded negative spatial result; **DIAGNOSTIC ONLY** physically | One fixed physical profile passes N16/N32 initialization; short response mismatch is `2.7898e-3`; both bounded trajectories pass individually; equal 63-step duration control preserves full rank and ledgers below `4.7e-12` | Duration `Delta ln(H/R)` mismatch remains `2.1033e-2 > 5e-3`; the failure is spatial at N16/N32, and N64 plus all physical searches remain blocked |
| Causal spatial-response audit WP10c5r | **CERTIFIED** for spatial-error classification; **DIAGNOSTIC ONLY** physically | Exact term tangents reconstruct below `2.7e-11`; face transport controls the N16/N32 mismatch at `24.05 s^-1`; central flux converges at order `>=1.996`, Rusanov/full transport at `>=1.106`, all source terms at expected order, and exact stream moments restrict below `2.26e-16` | The discrepancy is ordinary first-order coarse-grid truncation; no operator change is justified; its N64 authorization is consumed by WP10c5s-t |
| Causal N64 confirmation WP10c5s-t | **CERTIFIED** for bounded contraction and ledger closure; **DIAGNOSTIC ONLY** physically | Independent N64 short error `8.65e-4`; strict 63-step duration has full `320/320` and `965/965` rank, no retries, five-field defect `1.81e-12`; N32/N64 error contracts `2.1033e-2 -> 6.6677e-3` at order `1.657` | The `5e-3` duration gate is not yet met; exactly one N128 confirmation is authorized, while N96/N256, longer evolution, tide, wind, stability, hot-state, and cycle work remain blocked |
| Causal N128 mesh certification WP10c5u | **CERTIFIED** for bounded first-order spatial convergence; **DIAGNOSTIC ONLY** physically | Independent N128 datum has full `640/640` descriptor and `1925/1925` consistency rank; exact-time duration has 63 steps, no retries, ledger defect `1.77e-12`; N64/N128 error is `2.5897e-3 < 5e-3` at order `1.364` | Only `9.98e-10 t_load` is evolved and the run costs roughly two wall-clock hours; no further fine mesh or direct microstep duration is justified, and physics searches remain blocked pending a timestep-ceiling audit |
| Causal N16 timestep ceiling WP10c6a | **CERTIFIED** for local temporal control; **DIAGNOSTIC ONLY** physically | One-full/two-half backward-Euler ladder passes through `1.92182e-3 s` and first fails temporal accuracy at `3.84364e-3 s`; all solver, ledger, causal, optical, and Roche gates pass; the inherited step is enlarged by `256x` | One N16 checkpoint only; the ceiling is `4.0-8.0%` of the shortest physical clock and requires an N32 mesh check before becoming a production controller |
| Causal N16/N32 temporal-controller contract WP10c6b | **CERTIFIED** for local temporal control; **DIAGNOSTIC ONLY** physically | N32 exactly reproduces the N16 passing/failing bracket and the same cooling/`H/R` failure observables while its shortest cell-crossing clock falls to `2.16380e-2 s`; all solver, ledger, causal, optical, and Roche gates pass | The step-doubling controller contract is authorized but not yet implemented or tested over a matched duration; no N64/N128, long evolution, tide, wind, stability, hot-state, or cycle run is authorized |
| Causal accumulated-error controller WP10c6c | **CERTIFIED** as a bounded negative production result; **DIAGNOSTIC ONLY** physically | N16 local controller takes 9 accepted/0 rejected steps, restart replay is bitwise, all contracts pass, and Jacobian work is `2.19x` below a 64-step reference | Final cooling and `H/R` errors reach `2.26-3.69x` their gates; N32 is correctly skipped and the local contract is not a production accumulated-error law |
| Causal horizon-budget reference WP10c6d | **CERTIFIED** for first-order temporal convergence; **DIAGNOSTIC ONLY** physically | All 224 N16 fixed steps at 32/64/128 subdivisions pass and all six observables converge at order `0.9947-1.0030` | Raw 64-to-128 uncertainty is `0.369-0.605` of three cooling/`H/R` gates, above the locked `0.25`; controller and N32 runs are correctly blocked |
| Causal refined reference WP10c6e | **CERTIFIED** for the bounded N16 temporal reference; **DIAGNOSTIC ONLY** physically | All 896 fixed steps at 128/256/512 subdivisions pass; saved checkpoints reload bitwise; all six observables converge at order `0.9987-1.0008` | Raw 256-to-512 uncertainty is at most `0.1517` of a gate; only one separate N16 horizon-budget closure is authorized |
| Causal horizon-budget closure WP10c6f | **CERTIFIED** for bounded N16 temporal accuracy and restart; **DIAGNOSTIC ONLY** physically | Exact horizon in 46 accepted steps; split replay is bitwise; controller error plus S256/S512 uncertainty is at most `0.85294` of a gate; Jacobian work is `0.27539` of S512 | The `dt/T_output` first-order budget scales poorly with horizon; backward Euler is frozen as reference/fallback and only WP10c7a method work is authorized |
| Increment-primary BDF method WP10c7a | **CERTIFIED** for method-level BDF1/BDF2 and complete history; **DIAGNOSTIC ONLY** physically | Scalar/index-one/vertical tests converge at order `2.006-2.074`; BDF1 parity and five-field history defects are zero; N4 Jacobian is `65/65`; restart is bitwise | No N16 BDF2 disk trajectory or adaptive controller yet; only WP10c7b fixed-step N16 certification is authorized |
| Fixed-step N16 BDF2 WP10c7b | **CERTIFIED** for bounded second-order temporal evolution and restart; **DIAGNOSTIC ONLY** physically | All six observables converge at order `1.994-2.005`; S64 plus S256/S512 uncertainty is at most `0.27474` of a gate; all physical-ledger components converge at order two; replay is bitwise | Only WP10c7c adaptive N16 BDF2 is authorized; N32 and physical-duration evolution remain gated |
| Adaptive N16 BDF2 WP10c7c | **CERTIFIED** for bounded adaptive temporal evolution and restart; **DIAGNOSTIC ONLY** physically | Exact horizon in 20 accepted steps with zero retries and five independent audits; endpoint plus reference uncertainty is at most `0.28886` of a gate; physical-ledger defect is `7.11e-5`; replay is bitwise; Jacobian work is `0.4125` of fixed S64 | Only matched N32 WP10c7d is authorized; no N64/N128, physical-duration evolution, tide, wind, stability, hot-state, or cycle run |
| Matched N32 BDF2 WP10c7d | **CERTIFIED** for bounded N32 temporal evolution; **REJECTED** for N16/N32 spatial response; **DIAGNOSTIC ONLY** physically | N32 fixed orders `1.995-2.007`; adaptive plus S32/S64 uncertainty is at most `0.11411` of a gate; replay is bitwise; Jacobian work is `0.2844` of fixed S64 | N16/N32 `Delta log(H/R)` response differs by `0.61293 > 0.005` near `20.86 rg`; no N64/N128 or longer/physical run is authorized before a localized spatial audit |
| Localized spatial response WP10c7e | **CERTIFIED** for spatial-error classification; **DIAGNOSTIC ONLY** physically | Exact restriction gives fixed/adaptive N16/N32 `Delta log(H/R)` differences `0.613215/0.613234`; fixed/adaptive history differs by at most `7.62e-5`; the first S64 step already fails; the initial DAE tangent is controlled by total face transport at `24.14 s^-1`, with Rusanov exceeding central transport | Exactly one N64 fixed-BDF2 contraction diagnostic is authorized as WP10c7f; no operator change, N128, longer duration, tide, wind, stability, hot-state, or cycle run |
| N64 BDF2 contraction WP10c7f | **CERTIFIED** for N64 temporal accuracy and contraction measurement; **REJECTED** for the spatial gate; **DIAGNOSTIC ONLY** physically | N64 raw S32/S64 `Delta log(H/R)` uncertainty is `1.536e-4 < 2.5e-4`; all state/ledger gates pass; exact N32/N64 response contracts `0.613215 -> 0.134682` at order `2.187` | The error remains `26.9x` the gate; measured-order N64/N128 projection is `0.0296 > 0.005`, so N128 and uniform refinement are closed; only WP10c7g operator-level reconstruction work is authorized |
| Causal PLM reconstruction WP10c7g | **CERTIFIED** for method-level interior reconstruction; **DIAGNOSTIC ONLY** physically | Smooth-PLM finest-pair manufactured order is at least `1.910`; diagnosed-band total/full tangent orders are `2.116/2.172`; the full tangent discrepancy falls by `5.235x`; N8 colored/dense Jacobians agree to `1.27e-10`; N16/N32 consistency rank is full | Full-domain order remains boundary limited because the physical boundary traces are unchanged and first order; certification authorizes only the bounded WP10c7h trajectory |
| Reconstructed-flux trajectory WP10c7h | **CERTIFIED** as a bounded negative result; **REJECTED** for spatial adequacy; **DIAGNOSTIC ONLY** physically | All 192 N32/N64 S32/S64 fixed steps pass; temporal thickness uncertainty is below `1.48e-4`; source restriction is `1.73e-16`; physical ledgers are below `1.58e-4`; restart is bitwise | N32/N64 `Delta log(H/R)` still differs by `0.04462` over the full domain and `0.02141` over `15-60 rg`, versus `0.005`; N128 and longer evolution remain closed; only WP10c7i method-level balance work is authorized |
| Causal spatial balance WP10c7i | **CERTIFIED** for method-level full-domain consistency; **DIAGNOSTIC ONLY** physically | Quadratic admissible faces plus reconstructed local-rate source quadrature reduce the N32/N64 full tangent by `23.87x` and the `15-60 rg` tangent by `13.50x`; both have order `2.361`; projected error is `1.528e-3 < 2.5e-3`; N16/N32 rank is full | No trajectory was run; exactly one fresh N32/N64 fixed-BDF2 bounded confirmation is authorized before N128, longer evolution, or new physics |
| Spatial-balance trajectory WP10c7j | **CERTIFIED** for bounded N32/N64 spatial evolution; **DIAGNOSTIC ONLY** physically | All 192 fixed steps pass; endpoint `Delta log(H/R)` difference is `1.52769e-3`; adding both temporal uncertainties gives `1.81679e-3 < 0.005`; the measured response is `0.999803` of the WP10c7i projection; ledgers remain below `2.02e-4`; restart and snapshots are bitwise | Exactly one matched adaptive-BDF2 confirmation is authorized; no longer physical horizon, N128, tide, wind, stability, hot-state, or cycle work yet |
| Matched adaptive spatial balance WP10c7k | **CERTIFIED** for bounded N32/N64 adaptive evolution; **DIAGNOSTIC ONLY** physically | Both meshes take 13 accepted/0 rejected steps with four audits; endpoint `Delta log(H/R)` is `1.52763e-3`; adding adaptive and fixed-reference uncertainty gives at most `1.85230e-3 < 0.005`; ledgers are below `7.60e-5`; replay is bitwise; Jacobian work is `0.3281` of fixed S64 | Exactly one matched no-tide extension toward the `~0.05 s` characteristic-crossing rung is authorized; N128, later physical clocks, tide, wind, stability, hot-state, and cycle work remain closed |
| Characteristic-crossing extension WP10c7l | **CERTIFIED** for robust N32/N64 temporal evolution and the conservative `0.0375 s` spatial horizon; **REJECTED** at the `0.05 s` spatial gate; **DIAGNOSTIC ONLY** physically | All four production/control campaigns reach exact `0.05 s` with no retries; temporal audits, state gates, ledgers, source restriction, work, and replay pass; the conservative spatial total is `0.002845/0.004101` at `0.025/0.0375 s` | At `0.05 s`, raw `Delta log(H/R)=0.004944` becomes `0.005348 > 0.005` after uncertainty; growth is linear at `0.099015 s^-1`; stress/cooling/thermal rungs and new physics remain closed pending an evolved-state spatial-order audit |
| Evolved-state spatial order WP10c7m | **CERTIFIED** for one N128 reference campaign; **DIAGNOSTIC ONLY** physically | Two independent common-state N32/N64/N128 oracles give full-domain thickness-tangent order `1.989/1.996`, interior temperature order `2.127-2.131`, and scaled-energy order `1.875-1.876`; projected spatial plus temporal/oracle reserve is `0.001751 < 0.0025` | Full-domain raw temperature/energy maxima remain boundary limited; no duration extension or reduction calibration is authorized before measured N64/N128 `0.05 s` certification |
| Fresh N128 reference WP10c7n | **CERTIFIED** for N64/N128 evolution through `0.05 s`; **DIAGNOSTIC ONLY** physically | Fresh N128 production/control take 30/60 accepted steps with zero retries; raw N64/N128 `Delta log(H/R)=0.0012235`, conservative total `0.0014873`, observed order `2.0147`, Richardson remainder `0.0004023`; replay is bitwise | Only one third of a stress-relaxation time is covered; selected-state descriptor spectra and full/reduced validation are required before any slow-manifold or long-duration claim |
| Selected-state slow modes WP10c8a | **CERTIFIED** as a negative global-reduction result; **DIAGNOSTIC ONLY** physically | All finite N64/N128 modes and every isolated `P_R/chi` block are stable at `0/0.0375/0.05 s`; descriptor rank is full, maximum eigenpair defect is `2.14e-8`, and low-mode median mesh mismatch is `0.172-0.186` | The candidate fast block spans `0.013-1438 s` while retained high-wavenumber modes reach `0.014-0.029 s`, giving gaps near `1e-5`; extreme non-normality rejects global algebraic elimination and authorizes only trajectory-conditioned, region-aware feasibility work |
| Causal stress-time audit WP10c8b | **CERTIFIED** through `0.125 s`; **REJECTED** at the stronger `0.15 s` spatial gate; **DIAGNOSTIC ONLY** physically | Six matched N32/N64/N128 production/control campaigns pass with zero retries; conservative N64/N128 response at `0.15 s` is `0.0038168 < 0.005`, order is `1.9618`, and both final replays are bitwise | N128 Richardson remainder is `0.0012533 > 0.00125`; stress-target and radial-balance departures do not decay, so only an operator-level WP10c8c closure audit is authorized at the certified `0.125 s` state |
| Region-selective closure WP10c8c | **CERTIFIED** as a negative reduction result; **REJECTED** for nonlinear closure; **DIAGNOSTIC ONLY** physically | All 54 N64/N128 regional Schur audits have full descriptor rank and dynamic-solve defects below `2.55e-16`; every isolated fast block is stable; three `60-200 rg` charts preserve the tested short-time responses | No chart passes physical slaving or the fast/retained gap on either mesh; several effective operators become unstable, and the global joint closure has transient gain `4.72/8.43`; fieldwise algebraic reduction remains closed |
| Conservation-constrained mixed modes WP10c8d | **CERTIFIED** as a negative BPOD reduction result; **REJECTED** for a nonlinear ROM; **DIAGNOSTIC ONLY** physically | All six N64/N128 descriptors are full rank with explicit solve defects below `2.66e-16`; exact M/J/E coordinates are protected; the finite-horizon Hankel maps resolve only total orders `39-41`; low orders have some cross-mesh alignment | Every available order `8/16/32` reduced operator is unstable; held-out responses fail; the order-32 unresolved complement grows by `18.6x` at N128 by `0.1 s`; no nonlinear operator compression or loading-time speedup is authorized |
| Stationary-branch preflight WP10c8e | **CERTIFIED** as a bounded negative preflight; **REJECTED** for root continuation from the tested seeds; **DIAGNOSTIC ONLY** physically | Valid `0.1-1.0` source seeds have exact mass throughput and full `80/80` reduced stationary rank | Zero/weak seeds violate surface-density or optically thick cooling validity; valid seeds retain `0.896-0.898` angular-ledger defects, condition estimates `2.34e10-5.25e10`, scaled Newton corrections `69-202`, and no physical damped trial |
| Stable observable reduction WP10c8f | **CERTIFIED** as a negative rational-reduction result; **REJECTED** for a nonlinear ROM; **DIAGNOSTIC ONLY** physically | Exact N64/N128 global ledgers identify angular/energy clocks `9.36e5/1.44e6 s`; ledger-safe LQR stabilization keeps protected defects below `7.4e-17` and stabilizes every order `8-96` model | No dense Lyapunov metric is numerically positive definite; best trained response error is `1.0002 > 0.1`, cross-mesh transfer excess is `0.320-0.323 > 0.25`, and no compact nonlinear or memory model is authorized |
| Ledger equation-free preflight WP10c8g | **CERTIFIED** as a negative global-closure result; **REJECTED** for nonlinear lifting or macrosteps; **DIAGNOSTIC ONLY** physically | Global `M/J/E` factor-two projections are below `6.7e-7` of a gate and exact-rate/secant defects stay below `9.6e-3`; production/control and N64/N128 paths agree | Equal-ledger thermal directions change held observables by `19.73-19.75` gates and radial directions change projected responses by `13.05-13.23` gates; the eight-variable augmented state also fails |
| Conservative shell preflight WP10c8h | **CERTIFIED** as a negative shell-closure result; **REJECTED** for nonlinear shell microbursts or macrosteps; **DIAGNOSTIC ONLY** physically | Exact mesh-coincident five/eight-shell operators retain `15/24` finite-volume `M/J/E` coordinates with constraint-null defects below `1e-10` | Five/eight-shell AB2 errors are `0.569/1.752 > 0.25`; ledger-null thermal and radial redistributions change observables by `17.5-17.8` and `9.94-10.01` gates; compact global and shell-only equation-free routes are closed |
| Storage-consistent moment sufficiency WP10c8i | **IMPLEMENTED AND METHOD-TESTED** for the complete vector-storage and incremental moment-audit machinery; **INCONCLUSIVE** for moment sufficiency; nonlinear lifting and macrosteps **NOT AUTHORIZED** | Complete responsive-height storage is pulled back as a vector one-form in radial momentum, angular momentum, and Killing energy; its maximum action defect is `2.98e-7`; five cumulative five-shell levels retain `15/20/25/30/34` full-rank coordinates | Full generator FD-consistency scans fail at `t=0/0.10 s` on both meshes and consequential Rusanov branches remain at three anchors; conditional richest-level lower gains are `>340` gates, but are nonbinding; online cost is unevaluated and only a bounded tangent-certification package is authorized within this reduction branch |
| Evolving-tangent certification WP10c8j | **IMPLEMENTED AND METHOD-TESTED**; **REJECTED** for unchanged WP10c8i repetition or reduced evolution | Direct differentiation of the complete storage-rate action removes the nested mass-matrix derivative; assembled-generator step stability is below `1.64e-3`, factorization below `3.64e-12`, and selected storage reconstruction below `2.61e-11` in the matched N64/N128 `0.10 s` scans | N64 `0.05 s` outer thermal/density JVP defects remain `1.87e-2/1.06e-2` at the selected step, N128 `0.10 s` density reaches `1.0209e-2`, and all-face Rusanov coverage/nonlinear remainder data are absent; moment changes, lifting, healing, and macrosteps remain closed |
| Tangent localization WP10c8k | **CERTIFIED** as a bounded negative localization result; **REJECTED** for unchanged WP10c8i repetition or reduced evolution | The exact centered descriptor product closes near `1e-13`, the stationary derivative agrees near `5e-9`, and more than `99.98%` of the smooth primitive mismatch is mapped-storage-rate differentiation | The best tangent-only action retains strict infinity defects `0.01028-0.01186`; the aggregate Rusanov enclosure consumes `2.464/28.58` gates at N64 `t=0`, so both finite-difference tuning and the aggregate certificate architecture are closed |
| Unified descriptor and structured Rusanov WP10c8l | **IMPLEMENTED AND METHOD-TESTED**; Track A **REJECTED** at locked N64; Track B **FEASIBLE BUT NONBINDING** for cached branches | One audit-only discrete `S_map/DS_map/D2S_map` path gives factorization `5.46e-12` and stable secants; a face-aware nominal-semigroup preflight reduces cached-branch gate fractions to at most `3.63e-4`, with 64/128-panel change below `0.63%` | Track-A centered infinity defects remain `0.0184-0.0207 > 0.01`, so N128, all-face Track B, finite-neighborhood certification, WP10c8i repetition, moment changes, lifting, healing, and reduced evolution remain closed |
| Branch-frozen tangent and structured Rusanov WP10c8m | Track A **CERTIFIED** at locked N64/N128; cached Track B **FEASIBLE BUT NONBINDING**; all-candidate Track B **REJECTED** | Branch-frozen assembled mapped-storage derivatives give worst centered infinity defects `1.29e-5/4.37e-5` and step defects below `1.86e-9`; regenerated cached-branch fractions remain below `3.64e-4`; all 567 face/candidate factors close below `6.03e-16` | The pessimistic all-noncontroller superset consumes `0.06695 > 0.01`, controlled by interface-3 rest-mass flux; finite-neighborhood remainder work, WP10c8i repetition, moment changes, lifting, healing, and reduced evolution remain closed pending a sharper possible-winner certificate |
| Rusanov candidate-screen WP10c8n | **CERTIFIED** as a stop decision for the uniform exact-max generalized-tangent certificate; **DIAGNOSTIC ONLY** for nonlinear closure | The all-candidate result is reproduced within `4.2e-17`; direct branch-output change supplies `99.75%` of the `0.06695` fraction; the null-tube closure retains 449 alternatives and a nonlinear admissible face-58 switch occurs between weighted radii `0.0058177/0.0058294` | A containing common radius must exceed `2.05`, where the bound remains `0.06695 > 0.005/0.01`; this rejects candidate-gap rescue and the uniform tangent certificate, not the production flux or nonlinear fiber closure; WP10c8i and reduced evolution remain closed pending paired finite-amplitude lifting/healing |
| Exact nonlinear coordinate fiber WP10c8o | **CERTIFIED** as an N64/N128 truth-discretization counterexample; 34-coordinate instantaneous Markov closure **REJECTED on those certified meshes** | Exact equal-coordinate pairs close below `1.78e-15`; all state/fresh-rate/DAE-storage gates pass; descriptor ranks are `320/320` and `640/640`, maximum full-Schur parity defect is `8.72e-11`, and the same interface-4 angular-momentum flux controls both meshes at `0.32453/0.26609 > 0.25` with cross-mesh disagreement `0.05844 < 0.10` | The result is one-sided and is not a continuum no-go: it rejects raw deterministic algebraic closure on the certified truth discretizations but not healed closure, memory, or a coarse effective PDE; only matched BDF1-start natural-healing microbursts of the frozen decisive pair are authorized next |
| Natural coordinate-fiber healing WP10c8p | **CERTIFIED** as a matched N64/N128 rapid-healing rejection through `0.025 s`; **DIAGNOSTIC ONLY** for longer memory | Synchronized coarse/fine BDF1-start/BDF2 pairs, fresh rates, physical ledgers, exact flux splits, and bitwise replays all pass; interface-4 angular-momentum spread changes only `0.32452995 -> 0.32452655` at N64 and `0.26608550 -> 0.26608444` at N128, with temporal uncertainty below `2.81e-7` and coordinate drift below `9.09e-8` | The unresolved `M/J/E_K` transport is more than `99.9%` central perfect-fluid response and does not rapidly heal; this does not establish permanent memory or authorize an auxiliary/PDE, so only an N64 geometric extension to `0.05/0.10/0.125 s` is authorized before selecting a closure architecture |
| Extended healing and slow-rate fiber WP10c8q | **CERTIFIED** for persistent healing rejection and complete slow-rate nonclosure; the rank-two interface-4 interpretation is **SUPERSEDED by WP10c8r** | The shell-incidence audit proves real conservative redistribution for the original pair; exact-history `h/h/2` continuations pass through `0.125 s`, while interface-4 angular-momentum spread changes only `0.32452995 -> 0.32451281` (`5.28e-5` e-folds) | The later independent slow-rate cases have negligible absolute interface-4 responses; their unit-normalized SVD cannot authorize two face coordinates |
| Interface-state sufficiency WP10c8r | **CERTIFIED** as a significance-corrected stop decision; two-component interface-4 state **NOT AUTHORIZED** | All six independent slow-rate cases have interface-4 half-spreads only `2.65e-11-1.18e-8` gate units and all-interface maxima below `9.65e-5`; only the original healing family is significant and it remains rank one. The complete slow-rate tangent has `4-5` directions above `0.1` of the leading singular value with strong N64/N128 agreement | Large rate ambiguities occupy stress, thermal, momentum, and sub-shell structure rather than significant macro-interface-4 transport. Do not add two face coordinates; next audit nonlinear healing and localization of the complete-rate modes |
| Complete-rate healing WP10c8s | **CERTIFIED** as a nonlinear fail-fast architecture rejection; `q_34` plus only one interface-4 state **INSUFFICIENT** | Six exact equal-`q_34` nonlinear pairs have slow-rate half-spreads `26.95-431.04` and matched N64/N128 tangent support. An independent inner-shell mode remains above the healing gate at `0.025 s`, with uncertainty-inclusive lower bound `16.563 > 0.10`, while coordinate drift stays below `2.23e-7` | The strict decay-curve temporal gate does not pass, so no relaxation time is certified. The other five healing cases were deliberately stopped after the binding fail-fast result. Extend and confirm the inner mode before choosing localized extra states or a staggered coarse finite-volume/PDE architecture |

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
56. WP10c5n reaches the exact shared time `6.781724319e-4 s` at N16 and N32.
    Each resolution retains zero inner incoming modes, two closed-edge outer
    responses, scattering depth above `18.67`, full descriptor and
    consistency rank, no rejected steps, and aggregate mass/five-field
    defects below `4.5e-12`. The global fluxes agree, but the
    baseline-subtracted N16/N32 `Delta ln(H/R)` response differs by
    `1.2557e-2`, above the fixed `5e-3` gate, broadly around `12-16 rg`.
    Further evolution stops. The current initializer also tunes temperature
    at each moving first cell center, so one fixed physical continuum datum
    must be tested before the mismatch is called spatial nonconvergence.
57. WP10c5o-q removes that initial-data confound with one C2 profile anchored
    at fixed `6 rg` and `240 rg` plateaus and a shared physical-face
    thermodynamic state. N16/N32 initialization defects stay below `6.69e-3`,
    exact unit throughput and all causal/optical/Roche/rank gates pass, and
    the short source-on response mismatch is `2.7898e-3 < 5e-3`. At the
    bounded common time, however, `Delta ln(H/R)` differs by `2.1033e-2`.
    Giving both meshes the same maximum timestep and exactly 63 extension
    steps changes their responses by at most `3.00e-6`, so temporal alignment
    is excluded. The remaining failure is spatial at N16/N32; N64 and physical
    evolution remain blocked pending one term-resolved semidiscrete audit.
58. WP10c5r decomposes the exact constrained semidiscrete tangent into face
    transport and six production source components. Exact N32-to-N16
    finite-volume restriction localizes `24.0482 s^-1` of the thickness-rate
    mismatch to face transport at `55.5662 rg`; the next source contribution
    is only `1.58902 s^-1`, and linear/PCHIP reconstructions recover the same
    result. Operator-only N16-N128 tests give central-flux order at least
    `1.9961`, Rusanov/full-transport order at least `1.1058`, source-term
    order at least `1.9837`, and exact-stream restriction error below
    `2.26e-16`. The N16/N32 duration failure is quantified first-order
    truncation, not an inconsistent stencil. No correction is authorized;
    one bounded N64 confirmation is the next gate.
59. WP10c5s-t independently regenerates the fixed datum at N64. Its exact-time
    short response differs from N32 by `8.6493e-4 < 5e-3`. The baseline
    duration reaches the target with all state/rank/step gates but accumulates
    a `1.8082e-10` five-field defect because accepted residuals plateau near
    `4.85e-11`. One replay tightens only the solve residual to `1e-11`, lowers
    the ledger defect to `1.8084e-12`, and changes the response by only
    `5.01e-10`. N32/N64 duration error is `6.6677e-3`, contracting from
    N16/N32 at order `1.657` (RMS `1.463`). The mesh gate is not yet
    certified; exactly one bounded N128 confirmation is authorized.
60. WP10c5u independently regenerates N128 and passes the unchanged
    N64/N128 short gate with a `3.33334e-4` thickness-response difference.
    Its strict duration reaches exactly `8.48423267e-4 s` in 63 extension
    steps with no retries, full `640/640` descriptor and `1925/1925`
    consistency rank, five-field defect `1.77056e-12`, and mass defect
    `2.48535e-13`. The N64/N128 response error is `2.58967e-3 < 5e-3`,
    contracting at order `1.36443` (RMS `1.22561`). This certifies the
    bounded first-order spatial gate and closes further fine-mesh work. It
    does not authorize direct long evolution or physical claims.
61. WP10c6a defines versioned cooling, inner-flux, thickness, and conserved
    observables and measures characteristic, stress-relaxation, advection,
    cooling, thermal, and loading clocks on the accepted WP10c5q N16 datum.
    The shortest clock is the inner characteristic crossing time,
    `4.79165e-2 s`. One-full/two-half backward-Euler comparisons pass through
    `1.92182e-3 s`, `256x` the inherited controller step, and first fail at
    `3.84364e-3 s` only because total cooling and `Delta ln(H/R)` exceed their
    temporal gates. N32 is authorized solely to test whether this local
    ceiling is mesh supported.
62. WP10c6b repeats the unchanged audit on the accepted WP10c5q N32 datum.
    The N16 and N32 last passing timestep is identically `1.92182e-3 s`, the
    first failing timestep is identically `3.84364e-3 s`, and both failures
    violate only total cooling and `Delta ln(H/R)`. The N32 shortest
    characteristic cell-crossing clock is smaller, `2.16380e-2 s`, while the
    observable ceiling is unchanged. This certifies a local step-doubling
    controller contract with a two-half-step accepted state, initial
    `9.60911e-4 s` step, and bounded `0.8/sqrt(error)` update. It does not yet
    certify the controller implementation or any long/physical trajectory.
63. WP10c6c implements that contract with a two-half-step accepted state and
    exact restart history. N16 reaches a predeclared eight-ceiling duration
    in nine accepted steps with no retries; all 27 implicit trial solves and
    all 64 tighter fixed-reference steps pass. Restart continuation is
    bitwise and Jacobian work falls by `2.19x`. However, final total cooling,
    exterior cooling, and `Delta ln(H/R)` errors are `3.44e-3`, `2.26e-3`,
    and `7.39e-3`, giving a maximum normalized error `3.695`. The local
    controller therefore fails the accumulated-accuracy gate. N32 is not
    launched and the next package must calibrate a horizon-wide error budget.
64. WP10c6d encodes the horizon rule and first certifies its reference. All
    32/64/128 fixed N16 trajectories pass every solver and physical contract,
    and all immutable observables converge almost exactly at first order.
    The 64-to-128 uncertainty nevertheless consumes `0.561` of the total
    cooling gate, `0.369` of exterior cooling, and `0.605` of the
    `Delta ln(H/R)` gate, above the predeclared `0.25` allowance. The hard
    stop correctly prevents the adaptive and N32 campaigns. A direct
    128/256/512 reference refinement is required without changing any gate.
65. WP10c6e completes that direct refinement. All 128/256/512 endpoints pass
    and are saved as checksummed, bitwise-reloadable restarts. Every immutable
    observable remains first order with `p=0.99866-1.00075`; the largest raw
    256-to-512 uncertainty is `0.15165` of its gate, below the locked `0.25`.
    This certifies the N16 reference and authorizes exactly one separate
    horizon-budget closure. N32, BDF2 disk trajectories, long evolution, tide,
    wind, stability, hot-state, and cycle work remain blocked.
66. WP10c6f closes backward Euler with the single authorized N16
    horizon-budget experiment. It reaches the exact target in 46 accepted
    steps after one initial temporal rejection; all physical contracts pass,
    and a persisted step-3 restart replay is bitwise identical. Conservatively
    adding controller-to-S512 error and raw S256-to-S512 uncertainty gives a
    largest normalized error `0.85294` in `Delta ln(H/R)`. The run uses 705
    Jacobians, `0.27539` of S512, and passes its bounded work gate. Backward
    Euler is nevertheless frozen as reference/startup/fallback because the
    horizon-wide first-order budget is not a long-duration strategy. WP10c7a
    BDF method tests are authorized; no BDF2 disk run or new physics is.
67. WP10c7a implements the generic increment-primary BDF1/BDF2 method,
    variable-step coefficients and stability guard, current/previous
    conserved and vertical-storage history, dual discrete/physical ledger
    primitives, and a checksummed complete restart. Stiff scalar, index-one
    DAE, and manufactured vertical tests converge at order `2.006-2.074`;
    physical interval defects converge near third order; BDF1 parity and both
    five-field history defects are zero; the N4 Jacobian is full `65/65`.
    This authorizes only WP10c7b fixed-step N16 BDF2 certification.
68. WP10c7b runs one BDF1 startup step followed by fixed equal-step N16
    BDF2 at 8/16/32/64 subdivisions. All six observable fine orders are
    `1.994-2.005`; the S64 endpoint plus raw S256/S512 uncertainty uses at
    most `0.27474` of a gate; the maximum discrete defect is `5.24e-12`;
    all five physical horizon-ledger components converge at order two; and
    split restart replay is bitwise. Only adaptive N16 WP10c7c is authorized.
69. WP10c7c implements variable-step adaptive N16 BDF2 with a quadratic
    three-state predictor, one ordinary implicit corrector, a method-order
    local estimator, and periodic independent full-versus-two-half audits.
    It reaches the exact bounded horizon in 20 accepted steps with zero
    retries; all five audits pass; endpoint error plus raw S256/S512
    uncertainty consumes at most `0.28886` of a gate; the physical ledger is
    below `7.12e-5`; and split replay is bitwise. Its 132 Jacobians are
    `0.4125` of fixed S64. This authorizes only matched N32 WP10c7d.
70. WP10c7d builds an independent N32 fixed BDF2 reference at 16/32/64
    subdivisions. All six observable orders are `1.995-2.007`, all five
    physical-ledger orders are near two, and the adaptive N32 endpoint plus
    raw S32/S64 uncertainty consumes at most `0.11411` of a gate. The
    adaptive trajectory uses 91 Jacobians, four independent audits, no
    retries, and bitwise replay. Temporal control is therefore certified
    through N32. The exact-common-time N16/N32 `Delta log(H/R)` response
    differs by `0.61293`, however, failing the `0.005` spatial gate by
    `122.6x` near `20.86 rg`. Spatial resolution is now the active blocker.
71. WP10c7e verifies the WP10c7d failure with exact Kerr-Schild restriction,
    native coincident faces, fixed/fixed and adaptive/adaptive endpoints, and
    exact fixed-S64 snapshots. The fixed/adaptive restricted thickness
    mismatch is `0.613215/0.613234`, while either mesh's temporal-history
    difference is at most `7.62e-5`. The mismatch exceeds `0.005` on the first
    S64 step and grows approximately linearly at first. At the initial
    checkpoint, total face transport controls the DAE-consistent mismatch at
    `24.1407 s^-1`; Rusanov contributes `13.5426 s^-1`, central transport
    `12.0895 s^-1`, and the next source only `2.60490 s^-1`. Combined with
    WP10c5r's measured first-order Rusanov/full-transport convergence, this
    confirms inherited coarse-grid transport truncation. Exactly one N64
    fixed S32/S64 contraction diagnostic is authorized as WP10c7f.
72. WP10c7f evolves the independently generated N64 datum with fixed BDF2
    at S32/S64 over the exact WP10c7d horizon. The raw temporal
    `Delta log(H/R)` uncertainty is `1.53598e-4`, passing both the `5e-4`
    maximum and `2.5e-4` preferred gates; all state, discrete-ledger, and
    physical-ledger contracts pass. Exact N64-to-N32 restriction reduces the
    response mismatch from `0.613215` to `0.134682`, observed order `2.18684`,
    but the result remains `26.9x` above `0.005`. Persistence of the measured
    order predicts N64/N128 error `0.02958`, still `5.92x` above the gate;
    direct N128 certification and uniform refinement are therefore closed.
    Only WP10c7g operator-level second-order face reconstruction is authorized.
73. WP10c7g adds optional unlimited and smooth piecewise-linear primitive
    reconstruction in `ln(R)` while freezing piecewise constant as the
    default. Reconstructed states feed the central flux, conserved jump,
    characteristic envelope, and Rusanov term consistently. The smooth
    method reaches finest-pair manufactured order `1.910`, diagnosed-band
    total/full tangent order `2.116/2.172`, and reduces the N32/N64 full
    tangent discrepancy by `5.235x`. The widened 23-color N8 Jacobian agrees
    with dense differences to `1.27e-10`, and N16/N32 consistency rank stays
    full. Full-domain order remains limited by the unchanged first-order
    physical boundary traces, so this certifies only the method-level
    interior upgrade and authorizes WP10c7h.
74. WP10c7h independently builds reconstruction-compatible N32/N64 histories
    and completes all 192 fixed S32/S64 steps. Raw temporal thickness
    uncertainty stays below `1.48e-4`, source restriction is `1.73e-16`,
    physical ledgers remain below `1.58e-4`, and checkpoints reload bitwise.
    The N32/N64 S64 thickness response improves from the WP10c7f `0.134682`
    to `0.044619` over the full domain and `0.021412` over `15-60 rg`, but
    both exceed `0.005`. The full peak moves to the first N32 center at
    `1.953 rg`; the interior thermodynamic peak persists at `19.220 rg`.
    N128, longer evolution, and new physics remain closed. Only a
    method-level, nonzero-baseline-preserving balance audit is authorized.
75. WP10c7i separates boundary traces, cell rates, source quadrature, storage,
    and face reconstruction on N16/N32/N64, with N128 residual/JVP oracles.
    One-sided traces remove the boundary peak, but the remaining error requires
    both admissibility-preserving quadratic face traces and shear/height rates
    evaluated along the same reconstructed path as the four-point source
    quadrature. The selected N32/N64 full and `15-60 rg` tangent difference is
    `0.0993839 s^-1`, observed order `2.36087`, projecting to `0.00152799`
    over the bounded horizon. Reductions are `23.87x/13.50x`, above the
    locked `20x/10x` gates. N16/N32 systems remain `245/245` and `485/485`;
    stream recovery and algebraic-map defects are zero. No baseline-specific
    residual correction is retained. Exactly one fresh N32/N64 fixed-BDF2
    trajectory is authorized; no N128, longer duration, or new physics is
    unlocked.
76. WP10c7j independently rebuilds selected-operator N32/N64 states and
    completes all 192 fixed S32/S64 BDF steps over `1.53746e-2 s`.
    The raw N32/N64 thickness-response difference grows monotonically from
    `1.90980e-4` at `T/8` to `1.52769e-3` at `T`; adding both meshes'
    temporal uncertainties gives at most `1.81679e-3 < 0.005`. The endpoint
    is `0.999803` of the WP10c7i tangent projection and improves on WP10c7h
    by `29.21x`. Exact source restriction is `1.73e-16`, physical ledgers
    remain below `2.02e-4`, no snapshot activates admissibility limiting,
    and every checkpoint/sidecar reloads bitwise. Only matched adaptive-BDF2
    confirmation is authorized before any duration extension.
77. WP10c7k advances the same fresh N32/N64 states with the unchanged
    adaptive-BDF2 controller. Both meshes take 13 accepted steps, 12 at
    BDF2 order, with four independent audits and no retries. Adaptive
    endpoints differ from fixed S64 in `Delta log(H/R)` by at most
    `2.10e-5`; the raw N32/N64 endpoint difference is `1.52763e-3`, and
    adding both adaptive-to-reference errors plus both S32/S64 reference
    uncertainties gives `1.85230e-3 < 0.005`. Physical ledgers remain below
    `7.60e-5`, all restarts and T/2 replays are bitwise, and Jacobian work is
    `0.328125` of fixed S64 on each mesh. Exactly one matched no-tide
    extension toward an absolute `~0.05 s` characteristic-crossing horizon
    is authorized.
78. WP10c7l restarts those exact histories and advances production plus
    half-ceiling temporal controls on both meshes to exact common outputs at
    `0.025`, `0.0375`, and `0.05 s`. All four campaigns have zero retries;
    accumulated temporal audits, state gates, physical ledgers, exact source
    restriction, work, checkpoint roundtrips, and endpoint replays pass. The
    conservative N32/N64 thickness budget passes at `0.025/0.0375 s` with
    totals `0.002845/0.004101`, but the raw `0.004944` endpoint response plus
    inherited and new temporal uncertainties gives `0.005348 > 0.005`.
    The raw difference follows a near-perfect `0.099015 s^-1` line, `0.9963`
    of the WP10c7i initial tangent. This is a narrow accumulated spatial stop,
    not a temporal or physical failure; only an evolved-state spatial-order
    and fine-reference audit is authorized.
79. WP10c7m evaluates one evolved N64 physical profile on exact N32/N64/N128
    DAE manifolds using independent PCHIP and natural-cubic oracles. The
    full-domain thickness tangent contracts at order `1.989/1.996`; the
    `15-60 rg` temperature and scaled-energy tangents have minimum order
    `2.127` and `1.875`. The measured WP10c7l endpoint projects to at most
    `0.001246` on N64/N128; adding a `0.0005` combined temporal reserve and
    `5.75e-6` oracle spread gives `0.001751 < 0.0025`. A sparse N128
    consistency solve closes below `5.35e-15`. One fresh N128 production
    plus half-ceiling temporal-control campaign is authorized; raw
    full-domain energy/temperature boundary orders remain diagnostic.

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
   control, WP10c5m source-compatible startup, WP10c5n bounded stop,
   WP10c5o-q mesh-common/temporal-parity controls, WP10c5r spatial
   classification, WP10c5s-t N64 contraction/ledger result, and WP10c5u
   N128 bounded mesh certification, WP10c6a N16 temporal ceiling, and
   WP10c6b N16/N32 controller contract, WP10c6c bounded accumulated-error
   stop, WP10c6d first-order reference gate, WP10c6e refined N16 reference,
   WP10c6f bounded horizon-budget closure, and WP10c7a method-level BDF
   contract as complete.
   The old PW
   plunge has superluminal transverse rotation and must not be mapped into the
   new variables. Continue only the selected one-domain ingoing-Kerr-Schild
   Valencia path. WP10c6e supplies the checksummed 512-step N16 reference with
   raw fine uncertainty below `0.152` of every gate, and WP10c6f closes the
   horizon-budget controller below every combined gate with bitwise restart.
   Backward Euler is now frozen as the reference/startup/fallback backend.
   WP10c7a-d close the bounded temporal-method question through N32. N32 fixed
   and adaptive BDF2 pass their independent temporal references, audits,
   ledgers, work gates, and replay. WP10c7e-f classify and confirm the
   inherited coarse-grid transport error. WP10c7g then certifies an optional
   smooth PLM interior reconstruction with second-order manufactured and
   common-state tangent behavior, full rank, and colored-Jacobian parity.
   WP10c7h completes the authorized N32/N64 trajectory and reduces the
   prior `Delta log(H/R)` mismatch `0.134682 -> 0.044619`, but still misses
   `0.005` by `8.92x`; the `15-60 rg` mismatch is `0.021412`. The
   full-domain peak is now the first-cell boundary trace, while an interior
   thermodynamic balance error persists near `19.22 rg`. WP10c7i resolves
   the method-level blocker with quadratic admissible face traces,
   measure-weighted storage, and source quadrature whose shear/height rates
   follow the same reconstructed path. Its N32/N64 tangent projection is
   `0.001528 < 0.0025`, with order `2.361` and reductions `23.87x/13.50x`;
   no baseline-specific correction is needed. WP10c7j then completes the
   fresh N32/N64 fixed-BDF2 trajectory: raw endpoint mismatch is
   `0.00152769`, and the spatial result plus both temporal uncertainties is
   `0.00181679 < 0.005` at its worst common time. WP10c7k then closes the
   matched adaptive confirmation with a raw endpoint mismatch of
   `0.00152763`, a stricter all-times conservative maximum of `0.00185230`,
   bitwise replay, and `0.3281` of fixed-S64 Jacobian work. WP10c7l reaches
   exact `0.05 s` on both meshes and both temporal controls with all numerical
   and physical gates passing, but the conservative spatial total grows from
   `0.00410055` at `0.0375 s` to `0.00534815` at `0.05 s`, failing the fixed
   `0.005` gate. The raw response grows linearly at `0.099015 s^-1`, matching
   the original selected-operator tangent. WP10c7m now measures full-domain
   thickness order `1.989-1.996` and projects the N64/N128 conservative
   authorization total to `0.001751 < 0.0025`. WP10c7n now replaces that
   projection with a fresh measured N128 trajectory: the raw endpoint
   difference is `0.0012235`, the conservative total is `0.0014873`, the
   observed order is `2.0147`, and the N128 Richardson remainder is
   `0.0004023`. The N128 cell-crossing clock falls to `0.00554 s` while
   stress relaxation remains `0.147 s`. Selected-state finite descriptor
   spectra are authorized. WP10c8a now finds every selected finite mode and
   every isolated `P_R/chi` block stable, but rejects the proposed global
   fieldwise reduction: the candidate fast block spans `0.013-1438 s`,
   retained high-wavenumber modes are as fast as `0.014-0.029 s`, and the
   resulting gap is only about `1e-5`. Dynamic and fast-block numerical
   abscissae are positive and right-eigenvector condition estimates reach
   `1e20`, so transient behavior cannot be inferred from eigenvalues alone.
   WP10c8b then advances all N32/N64/N128 production and temporal-control
   paths through `0.15 s`. The common contract passes through `0.125 s`.
   At `0.15 s`, the conservative N64/N128 total is still only `0.0038168`
   and the order is `1.9618`, but the stronger Richardson remainder is
   `0.0012533`, narrowly above `0.00125`. More decisively, weighted
   stress-target and radial-stationarity departures do not decay over the
   certified interval. Only a certified-state, region-aware operator audit
   is authorized. WP10c8c tests 27 regional `P_R`, `chi`, and joint Schur
   closures on each of N64 and N128. Every fast block is stable and every
   solve is accurate, but no chart passes physical slaving or the required
   timescale gap. The only three response-accurate charts lie at
   `60-200 rg`; their eliminated modes take `3-98 s`, retained modes remain
   as fast as `0.014-0.026 s`, and some effective operators become unstable.
   The global joint closure retains the prior `~1e-5` gap and shows transient
   gains `4.72/8.43`. Instantaneous global or region-selective field
   elimination is rejected. WP10c8d then tests conservation-constrained
   mixed balanced modes at N64/N128 and `0/0.05/0.125 s`. The descriptor and
   exact ledger coordinates pass, but the Hankel maps resolve only orders
   `39-41`; every tested order `8/16/32` realization is unstable, held-out
   responses fail, and the projected unresolved dynamics grow. No nonlinear
   ROM or hyper-reduction is authorized. WP10c8e separately tests the
   stationary-branch preflight. Source amplitudes below `0.1` leave the
   optically thick model domain; amplitudes `0.1-1.0` have full reduced rank
   but large angular/energy imbalance and no physically admissible damped
   Newton trial. Root continuation from these anchors is closed. WP10c8f
   then tests a stability-preserving observable-specific rational ladder.
   Its exact global ledgers reveal spatially converged angular-momentum and
   Killing-energy clocks of `9.36e5 s` and `1.44e6 s`, close to the
   `8.50e5 s` loading time. The dense Lyapunov metric is not numerically
   positive definite; ledger-null LQR corrections stabilize every reduced
   operator and preserve the exact ledger derivatives, but the best trained
   response error remains `1.0002` and cross-mesh transfer error remains
   about `0.32`. Stable compact BPOD and rational realizations are both
   closed. WP10c8g then tests global ledger-driven equation-free closure on
   the existing `0.05-0.125 s` checkpoints. Global `M/J/E` extrapolate
   smoothly, but exactly ledger-null thermal and velocity directions change
   scientific observables by `13-20` gates. The tested observable-augmented
   state also fails, so nonlinear lifting is skipped. WP10c8h retains radial
   conservation explicitly with mesh-coincident five-shell and eight-shell
   `M/J/E` states. Their AB2 errors remain `0.569/1.752` against the `0.25`
   reserve, while within-shell thermal and radial redistributions change
   observables by about `18/10` gates. Compact global and shell-only
   equation-free macrosteps are therefore closed. The full DAE remains the
   short-time truth model; the next work must reassess a physically derived
   dynamic moment/continuum closure or an independent ledger-compatible
   stationary/bifurcation anchor. WP10c8i then implements the complete
   vector responsive-height storage one-form, an evolving-anchor descriptor,
   and an incremental five-shell moment/null-space audit at six N64/N128
   anchors. The storage and coordinate machinery passes its method tests, but
   generator FD-consistency and exact finite-branch contracts do not both pass at
   every anchor. All 12 vector-storage audits pass with a maximum action
   defect of `2.98e-7`; all local tangent-differentiability checks also pass.
   The four declared full generator FD-consistency scans fail with physical JVP
   step-ladder defects `0.210-0.655`, and the exact branch audit finds
   `12/1/27/1` consequential branches at N64 `0/0.025 s` and N128
   `0/0.075 s`. Conditional lower bounds for even the richest 34-coordinate
   set exceed `340` gates and are controlled by interface-4 angular-momentum
   response, but these gains cannot bind the decision. No moment set is
   proven sufficient or insufficient, online cost remains unevaluated, and
   nonlinear lifting is skipped. WP10c8j then replaces the nested
   mass-matrix derivative by differentiating the complete storage-rate action
   directly and separates stationary, storage, storage-rate, factorization,
   and nonlinear-vector-field contracts. The assembled matrices pass: matched
   N64/N128 `0.10 s` generator step stability stays below `1.64e-3`, selected
   storage reconstruction below `2.61e-11`, and factorization below
   `3.64e-12`. The independent nonlinear response does not pass everywhere.
   At N64 `0.05 s`, the selected-step outer thermal/density defects are
   `1.87e-2/1.06e-2`, and the thermal failure persists at every locked secant;
   N128 `0.10 s` independently gives a `1.0209e-2` density defect. At N64
   `0/0.025 s`, every declared direction is reserved by the strict Rusanov
   screen. No all-face candidate coverage, finite neighborhood, or uniform
   nonlinear remainder is supplied, so every branch certificate remains
   nonbinding. WP10c8j therefore rejects an unchanged WP10c8i repeat. The next
   package must repair the outer thermodynamic/vector-field tangent and
   certify the finite-neighborhood Rusanov contract without changing the
   moment ladder. No loading-time macrostep, distributed
   tide, wind, stability, or hot/cycle claim is yet authorized.
   WP10c8k now localizes that smooth failure. The exact centered product
   identity closes near `1e-13`, the independently summed stationary
   derivative agrees near `5e-9`, and more than `99.98%` of the primitive
   mismatch is mapped-conserved storage-rate differentiation. A direct-action
   candidate at scaled displacement `1.28e-2` lowers all controlling L2
   defects below `0.01`, but strict infinity defects remain `0.01028` for the
   smallest outer-density secant and `0.01050-0.01186` for two outer-thermal
   secants. Fourth- and sixth-order mapped-action trials do not close the
   contract, so tangent-only step/order tuning is stopped. Independently, the
   existing aggregate Rusanov enclosure is infeasible even with perfect
   candidate coverage and zero nonlinear remainder: N64 `t=0` consumes
   `2.464/28.58` gates at `0.01/0.025 s`, and N64 `t=0.025 s` consumes
   `0.0193/0.1079`. The next package must make the nonlinear descriptor and
   tangent share one converged mapped-storage construction and replace the
   logarithmic-norm/triangle branch enclosure with a structured low-rank
   finite-time input-output bound. WP10c8i remains blocked and no reduced
   evolution is authorized.
   WP10c8l then makes the mapped descriptor and its rate derivative share one
   audit-only finite-difference implementation of the complete discrete
   instantaneous storage map. The base descriptor reconstructs exactly and
   factorizes at `5.46e-12`, but the locked N64 `0.05 s` fresh-vector-field
   comparison still fails: centered infinity defects are `0.02068-0.02072`
   for the outer-density direction and `0.01840-0.01844` for the outer-thermal
   direction. Both are controlled by `log(T)` rates near `121-131 rg`.
   N128 is therefore not run and further finite-difference tuning is closed.
   In parallel, a structured nominal-semigroup/Volterra preflight using the
   richest WP10c8i weighted null space, direct output changes, and face-aware
   switching gives cached-branch gate fractions no larger than `3.63e-4`,
   with 64/128-panel changes below `0.63%`. This makes the structured
   architecture promising but nonbinding: Track A has no certified final
   generator, the candidate set is incomplete, and no nonlinear neighborhood
   reserve or containment proof exists. The next package must implement an
   exact branch-frozen discrete mapped-storage JVP/Hessian action, then rebuild
   and complete the structured branch certificate serially. WP10c8i remains
   blocked and no reduced evolution is authorized.
   WP10c8m implements that branch-frozen assembled derivative without changing
   the production BDF operator. The locked N64 `0.05 s` and held-out N128
   `0.10 s` cases both pass: descriptor/rate step defects are below `1.86e-9`,
   factorization is below `9.10e-13`, and worst centered infinity defects are
   only `1.29e-5/4.37e-5` against `0.01`. The smooth tangent blocker is
   resolved. Regenerated cached Rusanov identities close below `5.01e-16` and
   their structured zero-remainder fraction remains below `3.64e-4`.
   Expanding pessimistically to all nine noncontrollers on every one of 63
   interior faces gives 567 exact factors but a converged fraction
   `0.06695 > 0.01`, controlled by interface-3 rest-mass flux. This rejects
   the all-candidate enclosure, not the production flux. A sharper certified
   possible-winner/localized branch bound is required before nonlinear
   remainder and containment work. WP10c8i and reduced evolution remain
   blocked.
   WP10c8n performs that possible-winner localization. The failing fraction is
   `99.75%` direct branch-output response and is concentrated at face 58. The
   nominal unit null tube requires a common weighted radius above `2.05`,
   while the unmodified nonlinear production map switches the controlling
   face-58 candidate at radius `0.00582-0.00583` with full reconstruction
   admissibility. The structured null-tube closure retains 449 alternatives
   and reproduces `0.06695`; no containing radius reaches the `0.005`
   headroom target. Candidate-gap screening and the uniform exact-max
   generalized-tangent certificate are therefore closed. The production flux
   remains unchanged. The next admissible reduction experiment is a paired
   finite-amplitude equal-coordinate lifting/healing audit that preserves the
   correlation between state direction and selected flux branch.
   WP10c8o performs that exact nonlinear fiber test. Eight N64 pairs are
   corrected independently onto the exact richest 34-coordinate fiber; all
   coordinate, amplitude, reconstruction, DAE-state, fresh-rate, descriptor,
   and independent storage-action gates pass. The smallest predeclared
   leading-direction counterexample has
   interface-4 angular-momentum half-spread `0.32452995 > 0.25` with maximum
   pairwise coordinate defect `1.17e-15`. Piecewise-constant prolongation of
   its physical perturbation to N128, without N128 output optimization,
   reproduces the same controller at `0.26608550 > 0.25`, cross-mesh spread
   disagreement `0.05844445 < 0.10`, and coordinate defect `1.78e-15`.
   The decisive descriptor ranks are `320/320` and `640/640`, maximum
   full-Schur parity defect is `8.72e-11`, and maximum independent
   path-storage action defect is `1.66e-7`. The face-58 witness ladder gives
   `0.27183-0.30045`, varying smoothly through the exact controller switch.
   The existing 34-coordinate instantaneous deterministic Markov closure is
   therefore rejected on the certified N64/N128 truth discretizations, not as
   a continuum-limit theorem. A healed/equation-free closure, memory variable, or
   conservative coarse PDE remains open; only matched BDF1-start natural
   microbursts of the frozen pair are authorized before selecting one measured
   transport coordinate or auxiliary.
   WP10c8p performs that matched natural-healing screen. Both N64 and N128 use
   synchronized fixed steps, discard the parent history and predictor, take
   one fresh BDF1 startup followed by BDF2, and pass complete coarse/fine,
   fresh-rate, state, physical-ledger, flux-decomposition, and bitwise replay
   contracts. The controlling interface-4 angular-momentum half-spread changes
   only from `0.32452995` to `0.32452655` at N64 and from `0.26608550` to
   `0.26608444` at N128. The fractional decays are `1.05e-5/3.99e-6`, temporal
   uncertainties are below `2.81e-7`, coordinate drift is below `9.09e-8`,
   and final cross-mesh disagreement remains `0.05844 < 0.10`. More than
   `99.9%` of each `M/J/E_K` transport difference is the central perfect-fluid
   flux; causal stress and Rusanov dissipation are negligible. Rapid healing
   through `0.025 s` is therefore rejected on both certified meshes. This is
   not a permanent-memory result: extend N64 only to `0.05/0.10/0.125 s`
   before choosing between healed closure, a measured dynamic interface state,
   or a conservative coarse PDE.
   WP10c8q performs that geometric extension and the direct slow-rate
   sufficiency audit. Applying the five-shell incidence operator to the saved
   interface differences reconciles the complete shell ledgers and rejects the
   flux-gauge interpretation: the decisive pair produces real conservative
   redistribution, with order-unity ambiguity in the leading slow-time vector
   field. A path-integrated perfect-fluid trace decomposition closes below
   `4.80e-14` relative defect and identifies the left radial-velocity trace as
   the controlling primitive contribution. Exact-history N64 `h/h/2`
   continuations reach `0.05/0.10/0.125 s` without another BDF1 startup and
   pass all numerical, physical-ledger, and bitwise-replay contracts. The
   interface-4 angular-momentum spread changes only from `0.32452995` to
   `0.32451281`, corresponding to `5.28e-5` e-folds, so healing is rejected
   throughout the certified horizon. Multiple amplitudes, a held-out
   equal-coordinate direction, a second anchor, and N128 show that the
   unit-normalized interface-4 `M/J/E_K` vectors appear to occupy a rank-two
   plane. WP10c8r adds the missing absolute-significance gate and supersedes
   that final inference: the six independent slow-rate cases have
   interface-4 half-spreads only `2.65e-11-1.18e-8` gate units and
   all-interface maxima below `9.65e-5`. Only the original healing family is
   significant, and it remains approximately rank one. The complete
   slow-rate tangent instead has `4-5` singular directions above `0.1` of the
   leading value across N64/N128; their controlling coordinates span stress
   storage, temperature, radial momentum, angular momentum, and energy, while
   no tested macro-interface response reaches `0.1` gate units. A
   two-component interface-4 state is therefore not authorized. WP10c8s
   constructs six exact nonlinear equal-`q_34` pairs from those complete-rate
   modes. Their maximum slow-rate half-spreads are `26.95-431.04`, with
   matched N64/N128 tangent directions. Modes 0-3 are strongly localized in
   the innermost shell. The binding independent mode-0 N64 `h/h/2`
   trajectory decays but remains decisively unhealed at `0.025 s`: its
   uncertainty-inclusive final lower bound is `16.563` against the `0.10`
   gate, while retained-coordinate drift is below `2.23e-7`. The strict
   decay-curve temporal gate fails, so no relaxation time is claimed; the
   binary nonhealing result is nevertheless separated from the gate by about
   `166x`. This fail-fast result rejects `q_34` plus only one interface-4
   coordinate. The next reduction package must preserve exact BDF history,
   extend and confirm the binding inner mode, and then choose between
   localized additional states and a conservative staggered coarse
   finite-volume/PDE model. No macrostep, tide, or wind work is authorized.
9. Continue one physical distributed tide only after the global no-tide
   duration gate is computationally practical and passes; search for
   accumulation, fronts, hot phases, and limit cycles.
10. Add wind only after the tidal and stability gates pass.

## Review Entry Points

- Equations: [`MODEL_EQUATIONS.md`](MODEL_EQUATIONS.md)
- Reproduction and archive recovery: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- Compact evidence: [`../results/README.md`](../results/README.md)
- Latest causal result:
  `reports/current/CODEX_CAUSAL_COMPLETE_RATE_HEALING_WP10C8S_RESULTS_2026-07-24.md`
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
- Bounded source-compatible duration: `reports/current/CODEX_CAUSAL_BOUNDED_DURATION_WP10C5N_RESULTS_2026-07-18.md`
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
- Causal N64 confirmation WP10c5s-t: `reports/current/CODEX_CAUSAL_N64_CONFIRMATION_WP10C5S_T_RESULTS_2026-07-18.md`
- Causal N128 mesh certification WP10c5u: `reports/current/CODEX_CAUSAL_N128_MESH_CERTIFICATION_WP10C5U_RESULTS_2026-07-18.md`
- Causal N16 timescale and timestep ceiling WP10c6a: `reports/current/CODEX_CAUSAL_N16_TIMESCALE_TIMESTEP_CEILING_WP10C6A_RESULTS_2026-07-18.md`
- Causal N32 temporal-controller contract WP10c6b: `reports/current/CODEX_CAUSAL_N32_TEMPORAL_CONTROLLER_WP10C6B_RESULTS_2026-07-18.md`
- Causal accumulated-error controller WP10c6c: `reports/current/CODEX_CAUSAL_TEMPORAL_CONTROLLER_WP10C6C_RESULTS_2026-07-18.md`
- Causal horizon-budget reference WP10c6d: `reports/current/CODEX_CAUSAL_HORIZON_BUDGET_WP10C6D_RESULTS_2026-07-18.md`
- Causal refined temporal reference WP10c6e: `reports/current/CODEX_CAUSAL_REFINED_REFERENCE_WP10C6E_RESULTS_2026-07-18.md`
- Causal horizon-budget closure WP10c6f: `reports/current/CODEX_CAUSAL_HORIZON_BUDGET_CLOSURE_WP10C6F_RESULTS_2026-07-18.md`
- Increment-primary BDF method WP10c7a: `reports/current/CODEX_CAUSAL_BDF_METHOD_WP10C7A_RESULTS_2026-07-18.md`
- Fixed-step N16 BDF2 WP10c7b: `reports/current/CODEX_CAUSAL_FIXED_BDF2_WP10C7B_RESULTS_2026-07-18.md`
- Adaptive N16 BDF2 WP10c7c: `reports/current/CODEX_CAUSAL_ADAPTIVE_BDF2_WP10C7C_RESULTS_2026-07-18.md`
- Matched N32 BDF2 WP10c7d: `reports/current/CODEX_CAUSAL_MATCHED_BDF2_WP10C7D_RESULTS_2026-07-18.md`
- Localized spatial response WP10c7e: `reports/current/CODEX_CAUSAL_SPATIAL_RESPONSE_WP10C7E_RESULTS_2026-07-19.md`
- N64 BDF2 contraction WP10c7f: `reports/current/CODEX_CAUSAL_N64_CONTRACTION_WP10C7F_RESULTS_2026-07-19.md`
- Causal PLM reconstruction and bounded trajectory WP10c7g-h: `reports/current/CODEX_CAUSAL_SPATIAL_RECONSTRUCTION_WP10C7G_H_RESULTS_2026-07-19.md`
- Causal spatial balance WP10c7i: `reports/current/CODEX_CAUSAL_SPATIAL_BALANCE_WP10C7I_RESULTS_2026-07-19.md`
- Causal spatial-balance trajectory WP10c7j: `reports/current/CODEX_CAUSAL_SPATIAL_BALANCE_TRAJECTORY_WP10C7J_RESULTS_2026-07-19.md`
- Matched adaptive spatial balance WP10c7k: `reports/current/CODEX_CAUSAL_SPATIAL_BALANCE_ADAPTIVE_WP10C7K_RESULTS_2026-07-19.md`
- Characteristic-crossing extension WP10c7l: `reports/current/CODEX_CAUSAL_CHARACTERISTIC_EXTENSION_WP10C7L_RESULTS_2026-07-19.md`
- Evolved-state spatial order WP10c7m: `reports/current/CODEX_CAUSAL_EVOLVED_SPATIAL_ORDER_WP10C7M_RESULTS_2026-07-19.md`
- Fresh N128 reference WP10c7n: `reports/current/CODEX_CAUSAL_N128_REFERENCE_WP10C7N_RESULTS_2026-07-19.md`
- Selected-state slow modes WP10c8a: `reports/current/CODEX_CAUSAL_SLOW_MODE_AUDIT_WP10C8A_RESULTS_2026-07-19.md`
- Causal stress-time audit WP10c8b: `reports/current/CODEX_CAUSAL_STRESS_TIME_AUDIT_WP10C8B_RESULTS_2026-07-19.md`
- Region-selective closure WP10c8c: `reports/current/CODEX_CAUSAL_REGION_SELECTIVE_CLOSURE_WP10C8C_RESULTS_2026-07-19.md`
- Interface-state sufficiency WP10c8r: `reports/current/CODEX_CAUSAL_INTERFACE_STATE_SUFFICIENCY_WP10C8R_RESULTS_2026-07-24.md`
- Complete-rate healing WP10c8s: `reports/current/CODEX_CAUSAL_COMPLETE_RATE_HEALING_WP10C8S_RESULTS_2026-07-24.md`
- Mixed-mode, stationary-preflight, and stable-observable reduction
  WP10c8d-f: `reports/current/CODEX_CAUSAL_STABLE_OBSERVABLE_REDUCTION_WP10C8F_RESULTS_2026-07-20.md`
- Causal inner thermodynamics WP10a: `reports/current/CODEX_CAUSAL_INNER_THERMODYNAMICS_WP10A_RESULTS_2026-07-17.md`
- Horizon-penetrating Valencia core WP10b: `reports/current/CODEX_HORIZON_PENETRATING_VALENCIA_WP10B_RESULTS_2026-07-17.md`
- Valencia gas+radiation primitive recovery WP10c1: `reports/current/CODEX_VALENCIA_GAS_RADIATION_PRIMITIVE_RECOVERY_WP10C1_RESULTS_2026-07-17.md`
- Kerr-Schild geometric finite volume WP10c2: `reports/current/CODEX_KERR_SCHILD_GEOMETRIC_FINITE_VOLUME_WP10C2_RESULTS_2026-07-17.md`
