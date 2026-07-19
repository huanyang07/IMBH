# Scientific and Numerical Milestones

The full development diaries remain available at tag
`pre-cleanup-p0-2026-07-11`. This file retains the meaningful accepted and
rejected sequence.

1. **Layered baseline:** fiducial IMRI scales, local vertical S-curves, and a
   one-zone recurrence model reproduced the intended scale estimates.
2. **Imposed-advection caveat:** a local `xi` hot branch was identified as a
   target, not a physical global solution.
3. **Radial entropy upgrade:** `Qadv` was computed from the radial entropy
   derivative; the imposed local hot branch failed the global audit.
4. **Standard slim benchmark:** the isolated no-wind solver recovered the thin
   branch and was continued robustly through `Mdot/Edd=5`.
5. **Transonic free boundary:** radial momentum and sonic regularity were added;
   continuation required analytic/local Jacobians and adaptive meshes.
6. **Finite minidisk and stream source:** finite `Rout`, distributed mass and
   torque sources, compact source shapes, and residual-aware remeshing produced
   a mesh-supported stream-fed no-wind branch.
7. **Heating and wind pilots:** stream heating could be continued, but early
   wind branches exposed hidden source-band and broad finite-volume defects.
8. **Conservative mass coordinate:** global `F=Mdot/Mdot_inner` and source-band
   finite-volume rows replaced hidden differential mass defects.
9. **Lobatto failure:** mixed Hermite/Lobatto source elements could solve local
   DAE equations but their polynomial derivatives were incompatible with the
   stiff physical tangent branch.
10. **Phase-space DAE:** an intrinsic arclength segment solved the stiff source
    transition without dividing by the radial tangent.
11. **Endpoint classification:** the accepted positive branch approaches a
    formal finite-radius low-velocity limit; signed crossings are step-sensitive
    and no finite-state fold is certified.
12. **P0 validity review:** radial/vertical scale separation fails before the
    formal endpoint, the singular annulus mass is locally integrable, and an
    independent outer sheet gives only a near-match at the validity boundary.
13. **Current decision:** freeze `eta_E`; define physical angular transport and
    a unified conservative formulation before any further wind continuation.
14. **Signed conservative reservoir:** independent `Sigma`, exact stream
    moments, signed flux, and angular closure produced regular wall and
    accretion/decretion controls.
15. **Corrected total energy:** enthalpy flux was paired with the compatible
    vertical-work term; fixed-Keplerian transport was rejected near the ISCO.
16. **Conservative interface:** a one-way inner/outer composite closed
    `(Mdot,J,F_E)` and was interface-position stable, but failed primitive
    continuity.
17. **Projected pressure support rejected:** the staggered closure improved
    rotation only on the coarse grid and failed under refinement.
18. **Common stress and simultaneous pressure support:** sharing `W=alpha Pi`
    explains much of the old pressure jump. The simultaneous non-Keplerian
    reservoir closes at `40-60 rg`; `40 rg` is the best coupled-solve target,
    with a remaining N256 density mismatch of `5.7%`.
19. **Fully coupled eigenproblem:** a square inner-transonic/outer-reservoir
    solve releases the inner entropy, sonic state, and angular eigenvalue; the
    `40.0415 rg` interface closes with full Jacobian and boundary-response rank.
20. **Mesh and interface certification:** chained full-root prolongation reaches
    Ninner192/Nouter128, and full-rank roots at `35-50 rg` preserve luminosity
    and fixed-band thickness. The next uncertainty is physical tidal closure,
    not numerical splicing.
21. **Finite-minidisk correction and tidal-power gate:** the coupled reservoir
    edge was corrected from an inherited `10000 rg` numerical buffer to
    `335 rg`. The corrected root remains full rank and interface invariant.
    Paired binary pattern-speed work makes the Hill tidal band exceed
    `H/R=0.3` at 25% power, rejecting perfect confinement and selecting an
    open-overflow solve with emergent inner accretion rate.
22. **Open-overflow eigenvalue:** the fully coupled boundary continuation
    reaches a full-rank open root that accretes `16.9%` of the stream and
    overflows `83.1%`. It remains thin in the Hill band and converges at
    `144/96`, but fails the controlled `168/112` outer-endpoint refinement.
    The declared fallback is coupled conservative mass-energy evolution.
23. **Causal one-domain inner architecture:** the failed low-throughput
    boundary candidates selected one ingoing-Kerr-Schild Valencia column from
    inside the horizon to the Roche edge. Relativistic gas+radiation recovery
    and source-free geometric finite volumes pass local inversion,
    characteristic, tensor-source, and horizon-crossing controls.
24. **Causal relativistic alpha shear:** a rest-frame `R-phi` stress is
    transformed through the same Killing chart and evolved with finite
    Maxwell-Cattaneo relaxation. The common `alpha Pi` law is recovered at a
    reference shear, paired torque/work closes to roundoff, and all selected
    shear modes remain causal. A pressure-amplitude-only advected stress is
    rejected because its weak-field flux Jacobian has a step-stable complex
    pair.
25. **Responsive-height thermal ledger:** a quasi-hydrostatic
    `H(Sigma,T,Omega_perp)` column replaces the fixed-height chart. Its
    physical adiabat includes vertical pressure work, cooling and compression
    are transformed as comoving four-forces, and stress work remains solely
    in the tensor flux. Local acoustic/shear causality, source identities, and
    second-order finite-volume integration pass; the vertical-frequency
    provider and global stream/Roche migration remain open.
26. **Kerr-Schild stream and Roche migration:** one immutable stream
    four-state now supplies exact compact mass, radial-momentum,
    angular-momentum, and Killing-energy moments in the `x^0=ct` chart. The
    physical closed/choked Hill/Roche boundary consumes the relativistic edge
    Killing and angular moments while preserving its reduced local-Hill force
    and opening gate. Exact source, energy/Jacobi, characteristic-count, and
    base face-rank audits pass; the full stress-augmented stationary DAE is the
    next gate.
27. **Five-field causal DAE preflight:** the evolved stress changes the exact
    flux-primary count to `15N+5`. A covariant rest-frame shear operator
    recovers `-R dOmega/dR`, the responsive acoustic/contact/shear principal
    has five real causal modes, the inner excision has zero incoming modes,
    and the Roche edge has two independent incoming responses. Temporal
    height work is mapped into all Killing storage components. Production
    roots remain blocked until these pieces are assembled into one
    path-conservative nonlinear residual with a fifth zero-stress Roche face.
28. **Five-field causal DAE assembly:** the complete `15N+5` flux-primary
    residual now includes the straight covariant-shear path, responsive radial
    and temporal height work, cooling, exact optional stream moments, geometric
    sources, excision flux, and five-component Roche face. At N16 the
    descriptor storage has exact rank `80/80` and backward Euler is full rank,
    but the stationary response is stably `244/245`, localized to an outer
    thermal/stress direction. The locked gate stops before N64/N96 roots and a
    timestep pending one reduced primitive null-mode audit.
29. **Reduced primitive null audit:** exact elimination of the `165/165`
    conserved-state and face-flux identity block produces an `80/80` primitive
    stationary response. Direct remapping and the Schur operator agree to
    `2.99e-11`, the outer thermal/stress response is `2/2`, and opening the
    same Roche provider strengthens rather than supplies the weak direction.
    The former `244/245` result is flux-primary embedding conditioning at a
    nonroot seed, not a missing boundary condition or physical marginal mode.
30. **Index-one consistent initialization:** the N16 descriptor and algebraic
    tangent system is `245/245` and balances the nonzero conservation residual
    below `9.1e-15`. Two bounded tangent-sized backward-Euler attempts preserve
    exact algebraic maps but stop at `4.79e-6` and `1.40e-6`, localized to
    outer-cell mass and angular-momentum temporal storage. N32 and physical
    evolution remain blocked pending one cancellation-safe storage-increment
    audit.
31. **Cancellation-safe temporal storage:** direct endpoint subtraction differs
    from the converged primitive-path storage rate by as much as `7.05e-6`.
    The path identity converges below `2.53e-9` and telescopes below `5.60e-17`,
    but its two bounded N16 steps still stop at `3.77e-6` and `1.42e-6`.
    Storage cancellation is real but not the sole blocker; the remaining floor
    tracks a reduced Newton condition near `1.03e10`, and N32 remains blocked.
32. **Frozen reduced linear precision:** LAPACK equilibration reduces the final
    N16 matrix condition estimate from `1.03e10` to `27.5`, but direct and
    iteratively refined corrections agree to `2.62e-14` and solve the linear
    equation to `1.49e-16`. A fourth-order Jacobian agrees to `4.47e-12`, and
    every full correction gives nonlinear residual `3.35e-6`. No recoverable
    linear precision is found, so N16 is not repeated and N32 remains blocked.
33. **Component directional consistency:** compensated diagnostic differences
    reconstruct the residual change below `2.8e-16`. Flux, source, and
    responsive-height blocks follow the Newton direction below `2.1e-13`,
    while path conserved storage alone misses by `3.35e-6`. One authorized
    fixed-coordinate Jacobian-vector repair passes focused tests but its
    single N16 retry still stops at `1.42473e-6`; the post-repair storage defect
    is `1.42457e-6`. N32 remains blocked. Any continuation must carry
    `Delta U` as a primary unknown rather than perform another storage scan.
34. **Increment-primary causal startup:** the same complete `15N+5` DAE uses
    `(Delta U,Delta p,Delta F)` as Newton coordinates, with conserved storage
    entering backward Euler directly. Equilibrated N16 and N32 systems are
    `245/245` and `485/485`; both bounded steps pass below `8.8e-9`.
    One-full-step versus two-half-step differences are `2.76e-6` and
    `1.01e-6` of the full changes. Short source-on no-tide startup is now
    authorized; physical relaxation, stability, tide, wind, and hot-state
    claims remain blocked.
35. **Exact-stream sparse repeated startup:** exact circularized C2 source
    moments pass N16/N32 startup and temporal gates. An exact 18-color local
    Jacobian reproduces the dense matrices and roots while reducing each
    assembly to 36 residual evaluations. Eight N16 and seven equal-time N32
    steps pass adaptive, conservation, optical-depth, and bitwise-restart
    gates through `3.39278e-7 s`. The baseline-subtracted common-radius
    thickness response is mesh supported, but the duration is only about
    `2e-13 t_load` and the arbitrary preflight seed drains at about `9.2e4`
    times the stream supply. A source-compatible causal datum is required
    before longer physical evolution.
36. **Matched-source and source-compatible causal startup:** bitwise-identical
    source-on/source-off controls recover all four prescribed stream moments
    to `3.25e-6/1.08e-6` at N16/N32, with cross-mesh isolated mass and
    thickness-response defects below `1.39e-9`. A new constrained datum has
    unit inner throughput relative to the stream, `H/R=0.1`, scattering depth
    above `18.5`, zero incoming inner modes, a closed Roche channel, and full
    scaled/equilibrated rank. The repeated N16/N32 trajectories reach the
    exact shared time `5.54201e-5 s` with aggregate mass defects below
    `1.57e-11` and a baseline-subtracted thickness response difference of
    `1.00e-3`. One geometric no-tide duration extension to about
    `1e-9 t_load` is authorized; N64/N96, tide, wind, stability, hot-state,
    and cycle searches remain blocked.
37. **Bounded source-compatible duration stop:** N16 and N32 each reach the
    exact shared time `6.781724319e-4 s` with full descriptor/consistency
    rank, no rejected attempts, a closed Roche edge, zero inner incoming
    modes, scattering depth above `18.67`, and mass/five-field defects below
    `4.5e-12`. The inner and outer flux responses agree, but the common-radius
    `Delta ln(H/R)` response differs by `1.2557e-2`, failing the fixed `5e-3`
    mesh gate broadly around `12-16 rg`. The independently tuned moving-cell
    initial profiles prevent a clean continuum interpretation. Further
    duration stops pending one mesh-common physical initial datum; N64/N96,
    tide, wind, stability, hot-state, and cycle work remains blocked.
38. **Mesh-common startup and temporal-parity stop:** one fixed C2 primitive
    profile anchored at `6 rg` and `240 rg` gives exact unit throughput and
    passes the N16/N32 common-data, causal, optical, Roche, map, and rank gates.
    The short response mismatch is `2.7898e-3 < 5e-3`. Both bounded
    trajectories pass separately, but their common-time `Delta ln(H/R)`
    response differs by `2.1033e-2`. A control using the same maximum timestep
    and exactly 63 extension steps on both meshes changes either response by
    at most `3.00e-6`, excluding temporal alignment as the explanation. The
    remaining failure is spatial at N16/N32. A term-resolved semidiscrete audit
    is required before N64 or any physical evolution.
39. **Causal spatial-response classification:** exact constrained tangents
    reconstruct below `2.7e-11`, and conservative N32-to-N16 restriction
    localizes the broad thickness-rate discrepancy to face transport:
    `24.0482 s^-1` at `55.5662 rg`, versus `1.58902 s^-1` from the next
    source contribution. Linear and PCHIP comparisons give the same result.
    Operator-only N16-N128 checks recover central order at least `1.9961`,
    Rusanov/full-transport order at least `1.1058`, all source orders at least
    `1.9837`, and exact-stream restriction below `2.26e-16`. The prior
    duration mismatch is ordinary first-order coarse-grid truncation. No
    operator correction is justified; one bounded N64 confirmation is
    authorized before any longer or physical evolution.
40. **N64 contraction and ledger-tight confirmation:** an independently
    generated N64 datum passes the exact-time N32/N64 short gate with response
    error `8.6493e-4`. The baseline duration passes every state, rank, and step
    gate but accumulates a five-field ledger defect of `1.8082e-10`. One
    stricter-residual replay lowers that defect to `1.8084e-12` while changing
    the response by only `5.01e-10`. The bounded N32/N64 response error is
    `6.6677e-3`, above `5e-3`, but contracts from N16/N32 at order `1.657`
    (RMS `1.463`). Exactly one bounded N128 confirmation is authorized; no
    longer or physical evolution is unlocked.
41. **N128 bounded spatial certification:** an independently generated N128
    datum passes the N64/N128 short gate with response error `3.33334e-4`.
    The strict exact-time duration reaches `8.48423e-4 s` in 63 extension
    steps with no retries, full `640/640` descriptor and `1925/1925`
    consistency rank, and five-field defect `1.77056e-12`. The N64/N128
    response error is `2.58967e-3 < 5e-3`, contracting at observed order
    `1.36443` (RMS `1.22561`). The first-order bounded mesh gate is certified;
    further fine meshes and direct microstep duration extension are closed.
42. **Localized longer-horizon spatial classification:** exact Kerr-Schild
    restriction reproduces the WP10c7d fixed/adaptive N16/N32
    `Delta log(H/R)` mismatch at `0.613215/0.613234`; the largest
    fixed/adaptive history effect is only `7.62e-5`. The mismatch crosses
    `0.005` on the first fixed-S64 step and grows approximately linearly at
    first. Initial DAE-consistent attribution is controlled by total face
    transport at `24.1407 s^-1`, with Rusanov `13.5426 s^-1`, central
    transport `12.0895 s^-1`, and the next source `2.60490 s^-1`.
    Combined with the prior manufactured first-order Rusanov result, this
    confirms inherited coarse-grid truncation and authorizes exactly one N64
    fixed S32/S64 contraction diagnostic.
43. **Longer-horizon N64 contraction stop:** fixed N64 BDF2 S32/S64 reaches
    the exact WP10c7d horizon with all state and ledger gates passing. Raw
    temporal thickness uncertainty is `1.53598e-4`, below the preferred
    `2.5e-4`. Exact N32/N64 restriction contracts the response difference
    `0.613215 -> 0.134682` at order `2.18684`, but remains `26.9x` above the
    spatial gate. At the measured order, N64/N128 is projected to differ by
    `0.02958`, still `5.92x` above the gate. N128 and uniform refinement are
    closed; only a separate operator-level second-order reconstruction audit
    is authorized.
44. **Second-order interior reconstruction:** optional smooth and unlimited
    PLM reconstruction in `ln(R)` feeds the complete causal Rusanov face
    calculation while the piecewise-constant backend remains frozen.
    Smooth-PLM finest-pair manufactured order is at least `1.910`; diagnosed
    total/full tangent orders are `2.116/2.172`; and the N32/N64 full tangent
    discrepancy falls by `5.235x`. The widened N8 Jacobian uses 23 colors and
    agrees with dense differences to `1.27e-10`; N16/N32 consistency rank
    remains `245/245` and `485/485`. The unchanged physical boundary traces
    remain first order, so only one bounded reconstructed trajectory is
    authorized.
45. **Reconstructed-flux trajectory stop:** independently initialized N32
    and N64 S32/S64 campaigns complete all 192 fixed BDF steps with temporal
    thickness uncertainty below `1.48e-4`, source restriction `1.73e-16`,
    physical ledgers below `1.58e-4`, and bitwise restart. PLM reduces the
    prior N32/N64 thickness mismatch `0.134682 -> 0.044619` over the full
    domain and to `0.021412` over `15-60 rg`, but both fail `0.005`. The full
    peak moves to the first-cell boundary trace at `1.953 rg`; the interior
    thermodynamic peak persists at `19.220 rg`. N128 and longer evolution
    remain closed; only a nonzero-baseline-preserving balance audit is
    authorized.
46. **Full-domain spatial-balance certification:** boundary, rate, source,
    storage, and reconstruction ablations show that the WP10c7h error requires
    a state-dependent repair rather than a constant baseline correction.
    Admissibility-preserving quadratic face traces plus four-point source
    quadrature with locally reconstructed shear/height rates reduce the
    N32/N64 full and `15-60 rg` tangent discrepancy by `23.87x/13.50x`.
    Both converge at order `2.36087` and project to `0.00152799`, below the
    locked `0.0025` pre-trajectory budget. N16/N32 rank remains full, exact
    stream moments are unchanged, and the colored Jacobian has no omitted
    N4 entries. One fresh bounded N32/N64 trajectory is authorized; N128,
    longer evolution, and new physics remain closed.
47. **Bounded spatial-balance trajectory certification:** fresh N32/N64
    selected-operator histories complete all 192 S32/S64 fixed BDF steps.
    Exact common-time restriction gives a monotone `Delta log(H/R)`
    difference from `1.90980e-4` at `T/8` to `1.52769e-3` at the endpoint.
    Adding both meshes' temporal uncertainties gives at most
    `1.81679e-3 < 0.005`. The endpoint is `0.999803` of the WP10c7i tangent
    projection and `29.21x` below the prior smooth-PLM trajectory mismatch.
    Source restriction is `1.73e-16`, physical ledgers remain below
    `2.02e-4`, no stored snapshot activates limiting, and all restarts are
    bitwise. One matched adaptive-BDF2 confirmation is authorized before
    longer no-tide evolution.
48. **Matched adaptive spatial-balance certification:** the unchanged
    adaptive-BDF2 controller advances both selected-operator meshes through
    the WP10c7j horizon in 13 accepted steps with four independent audits
    and no retries. Adaptive-to-fixed S64 `Delta log(H/R)` errors remain
    near `2e-5`; the raw N32/N64 endpoint difference is `1.52763e-3`, and
    the strict spatial total including both adaptive errors and both fixed
    reference uncertainties is `1.85230e-3 < 0.005`. Physical ledgers stay
    below `7.60e-5`, T/2-to-T replays are bitwise, and each mesh uses
    `0.328125` of fixed-S64 Jacobian work. One matched no-tide extension
    toward the `~0.05 s` characteristic-crossing horizon is authorized.
49. **Characteristic-crossing spatial stop:** matched N32/N64 production and
    half-ceiling temporal-control trajectories reach exact `0.05 s` with no
    retries. Accumulated temporal audits, state gates, physical ledgers, exact
    source restriction, work, limiter, and bitwise replay contracts all pass.
    The conservative thickness-response budget passes at `0.025/0.0375 s`
    with `0.002845/0.004101`, but the raw endpoint difference `0.004944`
    becomes `0.005348 > 0.005` after inherited and new temporal uncertainty.
    The raw mismatch grows linearly at `0.099015 s^-1`, within `0.4%` of the
    WP10c7i initial tangent, so this is accumulated spatial truncation rather
    than a temporal or physical failure. Stress/cooling/thermal extension
    remains closed pending an evolved-state spatial-order and reference audit.
50. **Evolved-state N128 authorization:** independent PCHIP and natural-cubic
    representations of the N64 `0.05 s` state are remapped onto exact
    N32/N64/N128 DAE manifolds. The full-domain thickness tangent contracts at
    order `1.989/1.996`; the controlling interior temperature and scaled-energy
    orders are at least `2.127/1.875`. The projected N64/N128 endpoint
    difference is at most `1.2456e-3`; adding the complete planned temporal
    reserve and oracle spread gives `1.7513e-3 < 2.5e-3`. The sparse N128
    consistency solve closes below `5.35e-15`. One fresh N128 production plus
    temporal-control campaign is authorized, while boundary-limited raw
    temperature/energy orders remain diagnostic.
51. **Fresh N128 `0.05 s` spatial certification:** independently initialized
    N128 production and half-ceiling temporal-control trajectories reach exact
    `0.025`, `0.0375`, and `0.05 s` in `30/60` accepted steps with no retries.
    The measured N64/N128 thickness difference grows
    `6.135e-4 -> 1.223e-3`; adding complete temporal uncertainty gives
    `8.340e-4 -> 1.487e-3`, below both the original `0.005` gate and preferred
    `0.0025` half-gate. The observed order is `2.0147`, the remaining N128
    Richardson estimate is `4.023e-4`, physical ledgers remain below
    `1.52e-4`, and production replay is bitwise after excluding wall-clock
    telemetry. Selected-state finite descriptor spectra are authorized before
    any slow-manifold reduction claim.

Decisive negative results are retained in the canonical
`global_composite_failure` case and current reports. “Not found in the surveyed
manifold” is not treated as proof of global nonexistence.
