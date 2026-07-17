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

Decisive negative results are retained in the canonical
`global_composite_failure` case and current reports. “Not found in the surveyed
manifold” is not treated as proof of global nonexistence.
