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

Decisive negative results are retained in the canonical
`global_composite_failure` case and current reports. “Not found in the surveyed
manifold” is not treated as proof of global nonexistence.
