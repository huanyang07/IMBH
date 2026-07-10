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

Decisive negative results are retained in the canonical
`global_composite_failure` case and current reports. “Not found in the surveyed
manifold” is not treated as proof of global nonexistence.
