# ADR 0013: Global signed conservative transonic evolution

## Status

Accepted as the fallback architecture; physical flux closures are pending.

## Context

The coupled outer-evolving/inner-quasi-steady DAE passes bounded coarse tests
but fails evolved refinement at `24/16`. A cross-interface radial stencil fixes
the original radial endpoint defect. Primitive boundary elimination instead
moves the defect into the inner transonic core. Further splice conditioning is
closed under ADR 0012.

The fallback must remove the artificial interface and remain regular through
inflow, stagnation, decretion, and the sonic transition.

## Decision

Use one radial finite-volume domain with cell-integrated differential state

\[
U_i=(M_i,P_{R,i},J_i,E_i),
\]

where

\[
M_i=A_i\Sigma_i,
\qquad
P_{R,i}=M_iv_{R,i},
\qquad
J_i=M_iR_i^2\Omega_i.
\]

`E_i` is column total energy. Its final definition must use the same
Paczynski-Wiita potential, energy zero, vertical closure, and thermodynamics as
the existing total-energy identity.

All face fluxes are explicit algebraic variables:

```text
mass flux
radial-momentum flux
angular-momentum flux
total-energy flux
```

Fluxes are outward-positive. Therefore the established inward-positive disk
rates satisfy

\[
F_M=-\dot M,
\qquad
F_J=-{\cal J},
\qquad
F_E=-{\cal F}_E.
\]

Each component obeys

\[
\frac{dU_i}{dt}=F_{i-1/2}-F_{i+1/2}+S_i.
\]

The first mixed descriptor count is

\[
\boxed{
4N+4(N+1)=8N+4
}
\]

unknowns and rows. The descriptor storage rank is `4N`.

## Sonic treatment

There is no stationary sonic eigencondition in the time-dependent initial
value problem. Radial momentum is differential, so the sonic transition must
emerge dynamically. The inner boundary must be placed in a causally outgoing
region or implemented with a characteristic absorbing condition. A sonic
regularity condition may be used only when polishing a time-averaged steady
state, not as an extra evolution row.

## Required physical closures

The manufactured face closures are rank tests only. Production proceeds in
this order:

1. Conservative-to-primitive recovery using the shared vertical closure.
2. Inviscid mass and radial-momentum fluxes in the Paczynski-Wiita potential.
3. Geometric, gravitational, and centrifugal radial-momentum sources.
4. Common alpha stress in angular and total-energy fluxes, including torque
   work exactly once.
5. Radiative cooling and temporal column work.
6. Exact stream `S_M,S_J,S_E` moments.
7. Absorbing inner and open outer boundaries with no unconfigured inflow.
8. Physical distributed tide and paired power only after no-tide convergence.
9. One conservative wind state only after the tidal/stability gates pass.

## Prohibitions

- Do not divide a primitive closure by `v_R` or `Mdot`.
- Do not represent signed radial flow with a logarithm.
- Do not clip accepted mass, internal energy, or velocity states.
- Do not impose a perfect wall in the first global validation.
- Do not add tide or wind before the no-source and stream-fed open controls
  pass mesh and timestep convergence.

## Initial evidence

Manufactured `N=8,16,32` systems pass:

```text
descriptor rank                         4 N
backward-Euler rank                     8 N + 4
signed mass-flux zero crossing          present
maximum scaled cell residual            <= 1.1e-16
maximum relative global-ledger defect   <= 1.6e-15
maximum temperature round-trip error    <= 4.6e-12
constant-Pi Keplerian balance error      <= 3.3e-11
pressure-supported error ratios          7.63, 7.80
```

The conservative-to-primitive inversion uses the shared Paczynski-Wiita
potential and vertical closure, preserves signed radial velocity, and rejects
states whose internal energy cannot bracket a positive temperature. This
evidence certifies the layout, count, rank, and thermodynamic round trip. It is
not a physical transonic solution.

The first smooth inviscid operator pairs

\[
F_{P_R}=2\pi R(\Sigma v_R^2+\Pi)
\]

with the cell source

\[
S_{P_R}=2\pi\Delta R\Pi
+A\Sigma R(\Omega^2-\Omega_K^2).
\]

It preserves the constant-pressure Keplerian manufactured state to the
thermodynamic inversion floor. A pressure-supported power-law equilibrium
converges by factors `7.63` and `7.80` on `N=8,16,32`. This is a smooth-state
balance gate. A shock-capable, positivity-preserving numerical flux remains
required before production evolution.

The first shock diagnostic uses a local Lax-Friedrichs flux on deviations from
a supplied equilibrium. Potential energy is transported with the numerical
mass flux rather than diffused as thermal energy. On a discontinuous test, a
`CFL=0.005` step preserves positive mass and temperature; a ten-times larger
step is rejected and returns the original state without clipping. At the
reference state, the corrected flux equals the smooth well-balanced flux to
roundoff. This is a bounded first-order gate, not yet a production timestep
certification.

The declared open boundary blocks advective mass, angular momentum, and energy
when either edge velocity points into the domain. It retains pressure traction
and leaves genuine inner accretion and outer overflow unchanged. The boundary
does not create a ghost reservoir.

A source-free `N=32` equilibrium perturbation accepts one, two, and four
explicit steps over the same interval. The full-step/two-half-step error ratio
is `2.0008`, consistent with first-order Euler time integration. The maximum
ledger defect is `2.9e-17` relative to conserved storage; the activity-scaled
number is `6.1e-5` because the equilibrium changes are intentionally tiny.

Sampling one fixed analytic equilibrium on `N=16,32,64` meshes and advancing
four source-free steps gives interior radial-drift convergence factors `3.09`
and `3.51`. The maximum drift is instead localized in the final cell adjacent
to the open boundary; it remains below `3.1e-10 c` and decreases slightly with
refinement. This is classified as a bounded boundary response, not an exact
open-boundary equilibrium.

The common alpha stress is added as an outward viscous torque `+G` in the
angular flux and paired work `+Omega*G` in the total-energy flux. These signs
are required by the global outward-positive orientation and are equivalent to
the older inward-positive `Mdot*l-G` and `Mdot*B-Omega*G` conventions. Mass and
radial momentum are unchanged, and there is no separate viscous-heating source.
Stress-bearing evolution must use an implicit or IMEX update rather than the
acoustic explicit step.

The first explicit-inviscid/implicit-stress split accepts two manufactured
steps but rejects the third on thermal positivity. It is retained as a negative
pilot. A monolithic backward-Euler solve in positive `Sigma`, `Omega`, and
temperature plus signed radial velocity accepts four consecutive steps with a
maximum residual of `3.52e-14` and storage-scaled ledger defect `5.89e-16`.
Production evolution therefore uses the monolithic path; the split path is not
to receive a timestep or damping scan.

The monolithic path subsequently accepts eight steps at `N=8` and `N=16` over
the same `2.5800 s`. Four-versus-eight-step differences at `N=8` remain below
`1.45e-4` in the positive primitives. The `N=16` minimum temperature shifts by
about `2.7%`, so no continuum claim is made.

The total-energy equation is scaled by thermal storage. This prevents cooling
and viscous heating from hiding beneath orbital binding energy. Under that
stricter gate, a twelve-color nearest-neighbor Jacobian fails while dense
differencing passes; the colored path is closed pending analytic sparse
derivatives. Shared radiative diffusion cooling passes a local first-order
backward-Euler audit and a full monolithic cooled-versus-adiabatic comparison.

The first stream source uses exact compact-C2 cumulative increments and one
constant injected state `(v_R,s,l_s,E_s)`. All four integrated source moments
are mesh independent to roundoff on `N=16,32,64`, and one manufactured
source-bearing monolithic step passes. Radially varying ballistic moments must
later provide analytic cumulative moments or a separately certified high-order
quadrature; they must not fall back to independent center-sampled ledgers.

The physical circularization control maps `5 Mdot_Edd` into the global domain
with `v_R,s=0`, `l_s=l_K(248.96693 rg)`, and circular-orbit energy. One versus
two steps at `N=16` agrees, but the `N=16` and `N=24` outer fluxes differ by
`1.068` times the supply. A bounded `N=24`/`N=32` comparison still differs by
`0.0809` supply. Mapping-only evaluations at `N=16,24,32,48` reproduce the
boundary-flux dependence before evolution, while adding the exact stream cell
source changes no face mass flux. The defect is therefore assigned to the
pointwise primitive remap/open-face initialization, not to source quadrature
or the implicit solve. No tide, wind, or longer loading run may start until a
conservative finite-volume remap and open-face reconstruction pass the mesh
gate.

That remap is now implemented with exact annular geometry and eight-point
quadrature of all four conserved densities. Meshes through `N=48` are rejected
at primitive recovery rather than projected; `N=64` and `N=96` are admissible.
At `N=96`, the mapped inner and outer mass fluxes differ from the coupled
targets by `2.61e-4` and `1.62e-3` supply. A full `N=64` step and two half steps
all pass, with inner/outer flux differences `9.37e-8` and `9.45e-11` supply.
The remaining gate is evolved `N=64`/`N=96` agreement. Because dense thermal
differencing is already expensive at `N=64`, the prior rejected colored
Jacobian must be replaced by derivative-verified sparse blocks before that
comparison and any long evolution.

The replacement sparse path independently evaluates every local one-sided
column and certifies the declared stencil against the complete dense columns.
The physical `N=64` and `N=96` states have exactly zero measured off-pattern
derivative and both sparse solves pass below `1.3e-12`; exact matrix
factorization is retained because LSMR stalls above the fixed gate. The evolved
outer flux nevertheless changes by `0.02846` supply (`3.31%`) from `N=64` to
`N=96`, so the spatial gate fails. This shift is already present in the mapped
face values and is assigned to separately extrapolating `Sigma` and `v_R` at
the open edge. The next and only authorized boundary correction is a direct
conservative mass-flux face reconstruction, followed by the same one-time
`N=64`/`N=96` comparison.

That bounded correction is complete. A first-order conserved-donor outer face
uses one mass flux for radial, angular, and total-energy advection; all three
donor consistency defects are zero and the `N=96`/`N=128` mapping difference
passes at `0.00782` supply. The evolved `N=64`/`N=96` outer difference improves
from `0.02846` to `0.01150` supply but narrowly fails the fixed `0.01` gate.
The inner-flux, thickness, residual, Jacobian, and ledger gates pass. Further
reconstruction variants are prohibited. The only authorized outer-boundary
fallback is one characteristic or modeled Roche-overflow contract with a
declared exterior condition.

The characteristic preflight is now complete. On the conservative
`N=64,96,128` mappings the outer radial Mach number is only
`0.0090, 0.0128, 0.0100`. The radial Euler eigenvalues therefore contain
exactly one negative acoustic speed at the outward-facing edge: one exterior
condition is required. The live `335 rg` edge is not a Roche saddle; it is
`0.4485 R_H` and `0.8970` of the repository's current fiducial truncation
radius. No exterior pressure, entropy, or Bernoulli state is declared there.
Consequently an arbitrary pressure outlet, vacuum ghost state, or
mesh-dependent reference state is prohibited. The characteristic/Roche
boundary remains physically underdetermined until either (a) Layer 1 supplies
one exterior thermodynamic invariant at `335 rg`, or (b) the domain gains a
modeled Hill/Roche overflow layer terminating at an actual escape saddle.

WP2 retains the existing conserved total energy and completes its one-zone
identity with two matched terms. The outward-flux profile adds radial column
work, evaluated with inward-positive `Mdot=-F_M`, to the total-energy source:

```text
W_R = Mdot * (Pi/Sigma) * Delta ln H.
```

The backward-Euler storage row adds the trapezoidal temporal work

```text
W_t = 0.5 * (M_new h_new + M_old h_old) * ln(H_new/H_old).
```

This is the same enthalpy-compatible identity previously certified against the
transonic entropy equation. Manufactured radial and temporal tests, a
source-bearing corrected residual/ledger test, and a torque non-double-counting
test pass. Physical donor-face `N=64/96` tiny steps remain accepted with
column-work power near `0.0399 L_Edd`, residuals below `1e-11`, and ledgers
below `3e-16`. The later WP4 remap changes the numerical outer-flux comparison
without changing this identity or concealing the independent physical boundary
closure gate.

WP3 finds that the fixed `5.21024 rg` inner edge is not causally outgoing for
the time-dependent radial Euler block. After the final WP4 remap, the
conservative `N=64,96,128` first cells have Mach numbers
`-0.654,-0.763,-0.818` and exactly one positive
`v_R+c_eff` characteristic entering the domain. The former no-inflow diode is
therefore insufficient by itself.

The selected inner boundary projects only that incoming acoustic perturbation
to zero relative to the fixed certified transonic reference. It preserves the
outgoing acoustic invariant and the two inward-advected contact fields, carries
one projected donor state consistently through mass, radial momentum, angular
momentum, and Bernoulli energy, and retains the diode solely as a final guard
against black-hole ghost outflow. The reference profile is bitwise unchanged;
a manufactured outgoing mode is unchanged below `2e-8` relative. Physical
`N=64/96` tiny steps remove the incoming amplitude by more than seven orders of
magnitude, preserve the outgoing amplitude below `4e-8` relative, and pass
with residuals below `2.9e-12` and ledgers below `3.3e-16`. The accretion flux
changes by less than `2.4e-5` supply relative to the unprojected WP2 control.

WP4 removes the cell-average/center-point mechanical-energy contamination with
a fixed finite-volume reference correction. For each mapped annulus, 32-point
quadrature separately integrates mass-weighted mechanical and internal energy.
Primitive recovery subtracts

```text
delta e_mech = <Phi + v_R^2/2 + v_phi^2/2>_M
               - [Phi + v_R^2/2 + v_phi^2/2]_cell-center
```

from the specific total energy. The same correction is used when reconstructing
every trial state, while the conserved total-energy storage and physical face
flux remain unchanged. It is a fixed well-balanced reference, not a floor.
All conservative `N=16-128` mappings now recover positive internal energy.
The `N=64,96,128` 32/64-point quadrature comparison is below `1.2e-3` in every
conserved field and below `1.9e-4` in temperature. The correction decreases
monotonically on `N=64,96,128`. Selected `N=64/96` physical steps retain
residuals below `1e-11` and ledgers below `3e-16`; their outer-flux difference
is `0.00635` supply and passes the fixed numerical mesh gate.
