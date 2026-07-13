# Global signed descriptor prototype results

## Scope

This work starts the one-domain fallback required after the hybrid time DAE
failed evolved-mesh certification. It implements the exact four-field
finite-volume storage and ledger kernel, not yet the physical hydrodynamic flux
closure.

## Architecture

The differential cell state is

\[
(M,P_R,J,E),
\]

and all four outward-oriented face-flux profiles are algebraic unknowns. For
`N` cells:

```text
differential variables     4 N
algebraic face variables   4 (N + 1)
total unknowns/rows        8 N + 4
descriptor rank            4 N
```

Radial momentum is differential. The time-dependent sonic transition is
therefore not an interface or an added eigencondition.

## Manufactured audit

| `N` | Unknowns | Descriptor rank | BE rank | Max scaled residual | Max ledger defect | Temperature round trip |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 68 | 32 | 68 | `1.02e-16` | `1.49e-15` | `3.02e-12` |
| 16 | 132 | 64 | 132 | `8.54e-17` | `6.99e-16` | `4.59e-12` |
| 32 | 260 | 128 | 260 | `9.27e-17` | `1.57e-15` | `4.46e-12` |

Every mesh contains inward flow, outward flow, and an exact zero crossing in
the manufactured mass-flux profile. No state construction divides by that
flux.

The global mass, radial-momentum, angular-momentum, and total-energy ledgers
are audited from boundary fluxes and integrated sources independently of the
cell residual concatenation.

Conservative-to-primitive recovery also round-trips a physical
Paczynski-Wiita disk spanning `6.2-300 rg`, with signed radial velocity and the
shared gas-plus-radiation vertical closure. Surface density is recovered to
`2.3e-16`, radial velocity to `2.5e-19 c`, and temperature to `4.6e-12`.

## Smooth radial balance

The first inviscid operator evaluates outward face fluxes

\[
F_M=2\pi R\Sigma v_R,
\]

\[
F_{P_R}=2\pi R(\Sigma v_R^2+\Pi),
\]

with matching angular and enthalpy-carrying total-energy fluxes. Its radial
cell source is

\[
S_{P_R}=2\pi\Delta R\Pi
+A\Sigma R(\Omega^2-\Omega_K^2).
\]

The constant-`Pi`, Keplerian equilibrium residual remains below `3.3e-11` on
all tested meshes. For a nontrivial pressure-supported power-law equilibrium:

| `N` | Maximum normalized radial imbalance |
|---:|---:|
| 8 | `1.8966e-4` |
| 16 | `2.4869e-5` |
| 32 | `3.1889e-6` |

The refinement ratios are `7.63` and `7.80`. A separately constructed smooth
flow crosses a reconstructed zero-mass-flux face while all four inviscid
fluxes remain finite.

## Shock-flux preflight

A residual-equilibrium Rusanov flux supplies dissipation only to deviations
from a reference equilibrium. Its energy diffusion excludes the
Paczynski-Wiita potential-energy jump and transports face potential with the
numerical mass flux.

For a discontinuous density, temperature, and radial-velocity state:

```text
CFL                                      0.005
step accepted                            yes
maximum global-ledger defect             3.69e-14
ten-times larger step rejected           yes
rejected state returned unchanged        yes
equilibrium-corrected flux difference    0
unconfigured inner/outer inflow blocked  yes/yes
```

No mass, energy, temperature, or velocity clipping is used.

## Source-free temporal preflight

The pressure-supported `N=32` reference is advanced over `1.4098 s` using one,
two, and four explicit steps with the equilibrium-corrected flux and no-inflow
boundaries:

```text
full-step CFL                          0.02
all steps accepted                    yes
one-vs-two step error                 2.4255e-6
two-vs-four step error                1.2122e-6
error ratio                           2.0008
maximum storage-scaled ledger defect  2.87e-17
```

The activity-scaled ledger metric reaches `6.14e-5` because the manufactured
equilibrium changes are extremely small compared with the stored orbital
quantities. The signed absolute defects remain at floating-point storage
roundoff.

## Source-free mesh preflight

The same analytic pressure-supported reference is sampled independently on
three meshes and advanced for four explicit steps over `0.17239 s`:

| `N` | Maximum `|v_R|/c` | Interior `|v_R|/c` | Maximum-drift cell |
|---:|---:|---:|---:|
| 16 | `3.0229e-10` | `1.5145e-11` | 15 |
| 32 | `2.7856e-10` | `4.8976e-12` | 31 |
| 64 | `2.6741e-10` | `1.3943e-12` | 63 |

The interior drift converges by factors `3.09` and `3.51`. The maximum is
always the final cell because the open boundary is not an equilibrium boundary
for this manufactured rotating profile. That edge response remains below
`3.1e-10 c` and decreases under refinement. The evidence therefore separates
convergent interior balance from a bounded open-edge response instead of
claiming that an open domain preserves a closed equilibrium.

## Common-stress flux pair

The shared total-pressure alpha stress now supplies

\[
G=2\pi R^2 W_\alpha.
\]

Because this module orients every face flux outward, the stress contribution is

\[
F_J^{\rm stress}=+G,
\qquad
F_E^{\rm stress}=+\Omega G.
\]

This is the sign-reversed representation of the established inward-positive
fluxes `Mdot*l-G` and `Mdot*B-Omega*G`. At `alpha=0.1`, the canonical audit
finds zero mass and radial-momentum changes, zero normalized mismatch in both
the angular and torque-work pairs, and exact optional zero-torque boundary
faces. No separate viscous-heating source is added, so torque work is not
double counted.

The constitutive flux is certified here, but a physical stress-bearing update
must be implicit or IMEX because the transport is diffusive. It is not yet
included in the acoustic explicit timestep certification.

## Stress-bearing time integration

An angular-plus-energy backward-Euler substep at `N=8` has first-order
one/two/four-step convergence:

```text
error ratio                           2.1010
maximum nonlinear residual            5.47e-15
maximum storage-scaled ledger defect  1.18e-16
maximum nonlinear evaluations         5
```

However, composing that substep after an explicit inviscid update is not a
robust production method. The split pilot accepts two steps and then rejects
the third before returning a nonphysical thermal state. The rejected step
returns the preceding state unchanged.

The accepted architecture instead evaluates inviscid transport, radial
sources, common stress, and torque work together at the new time level:

| `N` | Steps | `dt` (s) | Max residual | Max storage ledger | Min `T` (K) | Max `|v_R|/c` |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4/4 | `0.6450` | `5.24e-9` | `3.51e-9` | `6.7849e5` | `4.74e-7` |
| 8 | 8/8 | `0.3225` | `2.51e-9` | `1.57e-9` | `6.7839e5` | `4.36e-7` |
| 16 | 8/8 | `0.3225` | `7.24e-10` | `5.99e-12` | `6.6095e5` | `1.01e-6` |

All runs cover the same `2.5800 s`. Comparing the two `N=8` runs gives maximum
fractional differences `2.17e-7` in surface density, `1.45e-4` in temperature,
and `2.37e-8` in rotation; the radial-velocity difference is `3.77e-8 c`.
The `N=16` temperature minimum differs from `N=8` by about `2.7%`, so this is a
supported two-mesh preflight rather than continuum certification.

The energy rows are scaled by thermal storage rather than the much larger
orbital binding energy. Under this physically stricter scaling, the
nearest-neighbor Jacobian pattern still has `352` nonzeros for the `32x32`
`N=8` system, but the twelve-color finite-difference solve fails the fixed gate
after 39 evaluations. Its returned state differs from the accepted dense root
by `2.36e-3` in temperature. Dense differencing therefore remains the
production method; the colored path is rejected pending analytic or
better-conditioned sparse derivatives.

Positive `Sigma`, `Omega`, and temperature are nonlinear coordinates, so this
result uses neither clipping nor post-solve projection. It is still a short
manufactured source-free preflight, not a physical minidisk evolution.

## Radiative cooling

The shared two-face diffusion cooling is inserted only as a negative
total-energy cell source. A local backward-Euler reference, holding the other
three conserved fields bitwise fixed, gives:

```text
one-vs-two temperature error             3.3622e-4
two-vs-four temperature error            1.7532e-4
error ratio                              1.9178
maximum storage-scaled ledger defect     7.51e-17
```

The full monolithic residual also distinguishes a cooled `10 s` step from its
adiabatic control. Both roots pass; the cooled root has maximum residual
`7.29e-10`, ledger defect `3.32e-15`, and positive temperature reductions in
every cell (`5.13e-5` to `1.08e-3 K`). These small changes are resolved because
the total-energy row is normalized by thermal, not orbital, storage.

## Exact stream moments

One compact-C2 cumulative source deposits a constant injected state

\[
(v_{R,s},l_s,E_s)
\]

through all four cell ledgers. Exact edge increments give the following active
source-cell counts on `N=16,32,64`: `6,10,19`. On every mesh, integrated mass,
radial momentum, angular momentum, and total energy agree with the prescribed
moments to `2.23e-16` or better. No center-sampled source renormalization is
used.

A manufactured absolute-source monolithic step also passes:

```text
dt                                      1.0 s
source mass rate                        1.2362e23 g s^-1
maximum normalized residual             2.11e-9
maximum storage-scaled ledger defect    1.14e-10
nonlinear evaluations                   29
```

This certifies the source machinery, not the physical Layer-1 stream state or
a long supplied open-boundary evolution.

## Physical open-control mapping

The accepted coupled `144/96` open state is mapped onto one global grid from
`5.2102` to `335 rg`. The source uses the existing physical circularization
control:

```text
Mdot_stream                 5 Mdot_Edd
source center               240 rg
compact-C2 log width        0.08
R_circ                      248.96693 rg
v_R,s                       0
l_s                         l_K(R_circ)
E_s                         circular-orbit energy at R_circ
alpha                       0.01
outer stress                zero torque
```

Zero source radial velocity is not a ballistic-impact claim. It is the unique
radial component consistent with the inherited circularized assimilation
state. A future ballistic Layer-1 export must replace the complete source
state, not add an independent heating factor.

At `N=16`, one full step and two half steps over `1e-9` loading times both pass.
Their inner and outer mass-flux differences are `3.02e-8` and `6.56e-11` times
the supply, respectively. Maximum `H/R` differs by `6.8e-14`.

An initial-state audit isolates the mesh sensitivity before any evolution. The
exact stream source changes no face mass flux, as required for a cell source:

| Global cells | Active source cells | Initial inner outward flux / supply | Initial outer outward flux / supply |
|---:|---:|---:|---:|
| 16 | 2 | `-0.121349` | `1.726138` |
| 24 | 2 | `-0.139629` | `0.658504` |
| 32 | 3 | `-0.153997` | `0.739347` |
| 48 | 3 | `-0.165054` | `0.897548` |

The boundary-flux dependence is therefore already present in the
primitive-interpolated initial state and its open-face reconstruction. It is
not caused by source quadrature, radiative evolution, timestep choice, or the
nonlinear backward-Euler solve.

The first mesh comparison does not pass. At the same evolved time:

| Global cells | Inner outward flux / supply | Outer outward flux / supply | Max `H/R` |
|---:|---:|---:|---:|
| 16 | `-0.120998` | `1.725951` | `0.140914` |
| 24 | `-0.139506` | `0.658362` | `0.141093` |
| 32 | `-0.153970` | `0.739236` | `0.141192` |

The negative inner outward flux is accretion. The outer overflow changes by
`1.068` times the supply from `N=16` to `N=24`, and by `0.0809` from `N=24`
to `N=32`, despite all roots closing below `1.6e-12`. Therefore the physical
open control is timestep supported but not mesh certified. The coarse flux
fractions must not be interpreted physically.

A strict finite-volume remap next integrates all four conserved densities over
each target annulus with eight-point Gauss-Legendre quadrature. This exposes a
real resolution requirement: `N=16,24,32,48` are rejected because the
cell-averaged orbital binding-energy variation overwhelms the much smaller
thermal budget in the final cell, so a positive-temperature primitive cannot
be recovered. The solver does not clip or project those states.

The first admissible conservative meshes are:

| Global cells | Inner outward flux / supply | Outer outward flux / supply |
|---:|---:|---:|
| 64 | `-0.167196` | `0.860856` |
| 96 | `-0.168676` | `0.832686` |
| Coupled target | `-0.168937` | `0.831063` |

Thus the `N=96` remap differs from the coupled target by `2.61e-4` of the
supply at the inner face and `1.62e-3` at the outer face.

One full `N=64` step and two half steps over `1e-9` loading times all pass:

```text
full-step maximum normalized residual       1.25e-12
half-step maximum normalized residual       5.04e-12
maximum storage-scaled ledger defect        2.83e-16
inner full/half flux difference / supply     9.37e-8
outer full/half flux difference / supply     9.45e-11
maximum H/R relative difference              1.49e-11
```

The physical preflight is now conservatively initialized and timestep
supported at `N=64`.

The rejected grouped forward-color Jacobian remains closed. Its replacement
evaluates each one-sided local column independently using the same `1e-6`
perturbation as the accepted dense solver. At the initial state, every dense
column is computed once and all off-pattern entries are audited before the
sparse local blocks are used. The `N=64` and `N=96` physical audits find zero
off-pattern contribution across 256 and 384 columns, respectively. Exact
factorization of the assembled matrix is retained at these sizes; iterative
LSMR did not meet the nonlinear residual gate.

Both sparse physical steps pass:

| Global cells | Max residual | Ledger defect | Inner flux / supply | Outer flux / supply | Max `H/R` |
|---:|---:|---:|---:|---:|---:|
| 64 | `1.25e-12` | `1.45e-16` | `-0.167232` | `0.860690` | `0.141002` |
| 96 | `9.85e-13` | `1.87e-16` | `-0.168731` | `0.832228` | `0.141107` |

The evolved-mesh flux gate fails. The outer flux changes by `0.02846` supply,
or `3.31%` relative to the `N=64` value, while the inner flux changes by only
`0.00150` supply and maximum `H/R` by `7.47e-4` relative. A mapping-only
`N=128` state gives outer flux `0.820541`, so the open-face value brackets the
coupled target rather than converging monotonically. Long evolution remains
blocked.

## Direct conserved-donor open face

One bounded WP1 correction reconstructs the complete cylindrical mass flux

\[
F_M=2\pi R\Sigma v_R
\]

from the final cell as a single donor quantity. The identical mass flux carries
donor radial velocity, specific angular momentum, and Bernoulli energy. The
pressure traction remains additive in radial momentum, and the open-control
outer torque remains exactly zero. The legacy product reconstruction is
retained only for comparison.

The donor radial, angular, and energy flux consistency defects are exactly zero
on `N=64,96,128`. Mapping-only outer fluxes become:

| Global cells | Legacy product | Conserved donor |
|---:|---:|---:|
| 64 | `0.860856` | `0.855889` |
| 96 | `0.832686` | `0.844674` |
| 128 | `0.820541` | `0.836854` |

The donor `N=96`/`N=128` mapping difference is `0.00782` supply and passes the
fixed `0.01` mapping gate. Both donor-mode tiny steps also solve:

| Global cells | Max residual | Ledger defect | Inner flux / supply | Outer flux / supply |
|---:|---:|---:|---:|---:|
| 64 | `1.74e-12` | `3.90e-16` | `-0.167232` | `0.855711` |
| 96 | `6.11e-13` | `1.91e-16` | `-0.168731` | `0.844214` |

The evolved outer difference falls from `0.02846` to `0.01150` supply, but it
still exceeds the fixed `0.01` gate. The inner difference (`0.00150` supply)
and maximum-`H/R` difference (`7.47e-4` relative) pass. No clipping,
projection, tolerance change, or target forcing is used.

WP1 is therefore a useful but narrowly failed reconstruction test. Per its stop
condition, no additional extrapolation formula will be scanned. The next
boundary work must define one physical characteristic/Roche-overflow contract.

## Characteristic and Roche-geometry audit

The radial Euler block has characteristic speeds

\[
v_R-c_{\rm eff},\quad v_R,\quad v_R,\quad v_R+c_{\rm eff}.
\]

At the outermost mapped cell the conservative meshes give:

| Global cells | `v_R` [cm/s] | `c_eff` [cm/s] | radial Mach | incoming characteristics |
|---:|---:|---:|---:|---:|
| 64 | `7.7755e4` | `8.6511e6` | `0.00899` | 1 |
| 96 | `7.0380e4` | `5.5164e6` | `0.01276` | 1 |
| 128 | `7.6250e4` | `7.6107e6` | `0.01002` | 1 |

The overflow is therefore extremely subsonic and requires exactly one
exterior boundary condition. A zero-gradient or donor rule does not supply
that physical datum; it only chooses a numerical extrapolation.

The geometry also rules out calling the present edge a Roche outlet. For the
checked-in fiducial binary, the live `335 rg` face is `0.44852 R_H` and
`0.89704` of the current fiducial `0.5 R_H` truncation estimate. It is not an
L1/L2 saddle. The repository contains no exterior pressure, entropy, or
Bernoulli state at this radius. A pressure outlet copied from one mesh would
force the target, while a vacuum state would introduce an uncalibrated
rarefaction.

The characteristic implementation is therefore intentionally stopped before
inventing an exterior state. The boundary contract can be closed in only one
of two declared ways:

1. Layer 1 provides one physical exterior thermodynamic invariant at the
   `335 rg` truncation surface.
2. The global domain is extended through a Hill/Roche overflow layer whose
   effective potential and escape/nozzle condition terminate at an actual
   saddle.

This is a physics-closure gate, not a solver failure. WP2 energy conditioning
must not be used to conceal it, and tide/wind evolution remains blocked.

## WP2 physical column-energy closure

WP2 retains the existing total-energy storage. The accepted enthalpy flux is
completed by radial and temporal one-zone column work:

\[
W_{R,i}=\dot M_i\frac{\Pi_i}{\Sigma_i}\Delta\ln H_i,
\]

\[
W_{t,i}=\frac{1}{2}
\left(M_i^{n+1}h_i^{n+1}+M_i^nh_i^n\right)
\ln\frac{H_i^{n+1}}{H_i^n}.
\]

The global module uses outward-positive face fluxes, so the radial-work helper
makes the sign conversion `Mdot_in=-F_M,out` explicitly. `W_R` enters only the
total-energy cell source. `W_t` enters only the backward-Euler energy storage
row and its independent telescoped ledger. Torque work remains exclusively in
the paired `Omega G` face flux; no local viscous-heating term is added.

The global radial-work operator matches the previously certified signed
finite-volume operator to roundoff. The temporal operator exactly integrates a
manufactured path linear in `M h` versus `ln H`. A source-bearing four-ledger
backward-Euler state closes below `3e-16` relative, and enabling column work
leaves all mass, radial, angular, and torque-work face fluxes bitwise unchanged.

The physical donor-face tiny-step results are:

| Global cells | Column work / `L_Edd` | Max residual | Ledger defect | Inner flux / supply | Outer flux / supply |
|---:|---:|---:|---:|---:|---:|
| 64 | `0.0399076` | `2.66e-12` | `2.00e-16` | `-0.167213` | `0.852768` |
| 96 | `0.0398574` | `9.25e-12` | `2.80e-16` | `-0.168735` | `0.846416` |

The final figures include the subsequent WP4 reference correction. WP2
corrects the physical identity without tuning either radial boundary.

## WP3 inner characteristic absorber

The current global edge is fixed at `5.21024 rg`, close to the stationary sonic
node but not inside a causally outgoing plunge for the time-dependent Euler
system. The mapped first-cell audit gives:

| Global cells | Radial Mach | `v_R+c_eff` [cm/s] | Incoming characteristics |
|---:|---:|---:|---:|
| 64 | `-0.6542` | `2.4310e8` | 1 |
| 96 | `-0.7630` | `1.5817e8` | 1 |
| 128 | `-0.8179` | `1.1860e8` | 1 |

The old diode only blocked advective mass inflow. It did not close this incoming
acoustic mode. WP3 now decomposes the inner-cell perturbation relative to the
fixed certified transonic reference as

\[
w_+=\delta v_R+\frac{\delta\Pi}{\Sigma_{\rm ref}c_{\rm ref}},\qquad
w_-=\delta v_R-\frac{\delta\Pi}{\Sigma_{\rm ref}c_{\rm ref}}.
\]

At the boundary, `w_+` is set to zero and `w_-` is preserved. Surface density
and angular velocity are inherited from the interior. Temperature is recovered
from the projected positive integrated pressure without clipping. The same
projected state supplies all four inviscid face fluxes; alpha torque and paired
work are added afterward exactly as before.

The certified reference flux is bitwise unchanged. A manufactured outgoing
acoustic perturbation changes by less than `2e-8` relative and generates no
incoming reflected mode at that gate. Arbitrary perturbations preserve radial,
angular, and Bernoulli flux consistency while annihilating the incoming linear
invariant.

The physical results are:

| Global cells | Incoming before [cm/s] | Incoming after [cm/s] | Max residual | Ledger defect | Inner flux / supply |
|---:|---:|---:|---:|---:|---:|
| 64 | `7.6833e4` | `3.61e-3` | `2.64e-12` | `2.00e-16` | `-0.167228` |
| 96 | `1.1332e5` | `4.91e-3` | `9.20e-12` | `2.63e-16` | `-0.168755` |

The absorber changes the inner flux by only `2.23e-5` and `2.36e-5` supply
relative to the matching WP2 runs. The outer flux is unchanged by WP3. WP3 is accepted as a reference-state
absorbing boundary for the preflight; a future supersonic inner extension may
replace it but is not required before WP4 conditioning work.

## WP4 finite-volume mechanical-energy conditioning

The original conservative remap averaged total energy over each annulus but
subtracted mechanical energy evaluated only at the cell center. Near the inner
edge, the difference between those two mechanical quantities exceeded the
thermal energy and caused false negative-temperature recovery on `N<=48`.

WP4 retains the physical total-energy storage and introduces one fixed
well-balanced correction for each reference cell:

\[
\delta e_{{\rm mech},i}
=\left<\Phi+\frac{v_R^2}{2}+\frac{v_\phi^2}{2}\right>_{M,i}
-\left(\Phi+\frac{v_R^2}{2}+\frac{v_\phi^2}{2}\right)_i.
\]

The mass-weighted average is calculated in the same annular quadrature as the
four conserved variables. Recovery subtracts the center value plus this fixed
correction; reconstruction adds it back. It is never clipped or adjusted by
the nonlinear solve.

All conservative mappings now recover positive internal energy:

| Global cells | Minimum specific internal energy [erg/g] | Maximum `|delta e_mech|` [erg/g] |
|---:|---:|---:|
| 16 | `3.273e15` | `4.817e17` |
| 24 | `1.645e15` | `2.323e17` |
| 32 | `1.020e15` | `1.318e17` |
| 48 | `5.345e14` | `5.731e16` |
| 64 | `3.765e14` | `3.343e16` |
| 96 | `1.895e14` | `1.492e16` |
| 128 | `1.549e14` | `8.405e15` |

The correction decreases monotonically over `N=64,96,128`. Comparing the
production 32-point remap with 64-point quadrature gives maximum relative
differences below `1.2e-3` in every conserved field, below `7e-4` in the
correction, and below `1.9e-4` in recovered temperature.

With the WP2 and WP3 closures active, the selected evolved comparison is:

| Global cells | Max residual | Ledger defect | Inner flux / supply | Outer flux / supply | Minimum `T` [K] |
|---:|---:|---:|---:|---:|---:|
| 64 | `2.64e-12` | `2.00e-16` | `-0.167228` | `0.852768` | `9.779e5` |
| 96 | `9.20e-12` | `2.63e-16` | `-0.168755` | `0.846416` | `7.357e5` |

The outer-flux difference is `0.006353` supply and passes the fixed `0.01`
numerical mesh gate. This closes the cell-average energy-conditioning problem.
It does not provide the still-missing exterior thermodynamic invariant at the
subsonic truncation boundary.

## Interpretation

The one-domain architecture passes its first structural gate. This removes the
specific boundary-rank problem of the hybrid model, but it does not yet show
that a physical transonic disk can be evolved.

The smooth radial pair, bounded shock-flux preflight, source-free temporal and
two-mesh gates, common-stress identity, and monolithic implicit preflight pass.
The split IMEX and colored-Jacobian pilots are rejected. Dense monolithic
transport with common stress, cooling, and exact source moments passes its
current gates. The physical open mapping passes timestep refinement but fails
both coarse pointwise mesh comparisons in the outer flux. Conservative remap
removes that ambiguity, rejects under-resolved meshes, and approaches the
coupled boundary fluxes at `N=64,96`; the `N=64` monolithic step also passes its
temporal gate. The derivative-certified sparse path passes at both physical
meshes, but the bounded evolved comparison fails only in outer mass flux. The
direct conserved-donor correction improves but does not pass that gate. The
reconstruction investigation is closed. The characteristic count is now
certified, but the contract is physically underdetermined because the current
subsonic truncation edge is not the Roche saddle and has no declared exterior
state. The next boundary work must supply that state or a modeled Hill/Roche
overflow layer. The physical column-energy identity now passes WP2;
the reference-state inner absorber now passes WP3, and finite-volume mechanical
energy conditioning passes WP4. Longer evolution, tide, and wind remain
disabled because the exterior-state physics gate is still open.

## WP0 stored/physical energy cross-consistency

The joint WP3/WP4 review found that the fixed cell mechanical quadrature
offset was included in the current characteristic donor and conserved outer
donor, omitted from the projected characteristic state, and already omitted
from smooth physical face reconstruction. That mixed convention would produce
a finite energy-flux jump as a nonzero incoming amplitude tended to zero.

The corrected contract is:

```text
stored cell-average energy: includes delta e_mech
physical center energy:     stored energy minus delta e_mech
physical face Bernoulli:    excludes delta e_mech
Rusanov dissipative state:  retains stored conservative energy
```

The Rusanov physical flux, conserved outer donor, and both current and
projected characteristic fluxes now use physical Bernoulli energy. The
projected stored state includes the same offset before physical conversion. A
nonzero-offset amplitude sequence is continuous at zero.

The physical-flux Jacobian audit uses scaled center-conserved variables and
does not differentiate the mesh quadrature offset as continuum physics. In the
manufactured radiation-pressure state, the maximum analytic eigenvalue defect
is `2.40e-5 c_eff`, finite-difference refinement differs by `5.45e-7 c_eff`,
and the incoming acoustic left-vector alignment is `1.0`.

The mechanical-reference checkpoint stores the offset array, grid, schema,
generating-state SHA-256, offset SHA-256, and JSON provenance. Incompatible or
silently regenerated references are rejected.

Layer 1 was also audited for the outer-boundary decision. It has no ambient
pressure, entropy, temperature, or Bernoulli/Jacobi invariant at `335 rg`.
ADR 0014 therefore selects one adiabatic Hill/Roche overflow side channel
ending at a real `L1/L2` saddle. That nozzle remains unimplemented, so no long
evolution, distributed tide, or wind is authorized yet.
