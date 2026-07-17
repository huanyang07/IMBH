# Model Equations and Conventions

This document records the conventions needed to interpret the current
transonic/phase-DAE implementation. The production source remains under
`src/imri_qpe/layer3_minidisk_1d/`.

## Coordinates and Signs

- Radius increases outward; the ordinary independent variable is `x=ln R`.
- `Mdot>0` denotes inward mass flux.
- The radial inflow speed is represented by positive `u=-v_R` where used.
- Source and wind rates per `dlnR` obey

```text
dMdot/dlnR = Mdot_wind' - Mdot_stream'.
```

The current conservative coordinate is

```text
F = Mdot / Mdot_inner.
```

## Vertically Integrated State

The solver combines pseudo-Newtonian orbital dynamics, vertical hydrostatic
balance, alpha stress, optically thick radiative cooling, entropy advection,
stream source terms, and energy-limited wind terms. Algebraic vertical state
variables are evaluated from the radial unknowns rather than evolved as an
independent vertical grid.

The energy balance is organized as

```text
Qvisc + Qstream = Qrad + Qadv + Qwind.
```

`Qadv` is computed from the radial entropy derivative, not from an imposed
constant `xi`:

```text
T ds/dR = de/dR - P/rho^2 d rho/dR
Qadv    = Sigma v_R T ds/dR.
```

The unified conservative wind ledger uses launch power as the conditioned
quantity:

```text
P_wind'    = 2 pi R^2 Qwind
Mdot_wind' = P_wind' / E_launch
B_wind     = B_disk + Omega (l_w-l) + E_launch
```

The wind energy sink is assembled as

```text
Mdot_wind' [B_disk + Omega (l_w-l)] + P_wind',
```

which is algebraically identical to `Mdot_wind' B_wind`. A terminal-Bernoulli
audit compares `E_launch` with the energy required for `B_wind>=0`; it does not
alter the eta regression mode.

The physical alternative prescribes a terminal Bernoulli energy:

```text
E_launch = B_infinity - B_disk - Omega (l_w-l).
```

The state is invalid for this energy-limited closure if `E_launch<=0`; the code
does not hide that singular limit with an energy floor. An optional explicit
mass-availability constraint bounds `dMdot_wind/dlnR` relative to local
throughput and reduces launch power consistently when active.

## Finite-Volume Mass Equation

For interval `i`, the conservative mass row is

```text
F[i+1] - F[i]
  - integral_i (Mdot_wind' - Mdot_stream') / Mdot_inner dlnR = 0.
```

Source-band production uses compatible finite-volume mass and phase/DAE
dynamics. Old pointwise or midpoint mass rows are audits only where the
conservative row is active.

## Angular-Momentum Ledger

Define the inward angular flux

```text
J = Mdot*l - G,
```

where `G` is the viscous torque. The explicit conservative ledger is

```text
dJ/dlnR = Mdot_wind' l_w - Mdot_stream' l_s + tau_ext.
```

The current `representation` closure uses `J/Mdot` as the carried specific
angular momentum and closes algebraically. This is a mathematical identity,
not a physical prescription. A physical production model must independently
define `l_s(R)`, `l_w(R)`, and `tau_ext(R)`.

## Phase-Space DAE Segment

Near the stiff source transition, `ln R` ceases to be a numerically suitable
polynomial coordinate. The local phase representation uses an intrinsic
parameter `s` with

```text
z(s) = (logu, logT, F, logR)
p    = dz/ds.
```

Homogeneous radial and energy equations remain finite as `p_R=dlogR/ds`
approaches zero. Physical `d/dlnR` derivatives are reconstructed only where
division by `p_R` is conditioned.

Interfaces match state and conserved fluxes. Derivative continuity in `ln R`
is not imposed across a phase interface.

## Independent-Sigma Signed-Flux Bridge

The new outer/reservoir core stores `Sigma` independently and permits signed
mass flux. Under its initial nearly Keplerian, prescribed-viscosity closure,

```text
G             = -2 pi R^3 nu Sigma dOmega_K/dR
dM_cell/dt    = Mdot_outer - Mdot_inner + S_M
J_face        = Mdot l_K - G.
```

`Mdot>0` remains inward and `Mdot<0` is decretion. A tidal wall sets outer mass
flux to zero while retaining a finite torque. An open zero-torque edge permits
outward overflow. The implementation never divides by `Mdot` or radial
velocity, so a finite-density stagnation point is regular.

The accepted steady source-bearing model uses one cell-integrated source state
`(S_M,S_J,S_E)` and solves both conservation laws:

```text
Mdot[i+1] - Mdot[i] + S_M[i] = 0
J[i+1] - J[i] + S_J[i] + T_ext[i] = 0.
```

An open boundary sets `G=0`. An ideal tidal wall sets `Mdot=0` and returns its
required torque. The older `Mdot=dG/dl_K` source solve is retained only to
reproduce commit `53566fa`; it is not an accepted physical stream closure.
Time evolution with nonlocal source angular momentum remains disabled until
the coupled angular IMEX operator is implemented.

The first thermal extension evolves annular internal energy:

```text
E_th          = Sigma e A
F_e           = Mdot e_donor
dE_th/dt      = F_e,out-F_e,in + Qvisc A + S_M(B_s-E_orb) - Qrad A
nu            = alpha H(Sigma,T)^2 Omega_K.
```

Cooling is implicit and the steady mass, thermal, and viscosity equations are
iterated to a fixed point. This is not yet the final total-energy equation:
enthalpy/pressure work and inner transonic flux matching remain required.

The reported transport ratio is named `internal_energy_export_fraction`, and
the roundoff telescoping check is `internal_energy_ledger_defect`. Neither is a
total-energy or entropy-advection certification.

The WP2 total-energy extension instead defines

```text
B_col = Phi + v_phi^2/2 + v_R^2/2 + e + Pi/Sigma
F_E   = Mdot B_col - Omega G.
```

Its cell compatibility equation is

```text
Delta F_E + W_H - L_rad + S_E + P_ext = 0
W_H = Mdot (Pi Delta Sigma/Sigma^2 - P Delta rho/rho^2).
```

Using `Pi/Sigma=P/rho` and `Sigma=2 rho H`, the differential work is also
`W_H=Mdot(P/rho)dlnH`. The older
`Mdot(dPi/Sigma-P drho/rho^2)` expression belongs to an internal-energy flux;
pairing it with enthalpy adds an extra `Mdot d(Pi/Sigma)` term.
`-Omega G` carries viscous torque work, so `Qvisc` is not added independently.
`P_ext` is signed power applied to the disk, so a torque applied by a pattern
has `P_ext=Omega_pattern T_disk`. Distributed external torque and power are
named separately.

For the finite-minidisk wall control, the angular boundary applies
`T_disk=-G_out`. The conservative face flux initially carries disk-rate work
`-Omega_out G_out`. A continuation fraction `f_tide` converts this to the
binary pattern rate by depositing the differential work over a normalized
Hill-band kernel:

```text
P_heat = f_tide (Omega_out-Omega_pattern) G_out
P_wall = -Omega_out G_out + P_heat
       = -[(1-f_tide)Omega_out+f_tide Omega_pattern] G_out.
```

The kernel is zero below `0.35 R_H` and uses exact cell integrals. This pairing
closes the torque/power identity, but it does not make a zero-mass-flux wall
physical when the tidal band becomes geometrically thick. The production
coupled reservoir ends at `335 rg`; the superseded `10000 rg` numerical buffer
must not be interpreted as a Hill-truncated disk.

The open-overflow extension promotes the positive inner rate to an eigenvalue
while retaining signed outer flux. Candidate face ledgers are integrated from
the inner interface without imposing an outer condition:

```text
Mdot_out = Mdot_inner - sum(S_M)
J_out    = J_inner - sum(S_J + T_ext)
G_out    = Mdot_out l_out - J_out.
```

One added scalar row continues the mass wall to the open edge,

```text
R_edge(chi) = (1-chi) Mdot_out/Mdot_stream
            + chi G_out/G_scale.
```

The augmented system has one additional `log(Mdot_inner/Mdot_stream)` unknown
and remains square. At `chi=1`, outward overflow is permitted and `G_out=0`.
No outer inflow state is configured.

## Conserved Inner Interface

The shared inner/outer interface object carries inward-positive

```text
(Mdot, J, F_E)
J   = Mdot l - G
F_E = Mdot B - Omega G.
```

Both transonic and signed-reservoir extractors use these definitions. A
prescribed signed-reservoir boundary consumes `Mdot` and `J` in the steady
mass/angular solve and `F_E` in the total-energy row. A tidal-wall outer edge
retains exact zero outer mass flux and treats the prescribed inner `Mdot` as a
compatibility gate; an open edge is integrated outward from the prescribed
inner flux. Incompatible simultaneous boundary conditions are rejected.

## Experimental Pressure-Supported Reservoir

The diagnostic non-Keplerian extension evaluates

```text
Omega_force^2 = Omega_K^2 + (1/(R Sigma)) dPi/dR
l             = R^2 Omega
G             = -2 pi R^3 nu Sigma dOmega/dR.
```

The same `Omega` and `l` are used in angular transport, orbital energy,
`-Omega G` torque work, and `nu=alpha H^2 Omega`. Derivatives of `Pi` are
regularized over a stated log-radius width. Trial log-rotation slopes are
projected to decreasing, Rayleigh-stable profiles before the viscous solve:

```text
0.2 <= dln(l)/dln(R)
dln(Omega)/dln(R) < 0.
```

The projection mismatch from the unregularized radial-force equation is
reported explicitly. This staggered closure is diagnostic only: full-pressure
roots are coarse-grid-only and do not remove primitive-state discontinuity.
The production successor must solve `Sigma`, `T`, and `Omega` simultaneously.

## Common-Stress Simultaneous Reservoir

The successor uses the same vertically integrated alpha stress as the inner
transonic solver:

```text
W_alpha = integrated_stress(Sigma,T,alpha,mu_stress,stress_factor)
G_alpha = 2 pi R^2 W_alpha.
```

For the current benchmark, `mu_stress=0` and `stress_factor=1`, so
`W_alpha=alpha Pi`. The older steady reservoir relation
`nu=alpha H^2 Omega` is not an additional constitutive equation. An effective
viscosity may be reconstructed diagnostically after a root is found.

The simultaneous cell state is `(log Sigma,log T,log Omega)`. Exact source and
boundary integration first fixes `Mdot_face` and `J_face`; each trial rotation
then gives

```text
l             = R^2 Omega
G_required    = Mdot l - J
G_alpha       = G_required.
```

The radial equation is the same sign convention as the transonic solver:

```text
(1/2) d(v_R^2)/dlnR
  - R^2 (Omega^2-Omega_K^2)
  + (1/Sigma) dPi/dlnR = 0.
```

Writing the inertia as `d(v_R^2)/2` keeps the equation finite where signed
mass transport approaches zero. The third block is the corrected total-energy
equation above, evaluated with the same trial `Omega`, `l`, and
`G_required`. There is no projected rotation, pressure smoothing, or separate
`Qvisc` source.

Continuation replaces the radial block temporarily with

```text
(1-lambda) ln(Omega/Omega_K) + lambda R_radial = 0,
```

and advances from `lambda=0` to `lambda=1`. Only the full-support root is a
candidate physical reservoir state.

## Fully Coupled Inner/Outer Control

At a fixed physical interface edge `R_I`, the coupled no-wind control uses

```text
inner unknowns: log u, log T, log R_son, lambda0
outer unknowns: log Sigma, log T, log Omega
global unknowns: signed J_I, signed F_E,I.
```

The inner residual retains all interval equations and a frozen, independent
two-row sonic pair, but removes its two artificial outer-boundary rows. The
outer residual retains the simultaneous common-stress, radial-momentum, and
total-energy rows. Explicit extraction rows impose

```text
J_I   = (Mdot l - G)_inner
F_E,I = (Mdot B - Omega G)_inner.
```

The two remaining boundary freedoms are closed by continuity of `log Sigma`
and `log T` at the same physical edge. The outer cell-centered state is
positive-log reconstructed to that edge; no half-cell-separated comparison
is used. Pressure, rotation, scale height, and radial velocity are dependent
audits rather than extra boundary equations.

The coupling homotopy drives the known one-way primitive jump `C_ref` to zero:

```text
R_I(mu) = C(current) - (1-mu) C_ref,   0 <= mu <= 1.
```

This keeps the derivative of the physical continuity conditions active at
every stage. A direct blend between a canonical inner anchor and continuity
is not used because their inner-state derivatives have opposite signs and can
artificially lose rank.

Mesh continuation interpolates the complete accepted root in positive
logarithmic variables and carries the signed `J_I,F_E,I` values unchanged.
When `R_I` moves, the composite itself supplies the newly reassigned annulus:
inner primitives seed newly exposed outer cells for inward moves, while outer
primitives seed newly added inner nodes for outward moves.

Interface-position invariance is evaluated on fixed physical quantities. In
particular, a maximum over the whole outer numerical domain is not invariant
when its inner edge moves. The thickness gate therefore uses the maximum
`H/R` on the common `R>=60 rg` band and fixed-radius samples; the moving-domain
maximum is retained separately as a diagnostic.

## Global Signed Conservative Evolution

The one-domain finite-volume state is cell-integrated

```text
(M, P_R, J, E).
```

It permits either sign of radial velocity and does not divide by radial mass
flux. The finite-volume remap stores the true annular total-energy integral.
When a fixed cell mechanical quadrature offset is active, the energy contract
is:

```text
epsilon_stored  = epsilon_physical_center + delta_e_mech
B_face          = epsilon_physical_face + Pi/Sigma
```

`delta_e_mech` reconciles a cell average with center primitives. It is not
exported as physical Bernoulli energy. Smooth, Rusanov-physical,
conserved-donor, and characteristic face fluxes all use physical energy. The
Rusanov dissipative jump continues to act on the stored conservative field.
The fixed offset, mesh, hashes, and generating-reference provenance are part
of restart state.

The local characteristic audit differentiates the vertically integrated
physical flux with respect to physical center-conserved variables. The mesh
quadrature offset is not treated as a continuum thermodynamic variable.

### Causally outgoing inner plunge

The production global preflight no longer terminates at the subsonic side of
the inner sonic transition. The accepted stationary transonic profile is
continued from its regular sonic node at `5.210237 rg` to `4.5 rg` with the
same local equations,

```text
d(ln u, ln T)/d ln R = f(R,u,T; lambda0),
```

and the same PW potential, common alpha stress, vertical closure, angular
eigenvalue, and energy convention. The regular derivative branch is selected
by its agreement with the first resolved outer transonic interval. No
ballistic or free-fall closure is inserted.

At the finite-volume inner face, the characteristic speeds are

```text
v_R-c_eff, v_R, v_R, v_R+c_eff.
```

The face is admissible as a pure outflow boundary only when all four are
negative. The current `4.5 rg` stationary face has radial acoustic Mach number
`-9.45`; the mapped N64/N96/N128 first cells also have zero incoming modes.
The numerical boundary therefore uses the one-sided physical flux from the
first cell. It supplies no exterior invariant and applies no characteristic
projection. The incoming-mode count remains an acceptance diagnostic during
every evolution run.

#### Causal inner thermodynamic prototype

The Newtonian gas+radiation acoustic derivative is

```text
c_s,N^2 = (dP/d rho)_s.
```

For total energy density including rest mass,

```text
epsilon_total = rho (c^2 + e),
```

the first law along an adiabat gives

```text
d epsilon_total/d rho = c^2 + e + P/rho.
```

The local relativistic acoustic speed used by the WP10a diagnostic is

```text
a^2 = c^2 c_s,N^2 / (c^2 + e + P/rho).
```

It recovers the Newtonian cold limit and approaches `c^2/3` in the
radiation-dominated relativistic limit. It is not a cap applied to the
Newtonian sound speed.

In a one-dimensional local outward-oriented special-relativistic frame, the
diagnostic radial speeds are

```text
(v_R-a)/(1-v_R a/c^2), v_R, v_R, (v_R+a)/(1+v_R a/c^2).
```

This characteristic audit is not yet the production global flux. The current
stationary plunge and finite-volume evolution do not use one common
relativistic conservative system, and the prototype omits relativistic
transverse-rotation and spacetime lapse/shift effects. The first zero-incoming
point on the WP9 low-rate profile occurs only at `2.0001 rg`, too close to the
Paczynski-Wiita singularity to adopt as a production excision without that
complete system. The old profile also has `v_phi/c=1.71` by `3 rg` and
`357` at the radial-only crossing, so it supplies no subluminal full-state
candidate there.

#### Horizon-penetrating Valencia core

ADR 0017 selects an ingoing-Kerr-Schild Schwarzschild Valencia column system
for the next one-domain architecture. With

```text
H = 2 rg/R,
```

the equatorial 3+1 metric is

```text
alpha        = (1+H)^(-1/2)
beta^R       = H/(1+H)
gamma_RR     = 1+H
gamma_phiphi = R^2.
```

The coordinate radial light speeds are

```text
lambda_light,- = -1
lambda_light,+ = (1-H)/(1+H).
```

The outgoing cone is tangent to the horizon at `2 rg` and points toward
decreasing radius inside it.

For Eulerian velocity `v^i`, Lorentz factor

```text
W = (1-gamma_ij v^i v^j)^(-1/2),
```

and dimensionless column enthalpy

```text
h = 1 + (e + Pi/Sigma)/c^2,
```

the mass-equivalent conserved variables are

```text
D     = Sigma W
S_i   = Sigma h W^2 v_i
tau   = Sigma h W^2 - Pi/c^2 - D.
```

Momentum is divided by `c`, energy by `c^2`, and the covariant azimuthal
component `S_phi` is the angular-momentum density divided by `c`. Define

```text
q^R = alpha v^R - beta^R.
```

The radial coordinate flux divided by `c` is

```text
F_D     = D q^R
F_SR    = S_R q^R + alpha Pi/c^2
F_Sphi  = S_phi q^R
F_tau   = tau q^R + alpha (Pi/c^2) v^R.
```

The proper vertically integrated radial Jacobian is

```text
J_column = 2 pi R sqrt(gamma_RR).
```

For dimensionless sound speed `a` and

```text
v^2 = gamma_ij v^i v^j,
```

the two advected radial speeds are

```text
lambda_0 = alpha v^R - beta^R,
```

and the acoustic speeds are

```text
lambda_+/- = alpha/(1-v^2 a^2) [
    v^R(1-a^2)
    +/- a sqrt((1-v^2) [
        gamma^RR(1-v^2 a^2) - (v^R)^2(1-a^2)
    ])
] - beta^R.
```

This expression includes transverse rotation. The WP10b analytic speeds
match the numerical conservative-flux Jacobian below `9.8e-11`. Inside the
horizon every sampled physical fluid characteristic is negative.

The flux-primary DAE count is

```text
conserved cells + primitive cells + all face fluxes
= 4N + 4N + 4(N+1)
= 12N + 4 unknowns and rows.
```

The inner edge has four one-sided flux-definition rows and zero exterior
physical boundary rows. A stationary state is a root of this same
finite-volume operator. At an acoustic critical point its stationary flux
matrix loses one rank; no separately defined PW sonic condition is added.

The current implementation remains a local flux/rank prototype.
Gas+radiation primitive recovery, source-free geometry, and a causal
relativistic stress prototype have now been migrated. Cooling, dynamic
vertical structure, stream injection, and the Hill/Roche provider remain
pending.

#### Valencia gas+radiation primitive recovery

ADR 0018 certifies the local gas+radiation `P -> U -> P` map without importing
the old Paczynski-Wiita vertical gravity. For this thermodynamic gate, hold one
proper half-height `H` fixed:

```text
rho = Sigma/(2H)
P   = rho R_g T + a T^4/3
Pi  = 2H P
e   = R_g T/(gamma_g-1) + a T^4/rho.
```

The fixed height is a local EOS chart parameter, not the production vertical
equilibrium.

Given conserved variables and a trial pressure mass `p=Pi/c^2`, define

```text
S^2 = gamma^RR S_R^2 + S_phi^2/gamma_phiphi
Q   = tau + D + p
W   = Q/sqrt(Q^2-S^2)
Sigma = D/W.
```

The stable thermal inversion uses

```text
e/c^2 = [tau - D(W-1) - p(W^2-1)]/(D W).
```

After the monotone EOS inversion for `T`, solve

```text
p - Pi_EOS(Sigma,T)/c^2 = 0
```

in `log p`. The Eulerian velocities are

```text
v_hat_R/c   = S_R/[Q sqrt(gamma_RR)]
v_hat_phi/c = S_phi/(Q R).
```

The forward energy uses the equivalent stable expression

```text
tau = D[(W-1) + (e/c^2 + p/Sigma)W] - p.
```

The nine-state WP10c1 matrix spans `20`, `4.5`, and `1.8 rg`, includes
rotation, and crosses gas-dominated through radiation-dominated columns. Its
maximum primitive, conserved, and analytic/numerical characteristic defects
are respectively `7.42e-11`, `6.46e-15`, and `1.94e-8`. All audited
inside-horizon states retain zero incoming modes.

#### Kerr-Schild geometric finite volume

ADR 0019 selects an equatorial `2+1` source-free column in spacetime
coordinates `(ct,R,phi)`. With `H=2rg/R`, the nonzero metric terms are

```text
g_tt     = H - 1
g_tR     = H
g_RR     = 1 + H
g_phiphi = R^2.
```

The exact proper column face measure is

```text
A(R) = 2 pi R sqrt(1+2rg/R).
```

Cell measures use differences of

```text
V(R) = pi [
    (R+rg) sqrt(R(R+2rg))
    - rg^2 arcosh((R+rg)/rg)
].
```

The local primitive map remains in `(D,S_R,S_phi,tau)`, but the stationary
finite-volume chart uses

```text
U_K = (D,S_R,S_phi,E_K),
E_K = alpha(tau+D) - beta^R S_R.
```

Its inverse is

```text
tau = (E_K + beta^R S_R)/alpha - D.
```

The radial Killing-energy flux divided by `c` is

```text
F_EK/c = alpha(F_tau/c + F_D/c) - beta^R F_SR/c.
```

For the stationary axisymmetric metric,

```text
d[A F_K/c]/dR = A S_K,
```

with

```text
S_K = (
    0,
    alpha T^munu d_R g_munu/2,
    0,
    0
).
```

Mass, covariant angular momentum, and Killing energy therefore telescope
without geometric sources. The radial momentum source is independently
checked against the equivalent `3+1` lapse, shift, and spatial-metric form.
The local `tau` source is also checked both from extrinsic curvature and from
the differentiated Killing transform.

The vertical direction is already integrated into the column primitives.
The `2+1` radial source retains cylindrical azimuthal curvature but does not
add a separate `theta` source. This avoids double counting vertical geometry
before WP10c3b declares the relativistic vertical-work contract.

The WP10c2 source identities close below `4.85e-15`. Flat cylindrical
constant pressure balances at `9.54e-16`; circular dust orbits have zero
radial source; and marginally bound radial dust free fall from `20` through
`1.5 rg` converges at second order while preserving mass and Killing-energy
fluxes below `1.9e-15`.

#### Causal relativistic alpha shear

WP10c3a represents viscous transport by one off-diagonal stress in the local
fluid rest frame:

```text
t^munu = S (e_R^mu e_phi^nu + e_phi^mu e_R^nu),
S      = Sigma chi.
```

The tetrad vectors are orthonormal and perpendicular to the four-velocity, so

```text
t^mu_mu = 0,
t^munu u_nu = 0.
```

The common alpha prescription fixes the equilibrium stress amplitude at one
reference positive shear rate:

```text
chi_alpha = alpha Pi/(Sigma c^2),
nu_s      = chi_alpha/q_ref.
```

It is not inserted as an instantaneous algebraic stress. The resolved
Maxwell-Cattaneo law is

```text
tau_r u^mu nabla_mu chi + chi = nu_s q,
```

where `q` is the positive rest-frame `R-phi` shear rate. The relaxation time
and finite viscous signal speed obey

```text
c_nu^2/c^2 = nu_s/(tau_r h).
```

The bounded gate chooses

```text
c_nu = sqrt(alpha) a
```

and calibrates `tau_r` from this identity. The local transverse principal
matrix in variables `(delta v_phi/c, delta chi)` is

```text
[ 0              1/h       ]
[ h c_nu^2/c^2   0         ],
```

with real rest-frame modes `+/-c_nu`. In the declared frozen-coefficient
principal model, the coordinate chart has five modes: two acoustic, one
material/contact, and two shear. The shear modes use the same covariant
Valencia cone formula as the acoustic pair with `a` replaced by `c_nu`.

The stress contribution to the Killing chart is evaluated directly from the
same tensor:

```text
delta S_i  = alpha t^0_i,
delta E_K  = -alpha t^0_t,
delta F_i  = alpha t^R_i,
delta F_EK = -alpha t^R_t.
```

The associated outward torque and Killing power are

```text
G_stress = A c^2 delta F_Sphi/c,
P_stress = A c^3 delta F_EK/c.
```

For a stationary circular flow,

```text
P_stress = Omega G_stress
```

to roundoff. In the weak-field limit the torque approaches

```text
G_stress -> 2 pi R^2 alpha Pi.
```

A rejected control advects an independent `D chi` and relaxes it toward
`alpha Pi` without including the shear-gradient principal coupling. Its cold
weak-field flux Jacobian has a finite-difference-stable complex pair. This
demonstrates that the old instantaneous pressure amplitude is an equilibrium
calibration only, not by itself a causal evolution law.

#### Responsive height, cooling, and vertical work

WP10c3b replaces the fixed-height thermodynamic chart by one algebraically
responsive, quasi-hydrostatic gas+radiation column:

```text
Sigma = 2 rho H,
P     = R_g rho T + a T^4/3,
Pi    = 2 H P,
Omega_perp^2 H^2 = Pi/Sigma = P/rho.
```

The positive quadratic root determines `H(Sigma,T,Omega_perp)`. The EOS
exports exact logarithmic derivatives with respect to all three inputs. The
vertical frequency is a required provider value; WP10c3b does not identify a
unique near-horizon frequency for a noncircular plunging flow.

The physical column adiabat includes the work done by the responsive height:

```text
de - (P/rho^2) d rho = 0.
```

Consequently, the acoustic derivative is

```text
a_col^2
    = c^2 (dPi/dSigma)_s
      / (c^2 + e + Pi/Sigma),
```

where the derivative follows the hydrostatic `H(Sigma,T,Omega_perp)` surface
at frozen `Omega_perp`. It is not the fixed-height three-dimensional
gas+radiation derivative. In the local fluid rest frame, the principal
variables `(dlnSigma,dv_R/c,dlnT)` give two modes `+/-a_col/c` and one entropy
mode. The analytic and direct matrix eigenvalues agree below `8.4e-17` in
the bounded audit.

The height work can be written equivalently as

```text
Pi/Sigma^2 dSigma - P/rho^2 d rho
    = (P/rho) dlnH.
```

The temporal finite-volume correction uses

```text
integral Pi dlnH
    ~= (Pi_old+Pi_new) ln(H_new/H_old)/2.
```

For source assembly, the caller supplies the full proper-time rate

```text
dlnH/dtau
    = H_Sigma dlnSigma/dtau
    + H_T dlnT/dtau
    + H_Omega dlnOmega_perp/dtau.
```

The signed comoving column power is

```text
q_H = -Pi dlnH/dtau,
```

so compression (`dlnH/dtau<0`) heats the column. Optically thick two-face
diffusion cooling is

```text
Q_rad = 16 sigma_SB T^4/(3 kappa Sigma),
tau_sc = kappa Sigma/2,
q_rad = -Q_rad.
```

Each isotropic comoving exchange is transformed through one four-force:

```text
G^mu = q u^mu/c^3,

S_K = (
    0,
    alpha G_R,
    alpha G_phi,
   -alpha G_t
).
```

Thus a moving cooling column carries the momentum and Killing-energy exchange
required by covariance while adding no comoving momentum. The contraction
`-c^3 u_mu G^mu` recovers `q` directly.

Stress work is not inserted again as a local total-energy heat source. The
same stress tensor already carries torque and Killing power. On one finite
interval the exact midpoint product rule is

```text
Delta(Omega G)
    = Omegabar Delta G + Gbar Delta Omega.
```

The second term is the resolved shear conversion in an internal/entropy
partition; the explicit source in the conservative total Killing-energy
equation is zero. Adding both `-Omega G` in the flux and a separate `Q_visc`
would double count the same work.

The WP10c3b gate covers only local states and source integration. The
responsive-height term changes the time-derivative mass matrix, so the old
fixed-height conservative flux Jacobian is not an independent characteristic
proof for this closure. No stream, tide, wind, stationary root, or timestep is
included.

#### Kerr-Schild stream and Roche migration

WP10c4 supplies the responsive column with one explicit vertical-frequency
provider:

```text
Omega_perp(R) = c sqrt(rg/R^3),
dlnOmega_perp/dlnR = -3/2.
```

This is the positive Schwarzschild curvature scale. It is finite through the
Kerr-Schild horizon and has the correct weak-field orbital limit. It remains a
quasi-hydrostatic closure, not a resolved vertical equation for plunging gas.

One injected stream state is specified by an absolute rest-mass rate and one
local Eulerian four-state. With

```text
hbar = 1 + e/c^2 + Pi/(Sigma c^2),
u_mu = g_munu u^nu,
```

the specific Killing-chart moments per injected rest mass are

```text
p_R/c = hbar u_R,
l/c   = hbar u_phi,
E_K/c^2 = -hbar u_t.
```

The transport radial velocity is

```text
v_tr = c [alpha beta_R/sqrt(gamma_RR) - beta_shift^R].
```

Mass, radial momentum, angular momentum, and Killing energy therefore come
from one immutable state rather than four independently tunable source
numbers. For the current constant-moment compact source, exact cell weights
are differences of the analytic cumulative C2 or C4 profile:

```text
w_i = S(ln R_{i+1}) - S(ln R_i),
sum_i w_i = 1.
```

The physical mass-equivalent rates are

```text
S_i = Mdot_stream w_i
      [1, hbar u_R, hbar u_phi, -hbar u_t].
```

Because the finite-volume time coordinate is `x^0=ct`, the source inserted
into that DAE is `S_i/c`. A future ballistic source may supply a different
four-state or conservative radial moment table, but it must preserve this
single-state ledger. The WP10c4 runner's circular source state is a bounded
regression fixture, not a Layer-1 ballistic calibration.

At the Roche edge, the Kerr-Schild column supplies

```text
v_R,nozzle = v_tr,
l_kin      = c u_phi,
l_flux     = c hbar u_phi,
B_inertial = -c^2 hbar u_t - c^2.
```

The reduced Hill nozzle retains its local PW-secondary plus Hill force
geometry. A constant potential shift makes its edge inertial Bernoulli equal
`B_inertial`; this changes neither force nor edge-to-saddle availability.
The flux angular momentum uses `l_flux`, while the local rotating nozzle
kinematics use `l_kin`.

The outward physical edge rates are converted to the Killing finite-volume
chart as

```text
F_edge/c = [
    Mdot/c,
    Pdot_R/c^2,
    Jdot/c^2,
    Edot_K/c^3
].
```

Closed states retain only pressure traction. Choked states add outward nozzle
mass, momentum, angular momentum, and total Killing energy including rest
mass. Nonzero outer shear stress, inward nozzle mass, a PW disk-energy zero,
or a failed Jacobi/pattern-power ledger is rejected.

The bounded outer fixture has exactly one incoming acoustic characteristic in
both closed and choked states. Exact stream sources add no unknowns or rows.
Relative to the four-field WP10b flux-primary base, the four outer face rows
remain full rank and the count remains `12N+4`. This is not yet the final
causal-stress augmented count or its nonlinear characteristic proof; those
belong to the first production assembly.

At `10000 rg`, a cold circular fixture recovers Newtonian specific angular
momentum and binding energy with relative defects `1.50034e-4` and
`-7.50338e-5`, respectively. Exact C2/C4 stream moments close below
`2.06e-16`; the source-per-`ct` conversion closes below `1.80e-16`.

At `335 rg` the current outer flow is subsonic and requires one incoming
acoustic condition. Layer 1 does not provide an exterior thermodynamic
invariant there. ADR 0014 therefore selects an adiabatic Hill/Roche overflow
side channel ending at a regular sonic throat at an actual `L1/L2` saddle.
The production edge now reconstructs one column at exactly `335 rg`, retains
pressure traction when the channel is closed, and adds a conservative nozzle
flux only when the Jacobi gate opens. Distributed tide and wind remain
blocked until the no-tide loading evolution passes.

### Gas-radiation Hill/Roche boundary

The selected boundary provider uses the secondary PW potential plus the local
midplane Hill tide:

```text
Phi_H(R) = -G M2/(R-R_PW) - (3/2) Omega_p^2 R^2.
```

The actual saddle solves `dPhi_H/dR=0`; it is not forced to the Newtonian Hill
radius. The disk-side reservoir supplies `rho`, `T`, radial velocity, and
specific angular momentum. The production provider uses the same gas plus
radiation EOS as the disk:

```text
P = R_g rho T + a T^4/3
e = R_g T/(gamma_g-1) + a T^4/rho
h = gamma_g R_g T/(gamma_g-1) + 4 a T^4/(3 rho)
s = R_g ln(T)/(gamma_g-1) - R_g ln(rho)
    + 4 a T^3/(3 rho) + constant.
```

The acoustic speed is the exact derivative `(dP/d rho)_s`, not a weighted
local gamma approximation. The fixed-gamma provider remains only as an
analytic regression control.

The rotating Bernoulli invariant is

```text
B_J = Phi_H + h + v_R^2/2 + (l/R-Omega_p R)^2/2.
```

For the production regular sonic throat, `rho_s` and `T_s` solve

```text
s(rho_s,T_s) = s(rho_0,T_0)
h(rho_s,T_s) + c_s^2/2 = B_J-Phi_s
c_s^2 = (dP/d rho)_s.
```

Fixed Gauss-Legendre integration over the quadratic transverse saddle follows
the same isentrope and returns mass and pressure moments. In the fixed-gamma
limit it regresses to

```text
A_rho = N_channel f_fill 2 pi c_s^2
        / [gamma sqrt(Phi_yy Phi_zz)]
Mdot_ov = rho_s c_s A_rho.
```

The saddle flux satisfies

```text
F_E = F_BJ + Omega_p F_J.
```

Angular momentum exchanged between the reservoir and corotating saddle is
reported as binary torque, with paired power `Omega_p T`. A constant shift in
the Hill effective potential makes `B_J+Omega_p l` at the reservoir edge equal
the disk's PW Bernoulli without changing forces or the opening threshold.

The finite-volume edge flux is

```text
closed:  (F_M,F_J,F_E) = (0,0,0)
open:    (F_M,F_J,F_E) = nozzle edge flux
F_PR = 2 pi R Pi + F_PR,nozzle
```

so the pressure traction is continuous and every nozzle contribution tends to
zero at the opening threshold. The edge is required to remain subsonic, no
inward mass is allowed, and the provider must close disk Bernoulli, binary
torque, pattern power, and Jacobi energy before its flux is accepted.

### Adaptive global continuation

The physical backward-Euler evolution uses accepted-state continuation. A
nonlinear root is accepted only when its standard residual/ledger gates pass
and

```text
max |Delta ln Sigma| <= delta_Sigma
max |Delta ln T|     <= delta_T
max |Delta(H/R)|/(H/R) <= delta_H.
```

Failed nonlinear attempts or excessive physical changes halve `dt`. Easy
accepted steps may grow `dt`; rejected candidates never replace the last
accepted state. Restart data contain both conservative and inner-reference
states, the exact mesh and mechanical offset, controller time and next step,
checksums, counters, and provenance.

## Validity Gates

Numerical residual acceptance is necessary but not sufficient. Current audits
also monitor radial gradient length relative to scale height, radial and
vertical optical depth, vertical adjustment time, self-gravity, and conserved
flux compatibility. The first current-model failure is `L_u/H<1`, before the
formal low-velocity endpoint.

### Five-field causal DAE preflight

The causal stress augments the four Killing conservation laws with

```text
Q_chi = D chi.
```

The production primitive order is

```text
(ln Sigma, beta_R, beta_phi, ln T, chi).
```

The stress relaxation law must retain its resolved spatial principal term:

```text
tau_r u^mu nabla_mu chi + chi = nu_s q,
q = -2 c e_(R)^mu e_(phi)^nu sigma_mu_nu.
```

For stationary axisymmetric profiles, `q` is evaluated from the full
Kerr-Schild connection and `d u_mu/dR`. Its Newtonian circular limit is
`-R dOmega/dR`.

With five cell-conserved states, five primitive states, and five fluxes at
every face, the exact flux-primary count is

```text
unknowns = rows = 5N + 5N + 5(N+1) = 15N+5.
```

The covariant shear-gradient term belongs in the fifth conservation row and
does not add a separate row. The responsive-height temporal work contributes
to the Killing storage map:

```text
Delta W_H = 0.5 (Pi_old + Pi_new) Delta ln H.
```

For four-velocity `u^mu`, its mass-equivalent storage correction is

```text
Delta U_H =
alpha Delta W_H u^0/c^2
    (0, u_R, u_phi, -u_t).
```

At a coordinate-stationary subsonic Roche edge, the five-field system has one
incoming acoustic mode, one incoming shear mode, one zero contact mode, and
two outgoing modes. The physical edge therefore supplies exactly two
conditions: the Hill/Roche acoustic contract and zero shear stress. The
inner face remains inside `2 rg` and supplies no physical boundary condition.

The local count and principal audit do not authorize a stationary root until
the covariant shear path, temporal storage map, and fifth Roche face are
assembled in one nonlinear finite-volume residual.
