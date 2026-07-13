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
