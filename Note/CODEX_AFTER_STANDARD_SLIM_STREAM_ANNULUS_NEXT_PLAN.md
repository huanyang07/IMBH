# Codex Next-Step Brief: Standard Slim Disk Recovered; Next Test Stream-Fed Angular-Momentum Forcing, Not Wind Yet

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
latest main branch
```

Start from:

```text
Note/CODEX_STANDARD_SLIM_MDOT_EQUILIBRATED_CONTINUATION_RESULTS.md
```

Most relevant latest outputs:

```text
outputs/tables/slim_benchmark_adaptive_outer_mesh_mdot1_scan.md
outputs/tables/slim_benchmark_finite_boundary_homotopy_mdot1_adaptive_mesh_7000_3000.md
outputs/tables/slim_benchmark_finite_boundary_homotopy_mdot1_adaptive_mesh_3000_1000.md
outputs/tables/slim_benchmark_outer_entropy_adaptive_continuation.md
outputs/tables/slim_benchmark_outer_angular_homotopy_mdot1_rout1000.md
outputs/tables/slim_benchmark_outer_angular_homotopy_mdot1_rout1000_tinysteps.md
```

## Executive assessment

The standard no-wind slim-disk benchmark is now recovered very strongly.

The latest key facts are:

```text
1. Mdot/Edd = 1 is a strict standard no-wind benchmark after residual-adapted
   outer mesh placement.

2. Finite outer radius alone does not kill the standard no-wind branch down to
   Rout = 1000 rg.

3. Finite reservoir entropy alone does not kill the branch over the tested
   entropy offsets when the outer boundary layer is resolved.

4. A hard outer angular-momentum / circularization boundary fails immediately,
   even for tiny log-l offsets of order 1e-5--1e-3.

5. Therefore the next physical problem is not wind yet. It is how to represent
   stream-fed angular momentum physically.
```

The main conclusion is:

```text
Do not impose stream circularization as a hard Dirichlet boundary condition on
Omega or l at Rout.

Instead, add the missing physics as a distributed outer stream annulus:
mass source, angular-momentum source/torque, and stream shock heating.
```

---

# 1. What is now established

## 1.1 Standard no-wind slim disk is recovered

The adaptive mesh scan at `Mdot/Edd = 1` shows that the earlier
near-Eddington residual floor was just an outer-grid placement problem.

The best adaptive mesh result gives:

```text
Mdot/Edd = 1
N = 640
final residual = 1.378e-7
dominant = interval_E
max H/R = 0.157
integrated advective fraction = 0.0316
```

This is a very clean standard no-wind slim benchmark.

Interpretation:

```text
The core transonic solver can recover a global no-wind standard slim disk.
```

This means the solver itself is no longer the main conceptual uncertainty.

---

## 1.2 Finite radius alone is not the problem

Using the adaptive outer mesh at fixed:

```text
Mdot/Edd = 1
```

the standard no-wind branch remains strict while shrinking the outer radius:

```text
Rout = 7000 rg
Rout = 5000 rg
Rout = 3000 rg
Rout = 2000 rg
Rout = 1000 rg
```

The `Rout = 1000 rg` result is especially clean:

```text
residual = 1.711e-7
Rson ~= 5.070 rg
max H/R ~= 0.157
int_adv ~= 0.032
```

Interpretation:

```text
A finite outer radius by itself does not reproduce the IMRI no-wind obstruction.
```

Therefore, the previous IMRI failure is not simply because the disk is finite.

---

## 1.3 Finite entropy reservoir alone is not the problem

At:

```text
Mdot/Edd = 1
Rout = 1000 rg
N = 640
```

the entropy boundary homotopy initially failed at moderate offsets on the
original grid, but adaptive outer remeshing rescued those cases.

After remeshing, the branch remains strict over at least:

```text
Delta logK = -0.010 to +0.020
```

with:

```text
Rson ~= 5.07 rg
max H/R ~= 0.157
int_adv ~= 0.032
```

Interpretation:

```text
A finite entropy reservoir alone does not kill the no-wind branch.
```

So the IMRI obstruction is probably not just finite reservoir thermodynamics.

---

## 1.4 Hard angular-momentum boundary is the first real obstruction

The outer angular-momentum/circularization test is qualitatively different.

At fixed:

```text
Mdot/Edd = 1
Rout = 1000 rg
N = 640
```

the code imposes an outer angular-velocity/l offset:

```text
outer_omega_log_offset
```

A one-per-mille offset fails:

```text
log-l offset = -0.001:
    residual ~= 9.99e-4
    dominant = outer_omega

log-l offset = +0.001:
    residual ~= 9.99e-4
    dominant = outer_omega
```

Even tiny offsets are barely tolerated:

```text
log-l offset = -1e-5:
    residual ~= 9.8e-6
```

Interpretation:

```text
The disk refuses to satisfy a hard outer angular-momentum Dirichlet condition.
It stays close to its pressure-supported angular velocity, and the mismatch
appears almost entirely as an outer boundary residual.
```

This does not mean stream angular momentum is physically impossible.

It means:

```text
A hard boundary value for l or Omega is the wrong mathematical model for a
stream-fed disk.
```

A stream does not magically replace the disk's angular velocity at the outer
edge. It injects:

```text
mass
angular momentum
kinetic/thermal energy
```

over a finite annulus and lets the disk adjust through torque and dissipation.

---

# 2. Physical interpretation

The standard slim benchmark now tells us:

```text
No-wind slim-disk physics itself can work.
```

The finite-radius and entropy homotopies tell us:

```text
The no-wind branch can survive finite reservoir size and finite reservoir
entropy, at least in standard-disk geometry.
```

The angular-boundary failure tells us:

```text
The first genuinely minidisk-specific difficulty is angular momentum
circularization.
```

Physically, stream-fed gas arrives with a specific angular momentum that need
not equal the local pressure-supported disk angular momentum. If we impose that
as a hard boundary value, the BVP becomes overconstrained.

The physical disk should instead solve a transition problem:

```text
stream angular momentum -> outer annulus torque/heating -> disk angular
momentum profile.
```

That transition requires source terms. Without them, the disk has no way to
convert the imposed outer circularization condition into a smooth transonic
profile.

---

# 3. Updated program direction

Do not add wind yet.

The next step should be:

```text
standard no-wind finite-boundary disk
+ distributed stream angular-momentum injection
+ optional stream shock heating
+ optional tidal torque
```

Only after that no-wind stream-fed model fails should wind be added.

The current sequence should be:

```text
A. complete finite-Rout shrink toward the physical minidisk radius;
B. replace hard angular boundary with a distributed stream annulus;
C. add stream heating;
D. add tidal torque;
E. then test whether no-wind still fails;
F. only then add wind.
```

---

# 4. Immediate numerical step A: finish finite-Rout shrink to physical minidisk scale

Previous estimate for the fiducial IMRI minidisk:

```text
Rout ~= 0.3--0.5 RH ~= 200--450 rg,2
fiducial Rout ~= 300 rg,2
```

The current finite-radius homotopy reaches:

```text
Rout = 1000 rg
```

Next shrink:

```text
1000 -> 700 -> 500 -> 400 -> 300 rg
```

Use:

```text
Mdot/Edd = 1
adaptive outer mesh
pressure-supported outer closure
no angular offset
no stream source yet
```

Acceptance:

```text
residual <= few x 1e-6
Rson stable
lambda0 stable
max H/R stable
int_adv stable
no sonic/fold pathology
```

This answers:

```text
Does a standard no-wind disk remain valid down to the physical minidisk Rout?
```

Expected result:

```text
probably yes, based on the smooth shrink to 1000 rg.
```

But this should be verified.

---

# 5. Immediate physical step B: replace hard angular boundary by stream-annulus torque/source

## 5.1 Do not use hard outer omega offset as the production model

The hard angular test is still valuable because it shows sensitivity, but it
should not be used as the physical stream model.

A hard Dirichlet condition says:

```text
Omega(Rout) = prescribed stream circularization value.
```

A stream-fed disk should instead have:

```text
mass source S_m(R)
angular momentum source S_m(R) l_in
stream shock heating Q_stream(R)
possibly tidal torque Lambda_tide(R)
```

The disk then decides its own Omega profile.

---

## 5.2 First implementation: torque-only annulus

Before adding mass source, test a simple external torque annulus. This is the
least disruptive modification of the current constant-Mdot solver.

Add to angular momentum balance a prescribed torque density:

```text
G_ext(R)
```

localized near the outer disk:

```math
G_{\rm ext}(R)
=
G_0
\exp\left[-\frac{(R-R_{\rm inj})^2}{2\Delta_R^2}\right].
```

Use:

```text
R_inj = 0.7--0.9 Rout
Delta_R = 0.05--0.15 Rout
```

Interpretation:

```text
This torque represents angular momentum exchange between incoming stream and
outer disk without yet changing Mdot(R).
```

Homotopy:

```text
epsilon_J = 0 -> 1
```

where:

```text
epsilon_J = 0: standard finite disk
epsilon_J = 1: target stream angular-momentum forcing
```

Diagnostics:

```text
Omega/Omega_K
outer angular offset achieved
residual
Rson
lambda0
int_adv
torque work if included
```

Acceptance:

```text
smooth continuation with residual <= few x 1e-6--1e-5
no artificial boundary residual
```

---

## 5.3 Second implementation: mass + angular-momentum source annulus

After torque-only works, generalize the steady equations to allow:

```math
S_\Sigma(R)\neq0.
```

With inward-positive accretion rate:

```math
\dot M=-2\pi R\Sigma v_R,
```

steady continuity gives:

```math
\frac{d\dot M}{dR}
=
2\pi R(\dot\Sigma_w-S_\Sigma).
```

For no wind:

```math
\frac{d\dot M}{dR}
=
-2\pi R S_\Sigma.
```

If mass is injected near the outer disk, then:

```text
Mdot is largest inside the source annulus,
and smaller outside it.
```

Add source profile:

```math
S_\Sigma(R)
=
\frac{\dot M_{\rm cap}}
{2\pi R_{\rm inj}\sqrt{2\pi}\Delta_R}
\exp\left[-\frac{(R-R_{\rm inj})^2}{2\Delta_R^2}\right].
```

Angular momentum source:

```math
S_\Sigma l_{\rm in}.
```

The angular momentum equation should include:

```math
S_\Sigma(l_{\rm in}-l)
```

in the local radial velocity / torque balance.

This is the physically meaningful version of stream circularization.

---

## 5.4 Third implementation: stream shock heating

Add:

```math
Q_{\rm stream}(R)
=
\frac{1}{2}
S_\Sigma
|\mathbf v_{\rm in}-\mathbf v_{\rm disk}|^2.
```

For a first scalar model use:

```math
Q_{\rm stream}
=
\epsilon_{\rm sh}
S_\Sigma v_K^2
```

with:

```text
epsilon_sh = 0, 0.01, 0.03, 0.1
```

This tests whether stream heating shifts the thermal branch.

Do this only after the mass/angular momentum source is stable.

---

# 6. Suggested homotopy sequence

Use fixed:

```text
Mdot/Edd = 1
Rout = 300 or 500 rg
N chosen by adaptive mesh
no wind
```

Then turn on physical ingredients:

## Stage 0: finite standard disk

```text
pressure-supported outer closure
no stream
no torque
no heating
```

## Stage 1: external angular momentum torque

```text
epsilon_J = 0, 0.1, 0.2, ..., 1
```

## Stage 2: mass source with matching angular momentum source

```text
epsilon_M = 0, 0.1, ..., 1
```

where \(\dot M(R)\) is allowed to vary.

## Stage 3: stream shock heating

```text
epsilon_sh = 0, 0.01, 0.03, 0.1
```

## Stage 4: tidal torque

Add an outer tidal torque:

```math
\Lambda_{\rm tide}(R)
```

localized near:

```text
Rout
```

or use a smooth barrier torque.

Only after Stage 4 should the project decide whether no-wind stream-fed
minidisk equilibria exist.

---

# 7. What would count as physical no-wind failure?

A physical no-wind failure would be:

```text
during the homotopy, the branch disappears through a documented fold or
critical event,
with residuals converged,
with mesh convergence,
and not dominated by an artificial boundary residual.
```

The failed hard angular-boundary test is not yet that. It is a boundary
overconstraint test.

A real failure would look like:

```text
distributed source/torque model
residuals controlled
mesh checked
then no solution beyond some stream angular-momentum forcing
```

That would be meaningful.

---

# 8. Why this plan may work

It may work because the latest results show the disk has no trouble with:

```text
finite radius
finite entropy reservoir
near-Eddington standard no-wind transonic structure
```

The only thing it rejects is:

```text
instantaneous hard angular-momentum replacement at the boundary.
```

A distributed source/torque annulus is much less singular. It gives the disk a
radial transition layer where angular momentum and energy can be exchanged.

Physically, that is what a stream impact/circularization region should be.

This is also the right bridge toward a future limit-cycle model, because the
stream annulus naturally supplies:

```text
mass loading
angular momentum loading
shock heating
```

which are exactly the source terms needed in the time-dependent disk.

---

# 9. Concrete scripts to add

## Finite-Rout continuation

```text
scripts/run_standard_slim_physical_rout_homotopy.py
```

Outputs:

```text
outputs/tables/slim_benchmark_physical_rout_homotopy.md
outputs/figures/slim_benchmark_physical_rout_homotopy.png
```

## Torque annulus homotopy

```text
scripts/run_stream_annulus_torque_homotopy.py
```

Outputs:

```text
outputs/tables/stream_annulus_torque_homotopy.md
outputs/figures/stream_annulus_torque_homotopy.png
```

## Source annulus steady model

```text
scripts/run_stream_annulus_mass_angular_source.py
```

Outputs:

```text
outputs/tables/stream_annulus_mass_angular_source.md
outputs/figures/stream_annulus_mass_angular_source.png
```

## Stream heating scan

```text
scripts/run_stream_annulus_heating_scan.py
```

Outputs:

```text
outputs/tables/stream_annulus_heating_scan.md
outputs/figures/stream_annulus_heating_scan.png
```

---

# 10. Code-level suggestions

## 10.1 Add source-aware profile variables

The current standard solver assumes constant:

```text
Mdot
```

For source annulus work, introduce:

```text
Mdot(R)
```

either as:

```text
prescribed from S_sigma
```

or as an additional unknown satisfying steady continuity.

First implementation should prescribe \(Mdot(R)\) from the source profile to
avoid too many new unknowns.

## 10.2 Add external angular momentum term

In the local angular momentum closure, add a source/torque contribution:

```math
\dot M(l-l_0)
=
2\pi R^2 W
+
\mathcal G_{\rm ext}(R)
```

or equivalent integrated form, where:

```math
\mathcal G_{\rm ext}(R)
=
\int_{R_{\rm in}}^R
2\pi R' S_\Sigma(R') [l_{\rm in}-l(R')]\,dR'
+
\int_{R_{\rm in}}^R
2\pi R'\Sigma\Lambda_{\rm tide}\,dR'.
```

Start with a prescribed smooth \(\mathcal G_{\rm ext}(R)\) before computing it
self-consistently.

## 10.3 Add stream heating to energy

```math
Q_{\rm visc}+Q_{\rm stream}
=
Q_{\rm rad}+Q_{\rm adv}
```

with \(Q_{\rm stream}\) localized near \(R_{\rm inj}\).

## 10.4 Keep pressure-supported outer closure

Do not revert to exact Keplerian or hard angular Dirichlet closure.

Use the pressure-supported closure plus source/torque annulus.

---

# 11. Recommended immediate experiment matrix

## Experiment A: physical Rout shrink

```text
Mdot/Edd = 1
Rout = 1000, 700, 500, 400, 300 rg
N adaptive
pressure-supported closure
```

## Experiment B: torque-only annulus

At:

```text
Rout = 500 and 300 rg
```

try:

```text
Rinj/Rout = 0.7, 0.85
DeltaR/Rout = 0.05, 0.1
epsilon_J = 0 -> 1
```

## Experiment C: source annulus

Use:

```text
Mdot_cap / Mdot_inner = 0.1, 0.3, 1
l_in/l_K(Rinj) = 0.8, 1.0, 1.2
```

## Experiment D: heating scan

Use:

```text
epsilon_sh = 0, 0.01, 0.03, 0.1
```

---

# 12. What not to do next

Do not add wind yet.

Do not conclude no-wind fails from the hard angular boundary.

Do not impose stream circularization as a Dirichlet \(\Omega\) boundary.

Do not move to time-dependent limit cycles until the stream-fed equilibrium
model exists or fails cleanly.

---

# 13. Compact Codex prompt

```text
Latest results show the standard no-wind slim disk is robust:
Mdot/Edd=1 becomes a strict anchor after adaptive outer mesh; finite Rout alone
survives down to Rout=1000 rg; finite entropy reservoir survives offsets
Delta logK=[-0.010,+0.020] after outer remeshing. The first real obstruction is
a hard outer angular momentum/circularization boundary: even log-l offsets of
1e-5--1e-3 are not absorbed and appear almost exactly as outer_omega residual.

This means hard angular Dirichlet conditions are the wrong stream model.

Next:
1. Continue finite Rout from 1000 to 700,500,400,300 rg with pressure-supported
   closure and adaptive mesh.
2. Replace hard angular offset by a distributed stream annulus:
   first torque-only, then mass+angular source, then stream heating.
3. Add a smooth external angular momentum torque term localized at Rinj~0.7--0.9 Rout.
4. Later allow Mdot(R) from a mass source S_sigma and add S_sigma(l_in-l) in
   angular momentum.
5. Add stream shock heating Q_stream after source/torque works.
6. Only after the no-wind stream-fed annulus succeeds or fails cleanly should
   wind be added.
```

---

# 14. Bottom line

The standard no-wind solver is now healthy.

Finite outer radius and finite entropy reservoir do not kill the branch.

The next meaningful physical test is:

```text
distributed stream-fed angular momentum and heating,
not hard angular boundary and not wind yet.
```
