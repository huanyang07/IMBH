# Review of Commit `5d36c24` and Locked Plan Toward a Hot Advective Solution

**Project:** IMBH/QPE stream-fed minidisk
**Repository:** `huanyang07/IMBH`
**Commit reviewed:** `5d36c244b5cf4f5ee72ef091eff9394b46169abe`
**Commit title:** `Audit pressure supported reservoir closure`
**Review date:** 2026-07-11, Asia/Tokyo
**Review type:** Static review of the checked-in status, equations, reports, canonical summaries, implementation, and tests. The production calculations were not independently rerun in the review runtime.

---

# 1. Executive verdict

Commit `5d36c24` makes a scientifically useful negative result explicit:

> The projected, staggered pressure-supported reservoir iteration is not a production solver.

The pilot:

- improves the angular-frequency match at the interface;
- retains machine-level conserved-flux matching;
- converges to full pressure support only on the coarse `N=64` grid;
- fails the fixed-point gate on every tested `N=128` configuration;
- does not reduce the dominant integrated-pressure or surface-density jump;
- and leaves a nonzero radial-force mismatch because the rotation profile is projected rather than solved.

The repository is correct to reject more damping, smoothing, or tolerance relaxation as the next route.

The path to a hot advective solution is still open.

There is already meaningful evidence for a candidate:

- the certified inner no-wind slim disk exists at `Mdot/Mdot_Edd=5`;
- the one-way conservative composite is nearly interface-position independent in total luminosity;
- its reservoir reaches `H/R ~ 0.30-0.32`;
- and the ideal tidal wall remains much warmer/thicker than an open overflow boundary.

What is not established is a single smooth, physically closed disk with:

- one stress prescription;
- continuous primitive variables;
- calibrated tidal torque and power;
- and a stability result.

The immediate plan should **not** jump directly from the rejected staggered iteration to a large new nonlinear solver.

First perform one cheap, decisive stress-closure parity test.

---

# 2. Current project status

## 2.1 Strong results

### Certified inner benchmark

The standard no-wind transonic/slim disk through

\[
\dot M/\dot M_{\rm Edd}=5
\]

remains the strongest numerical benchmark.

### Corrected signed total-energy ledger

The enthalpy-compatible work identity now closes directly against the transonic entropy equation, and the old mixed energy pairing has been superseded.

### Conservative interface machinery

The inner and outer modules share

\[
\dot M,\qquad
J=\dot M l-G,\qquad
F_E=\dot M B-\Omega G.
\]

The one-way interface sweep closes those fluxes to roundoff and gives a composite luminosity that changes by only about `0.20-0.24%` when the interface is moved from `30` to `60 r_g`.

### Signed reservoir

The outer disk has independent surface density, regular zero-flux states, decretion, an exact stream mass/angular/energy source, and an ideal-wall/open-edge comparison.

## 2.2 Diagnostic results

### One-way composite

The one-way transonic-to-reservoir composite is conservative but not smooth.

At `N=256`, the dominant interface mismatch is

\[
\Delta\ln\Pi \simeq -0.327\ \text{to}\ -0.334,
\]

while the outer surface density is lower by roughly `0.13-0.25` in log units and the angular frequency differs by about `4-5%`.

### Pressure-supported pilot

At the `40 r_g` interface, the coarse full-pressure pilot gives approximately

```text
rotation mismatch                    0.00366
integrated-pressure log mismatch    -0.3560
surface-density log mismatch        -0.3107
temperature log mismatch            -0.0833
maximum force-balance mismatch       0.01384
```

It therefore improves rotation but worsens the dominant pressure and density mismatch.

No `N=128` case passes the stated fixed-point gate.

### Hot-reservoir interpretation

The warm/thick wall state is still a candidate, not a certified physical branch.

The current evidence supports:

> Angular-momentum confinement can produce a warmer, thicker reservoir under the current equations.

It does not yet support:

> A smooth, stable, physically powered hot advective minidisk has been recovered.

---

# 3. New highest-priority finding: the inner and outer stress closures are not the same

This should be checked before implementing the proposed simultaneous `(Sigma,T,Omega)` solver.

## 3.1 Inner transonic stress

The inner transonic model uses

\[
W_\alpha=\alpha\Pi
\]

for the current total-pressure parameters

```text
mu_stress = 0
stress_factor = 1.
```

Its angular relation implies

\[
G_{\rm inner}=2\pi R^2W_\alpha
             =2\pi\alpha R^2\Pi.
\]

## 3.2 Outer reservoir stress

The signed reservoir uses

\[
G_{\rm outer}
=
-2\pi R^3\nu\Sigma\frac{d\Omega}{dR},
\]

with

\[
\nu=\alpha H^2\Omega.
\]

The shared vertical closure gives exactly

\[
\frac{\Pi}{\Sigma}=\Omega_K^2H^2.
\]

Therefore

\[
G_{\rm outer}
=
2\pi\alpha R^2\Pi\,
\underbrace{
\left(-\frac{d\ln\Omega}{d\ln R}\right)
\left(\frac{\Omega}{\Omega_K}\right)^2
}_{\chi_{\rm shear}}.
\]

The two domains agree only when

\[
\chi_{\rm shear}=1.
\]

For a Keplerian Paczynski-Wiita disk at `40 r_g`,

\[
-\frac{d\ln\Omega_K}{d\ln R}
=
\frac12+\frac{R}{R-2r_g}
\simeq1.5526.
\]

Thus, even before pressure support,

\[
G_{\rm outer}\simeq1.55\,G_{\rm inner}
\]

for the same \(\Pi\).

Equivalently, for the same required torque,

\[
\frac{\Pi_{\rm outer}}{\Pi_{\rm inner}}
\sim\frac{1}{1.55}
\simeq0.64,
\]

or

\[
\Delta\ln\Pi\sim-0.44.
\]

The measured mismatch is approximately `-0.33` to `-0.36`.

This is not proof that stress inconsistency explains the entire mismatch, but the scale and sign are close enough that it is the first issue to test.

The pressure-supported pilot changes rotation but retains the same diffusive-viscosity stress architecture, so it cannot by itself remove this mismatch.

## 3.3 Recommended common stress prescription

Use the exact same stress closure in both domains:

\[
W_\alpha
=
{\tt integrated\_stress}(\Sigma,T;\alpha,\mu_{\rm stress},
{\tt stress\_factor}),
\]

\[
G_\alpha=2\pi R^2W_\alpha.
\]

For the current benchmark,

\[
G_\alpha=2\pi\alpha R^2\Pi.
\]

Use

\[
Q^+=-W_\alpha\frac{d\Omega}{d\ln R}
\]

for viscous heating and retain

\[
-\Omega G_\alpha
\]

in the total-energy flux.

Do not infer the steady torque from

\[
\nu=\alpha H^2\Omega
\]

if the inner model remains an alpha-stress model.

If a kinematic viscosity is later needed for a time integrator, define the diagnostic/effective value from the common stress:

\[
\nu_{\rm eff}
=
\frac{W_\alpha}
     {-\Sigma R\,d\Omega/dR},
\]

with an explicit guard when shear approaches zero.

The steady physical equation should use the stress, not an unrelated viscosity approximation.

---

# 4. Does a path to a hot advective solution still exist?

## Yes, but the target must be stated precisely

The evidence supports a credible path because:

1. the inner high-rate slim branch already exists;
2. mass, angular momentum, and total-energy fluxes can be joined conservatively;
3. composite luminosity is nearly independent of the artificial interface position;
4. a confined wall state consistently reaches `H/R ~ 0.3`;
5. the latest failure is a rejected numerical splitting architecture, not a demonstrated absence of a nonlinear root;
6. the dominant primitive mismatch may be substantially contaminated by the stress-law inconsistency identified above.

## What would count as success

Define a hot advective solution before searching for it.

A suggested certification requires all of:

```text
smooth primitive state across the inner/outer transition
mass/angular/total-energy residuals pass
same stress and thermodynamic closure in both domains
no projected or clipped rotation profile
Rayleigh stable and negative shear
effective optical depth passes the chosen radiative gate
radial-force and scale-separation gates pass
solution is mesh supported
solution is insensitive to interface placement
entropy-advection fraction is measured directly
solution is stable, or its instability is demonstrated in time evolution
tidal torque and power are physically calibrated
```

Suggested reporting metrics:

\[
f_{\rm adv}
=
\frac{\int Q_{\rm adv}\,dA}
     {\int Q^+_{\rm available}\,dA},
\]

plus:

```text
H/R profile and contiguous hot-zone width
trapping radius
effective and scattering optical depth
radial Mach number
radial-pressure fraction
luminosity
inner accretion fraction
outer overflow fraction
tidal torque and power
```

Do not certify “hot” from a single maximum `H/R`.

A useful provisional target is:

```text
f_adv >= 0.5 over a physically resolved radial region
and/or
H/R >= 0.25 over a contiguous region wider than several scale heights
```

These are project reporting gates, not universal physical definitions.

---

# 5. Locked implementation plan

The purpose of this plan is to avoid another cycle of:

```text
modify one closure
scan damping
move interface
relax tolerance
repeat
```

Each work package has one hypothesis and one go/no-go result.

---

## WP0 — Freeze commit `5d36c24`

No new physics.

1. Preserve the pressure-supported pilot as a rejected-production witness.
2. Keep its canonical N64 roots and N128 failures.
3. Label it:
   ```text
   numerical_status = diagnostic
   production_status = rejected
   ```
4. Add the stress-closure distinction explicitly to `MODEL_EQUATIONS.md`.
5. Do not run more smoothing, damping, or tolerance scans on the staggered solver.

Deliverable:

```text
ADR: projected staggered pressure support is closed
```

---

## WP1 — Stress-closure parity audit

This is the immediate task and should be completed before the monolithic solver.

### Implementation

Add a steady reservoir mode using

\[
G=2\pi R^2W_\alpha
\]

with exactly the same `integrated_stress()` routine and parameters as the inner transonic solver.

Retain initially:

```text
Omega = Omega_K
corrected total-energy equation
same prescribed inner conserved flux
same stream state
same ideal outer wall
```

The purpose is to isolate stress closure only.

### Required tests

#### Local parity

On any shared \((\Sigma,T,R)\) state, require

```text
G_inner_alpha == G_outer_alpha
W_inner == W_outer
Qplus definitions agree for the same Omega gradient
```

to roundoff.

#### Certified-profile parity

Sample the certified no-wind transonic profile at `30,40,50,60 r_g`.

Compare:

```text
required torque G
alpha-stress torque
old diffusive-viscosity torque
chi_shear
predicted pressure offset
measured pressure offset
```

#### Interface sweep

Repeat the Keplerian one-way interface sweep at:

```text
R_interface = 30, 40, 50, 60 r_g
N = 128, 256
```

with no other change.

### Decision gate

#### Gate A — stress mismatch was dominant

Proceed directly to WP3 if:

```text
maximum primitive mismatch <= 0.10
pressure mismatch decreases monotonically with refinement
composite luminosity remains interface independent
```

A small non-Keplerian correction can then be included inside the coupled solve.

#### Gate B — stress mismatch was not sufficient

Proceed to WP2 if:

```text
pressure mismatch remains > 0.10
or
the result is not mesh supported.
```

Do not add more empirical stress factors to force a match.

---

## WP2 — Simultaneous non-Keplerian reservoir residual

Implement only if WP1 does not close the primitive mismatch.

### Unknowns

At each reservoir cell use:

\[
\log\Sigma_i,\qquad
\log T_i,\qquad
\log\Omega_i.
\]

Mass flux and angular flux are integrated exactly from the source and boundary conditions.

### Equations

#### Angular constitutive residual

\[
G_{\rm required}
=
\dot M l-J,
\]

\[
G_{\rm required}
-
2\pi R^2W_\alpha(\Sigma,T)
=
0.
\]

This uses the common alpha stress.

#### Radial momentum residual

Use the same physical equation as the inner solver:

\[
u^2\frac{d\ln u}{d\ln R}
-
R^2(\Omega^2-\Omega_K^2)
+
\frac{1}{\Sigma}\frac{d\Pi}{d\ln R}
=
0,
\]

where

\[
u=\frac{\dot M}{2\pi R\Sigma}
\]

with its signed interpretation handled consistently.

Do not omit the radial-inertia term merely because it is small in the initial overlap.

#### Total-energy residual

Use the corrected enthalpy form:

\[
F_E=\dot M B_{\rm col}-\Omega G,
\]

with the compatible one-zone vertical work, radiative cooling, stream energy, and named external power.

### Stress and viscosity

Use the common alpha stress directly.

Do not use

\[
\nu=\alpha H^2\Omega
\]

as an independent closure in this steady solve.

### Numerical architecture

- one residual;
- one sparse Jacobian;
- no staggered pressure update;
- no Savitzky-Golay smoothing inside the physical equation;
- no slope projection;
- no clipping.

Use logarithmic variables for positivity.

Rayleigh stability and negative shear are acceptance gates. A differentiable barrier may protect trial steps, but it must vanish at the accepted root and its contribution must be reported.

### Continuation

Use one homotopy only:

\[
{\cal R}_R(\lambda)
=
(1-\lambda)\ln(\Omega/\Omega_K)
+
\lambda\,{\cal R}_{\rm radial},
\qquad
0\le\lambda\le1.
\]

Sequence:

```text
N=64: lambda 0 -> 1
prolongate accepted root
N=128: lambda 1
N=256: lambda 1
```

Use pseudo-arclength only if the lambda branch folds.

Do not scan damping and smoothing grids.

### Acceptance gate

```text
all normalized equation blocks <= 1e-7
mass/J/F_E global ledgers <= 1e-9 relative
no projection or clipping
dln(l)/dlnR > 0
dln(Omega)/dlnR < 0
mesh-supported N128/N256
radial-force mismatch <= 1e-4
```

If this solver cannot produce a mesh-supported root from the stress-consistent WP1 state, stop outer-interface patching and move to the single-domain fallback in Section 8.

---

## WP3 — Fully coupled two-domain bordered solve

Do not keep the inner transonic solution frozen.

The one-way sweep fixes the inner eigenvalues and fluxes from the no-stream benchmark. A real outer wall and stream can feed back on:

```text
inner angular eigenvalue
sonic radius
interface fluxes
inner thermal state
```

### Coupled unknowns

Include as needed:

```text
inner transonic state
inner angular eigenvalue
sonic radius/eigenparameter
reservoir Sigma,T,Omega
interface flux triple
possibly interface radius
```

### Interface conditions

Require conservation:

\[
[\dot M]=0,\qquad[J]=0,\qquad[F_E]=0.
\]

Also require smooth-state compatibility over an overlap band, not only one node:

```text
Sigma
T or entropy
Pi
Omega
H
radial velocity
```

Use an overlap least-squares block across several neighboring points.

Do not impose all primitive quantities as independent hard conditions without first documenting the differential order and boundary-condition count.

### Mandatory pre-code document

Create:

```text
docs/decisions/0010-coupled-problem-dof-and-boundary-count.md
```

It must list:

```text
unknown functions
differential/algebraic order
global eigenparameters
inner conditions
outer conditions
interface conditions
rank expected
null modes
```

A numerical SVD rank audit must accompany the first root.

### Interface sweep

Solve at:

```text
30, 40, 50, 60 r_g
```

using continuation from a common root.

The physical solution must be insensitive to interface placement.

### Acceptance gate

```text
conserved flux mismatch <= 1e-8
primitive mismatch <= 0.05 in log/relative norm
composite luminosity spread <= 1%
key hot metrics spread <= 2%
N128/N256 agreement
all validity gates pass
```

---

## WP4 — Physical tidal torque and power

Only after WP3 closes.

Retain:

```text
open edge
ideal wall
```

as limiting controls.

Add one binary-calibrated torque family with pattern speed \(\Omega_p\).

The angular and energy terms must be paired:

\[
T_{\rm tide},
\qquad
P_{\rm tide}=\Omega_pT_{\rm tide},
\]

with differential-rotation dissipation assigned explicitly.

Continue one dimensionless torque amplitude from open to wall.

Report:

```text
required torque / stream angular flux
binary torque capacity
pattern-speed power
local tidal dissipation
inner accretion fraction
overflow fraction
f_adv
H/R
luminosity
```

### Scientific decision

A hot branch is physically plausible only if it exists for a torque that the companion can supply without violating the disk-plus-orbit energy ledger.

---

## WP5 — Stability and time evolution

After a smooth steady branch is obtained:

1. solve from cold and hot seeds;
2. identify multiple roots;
3. calculate the coupled Jacobian spectrum;
4. perturb the hot candidate;
5. implement coupled mass/angular/total-energy IMEX evolution;
6. allow accumulation, fronts, and limit cycles.

A steady hot root that is unstable is still valuable: it may organize the desired burst cycle.

If no steady root exists under physical feeding, time dependence becomes the main result rather than a fallback numerical trick.

---

## WP6 — Wind

Wind remains last.

Port the terminal-Bernoulli wind only after:

```text
stress parity
smooth inner/outer coupling
physical tide and power
stability/time-evolution baseline
```

all pass.

Use one wind state carrying mass, angular momentum, and energy.

Near stagnation, cap launch by:

```text
surface-layer mass
vertical launch time
momentum/radiation supply
```

not by the magnitude of radial throughput.

---

# 6. Tests Codex should add now

## Stress parity

```text
test_alpha_stress_torque_matches_inner_definition
test_vertical_closure_Pi_over_Sigma_equals_OmegaK2_H2
test_diffusive_alpha_factor_is_reported_not_silently_assumed
test_interface_pressure_offset_prediction
```

## Monolithic residual

```text
manufactured angular/radial/energy solution
Keplerian lambda=0 recovery
pressure-supported lambda=1 recovery
Rayleigh-stable root
negative-shear root
no-projection assertion
Jacobian finite-difference/analytic comparison
```

## Coupling

```text
degree-of-freedom rank test
inner/outer common-state manufactured match
interface-position invariance
conserved-flux plus primitive-continuity gate
```

## Hot-state audit

```text
entropy-advection computed independently from the final root
contiguous H/R hot-zone width
effective optical depth
radial and vertical gradient scales
tidal power ledger
```

---

# 7. Rules to prevent further back-and-forth

Codex should follow these rules until WP4 is complete.

1. **One production branch only.**
   Do not maintain several pressure-supported solvers.

2. **No more staggered damping scans.**
   Commit `5d36c24` has already answered that question.

3. **No tolerance relaxation as physics.**
   A failed mesh gate remains failed.

4. **Change one physical closure at a time.**
   Stress parity first; radial momentum second; tide third; wind last.

5. **Every work package ends in a binary decision.**
   Pass, reject, or escalate to the named fallback.

6. **No new large parameter map before a smooth baseline exists.**

7. **Do not freeze inner fluxes in the final coupled model.**
   The inner eigenvalues must respond to the outer disk.

8. **No projected accepted states.**
   Projection may generate a seed, never the reported root.

9. **Use the same stress, thermodynamics, potential, signs, and energy zero in both domains.**

10. **Preserve negative results, but stop rerunning rejected architectures.**

---

# 8. Single-domain fallback

If WP2 or WP3 fails after the stress-law correction, stop interface engineering.

Build one global conservative steady system with unknowns such as

\[
\Sigma,\quad T,\quad u,\quad\Omega
\]

from the transonic inner edge to the tidal boundary, with:

- signed mass flux;
- radial momentum;
- common alpha stress;
- angular momentum conservation;
- corrected total energy;
- exact stream moments;
- physical tide and power;
- a sonic regularity eigencondition.

This is larger, but it removes the possibility that the primitive jump is an artifact of incompatible domain reductions.

Do not attempt another intermediate splice architecture after the coupled two-domain gate fails.

---

# 9. Recommended immediate Codex prompt

```text
Review commit 5d36c24 and implement WP0-WP1 only.

Do not build the simultaneous pressure-supported solver yet.

First audit and remove the stress-law mismatch between the certified inner
transonic solver and the signed reservoir.

The inner benchmark uses:
    W = alpha Pi
    G = 2 pi R^2 W

The reservoir currently uses:
    G = -2 pi R^3 nu Sigma dOmega/dR
    nu = alpha H^2 Omega

Because Pi/Sigma = Omega_K^2 H^2, the reservoir torque differs by
chi = -(dlnOmega/dlnR)(Omega/Omega_K)^2.
At 40 rg in the Keplerian PW limit chi is about 1.5526, which can naturally
produce a negative log-pressure offset comparable to the observed
-0.33 to -0.36 mismatch.

Add a stress-consistent reservoir mode using the exact same
integrated_stress() routine, mu_stress, and stress_factor as the inner solver:
    G = 2 pi R^2 integrated_stress(...)

Use the same stress for Qplus and retain -Omega G in the total-energy flux.

Add local parity tests and repeat only the 30/40/50/60 rg, N128/N256
one-way interface sweep. Change no other physics.

Decision:
- If maximum primitive mismatch falls below 0.10 and is mesh supported,
  proceed to a coupled bordered inner/outer solve.
- Otherwise implement one simultaneous Sigma,T,Omega residual using the
  common alpha stress, exact radial momentum including radial inertia,
  and corrected total energy. No projection, smoothing, or damping scan.

Produce one report that quantifies how much of the pressure mismatch was
caused by the stress-law inconsistency.
```

---

# 10. Bottom line

Yes, there is still a credible path to a hot advective solution.

The project now has:

- a certified high-rate inner slim branch;
- a conservative signed outer reservoir;
- a corrected total-energy ledger;
- and strong evidence that angular-momentum confinement creates a warm/thick state.

The latest failure says that the **staggered projected algorithm** is wrong.

It does not say that the hot branch is absent.

Before paying for a large simultaneous solver, test the simpler and more fundamental inconsistency: the inner and outer domains currently do not use the same alpha-stress law.

That one work package may remove much of the primitive jump. If it does not, the project should proceed exactly once to a monolithic non-Keplerian residual, then exactly once to a coupled bordered solve, with a single-domain global solver as the final fallback.
