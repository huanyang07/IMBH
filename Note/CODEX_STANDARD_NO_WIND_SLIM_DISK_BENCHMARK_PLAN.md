# Codex Benchmark Plan: Reproduce a Published No-Wind Slim Disk Before Adding Wind

Repository target:

```text
https://github.com/huanyang07/IMBH
```

Goal:

```text
Before adding wind, test whether the current transonic machinery can recover a
standard single-black-hole, no-wind slim disk.

If this benchmark fails, the IMRI minidisk fold may still be a formulation or
numerical issue.

If this benchmark succeeds, the IMRI no-wind branch failure is physically more
meaningful, and moving to wind/stress/time-dependent physics becomes justified.
```

---

# 1. Scientific motivation

The current IMRI/minidisk solver finds that the smooth no-wind hot branch is
blocked by a radial projection fold near the inner region.

Before interpreting that as a physical failure of the no-wind minidisk branch,
we need a control experiment:

```text
Can the same code recover a known no-wind slim disk around a single black hole?
```

The classic slim disk problem is ideal for this because it has the same
essential mathematical structure:

```text
- vertically integrated disk equations;
- non-Keplerian radial momentum;
- advective cooling;
- transonic sonic-point regularity;
- an inner angular momentum eigenvalue;
- outer thin-disk boundary conditions;
- no wind.
```

A single-BH slim disk should be much easier than the IMRI/minidisk problem
because it has no tidal truncation, no stream source, no minidisk boundary, no
outer slope matching to a finite reservoir, and no Hill-sphere geometry.

---

# 2. Primary literature targets

Use these as benchmark references.

## Reference A: Abramowicz et al. 1988

Classic pseudo-Newtonian slim disk.

Key features to reproduce qualitatively:

```text
- Paczynski-Wiita potential;
- vertically integrated slim disk equations;
- radial inertial term;
- radial pressure-gradient term;
- advective heat transport term;
- outer Shakura-Sunyaev-like boundary;
- sonic regularity condition;
- angular momentum at the horizon as an eigenvalue.
```

Recommended role:

```text
Primary pseudo-Newtonian benchmark, closest to the current code architecture.
```

## Reference B: Sadowski 2009 / 2011

Global relativistic slim disk solved by relaxation.

Key features to reproduce qualitatively:

```text
- stationary transonic no-wind slim disk exists over a range of accretion rates;
- inner angular momentum / swallowed angular momentum is an eigenvalue;
- wrong eigenvalues miss the sonic point or follow the wrong branch;
- disk structure becomes non-Keplerian near the inner region;
- advection becomes increasingly important at high accretion rate;
- some non-monotonic inner structure can occur, but global solutions still exist.
```

Recommended role:

```text
Qualitative and topological benchmark for the sonic/eigenvalue structure.
```

## Reference C: Sadowski et al. 2011 vertical-structure slim disks / XSPEC slimbh

Use only as an optional later cross-check.

The public XSPEC `slimbh` documentation says the spectral model is based on
Sadowski et al. radial slim-disk solutions plus TLUSTY vertical structure.

Recommended role:

```text
Optional external sanity check. Not required for the first benchmark.
```

---

# 3. Benchmark hierarchy

Implement the benchmark in layers. Do not try to reproduce the full published
relativistic model in one step.

---

## Benchmark 0: current code sanity tests

Before any new physics, run the existing local functions in a standard
single-BH setup.

Configuration:

```text
M = arbitrary, e.g. 10 Msun or 1e4 Msun
dimensionless units preferred
potential = Paczynski-Wiita
R_PW = 2 r_g
R_ISCO = 6 r_g
no wind
constant Mdot
no stream source
no tidal torque
no minidisk outer boundary
```

Tests:

```text
1. Omega_K derivative matches analytic PW derivative.
2. l_K has minimum at R_ISCO=6 r_g.
3. vertical structure returns positive H, rho, P, tau.
4. Q_visc and Q_rad use a consistent two-face convention.
5. entropy-gradient formula passes manufactured tests.
6. sonic determinant/compatibility diagnostics behave smoothly on test states.
```

Required output:

```text
outputs/tables/slim_benchmark_sanity_tests.md
```

---

## Benchmark 1: low-Mdot thin-disk limit

This is the most important first acceptance test.

Use:

```text
Mdot/Mdot_Edd = 1e-4, 1e-3, 1e-2
alpha = 0.01
potential = Paczynski-Wiita
R_out = 1e4 r_g or larger
no wind
constant Mdot
outer boundary = thin disk
```

Expected behavior:

```text
Omega/Omega_K -> 1 outside inner few r_g
Q_adv/Q_visc << 1
Q_visc ~= Q_rad
H/R << 1
l0 close to l_K(R_ISCO) in the chosen potential
no radial projection fold before R_out
```

Use the analytic thin disk relation as the quantitative benchmark:

```math
\nu\Sigma
=
\frac{\dot M}{3\pi}
\left[
1-\left(\frac{R_0}{R}\right)^{1/2}
\right].
```

Here \(R_0\) should be the effective inner torque radius in the low-Mdot limit.

Acceptance criteria away from the inner boundary and away from the outer ghost
cells:

```text
|Omega/Omega_K - 1| < 1e-3 to 1e-2
|Q_visc - Q_rad| / Q_visc < 1e-3 to 1e-2
|Q_adv/Q_visc| < 1e-2
thin-disk nuSigma agreement < few percent
no phase-space R-turnaround before R_out
```

Required output:

```text
outputs/tables/slim_benchmark_thin_limit.md
outputs/figures/slim_benchmark_thin_limit_profiles.png
```

If this fails:

```text
Stop. Do not proceed to wind or high-Mdot minidisk work.
```

---

## Benchmark 2: pseudo-Newtonian no-wind slim disk sequence

This is the core benchmark.

Use the same equations and closure as the current IMRI solver, but remove all
minidisk complications.

Parameter grid:

```text
Mdot/Mdot_Edd = 0.03, 0.1, 0.3, 1, 3, 10
alpha = 0.01
potential = Paczynski-Wiita
R_out = 1e4 r_g
no wind
constant Mdot
```

Optional alpha scan:

```text
alpha = 0.001, 0.01, 0.1
```

Expected qualitative behavior:

```text
- smooth transonic solutions exist;
- the sonic point lies near the inner region;
- l0 varies smoothly with Mdot;
- advection becomes more important as Mdot increases;
- luminosity grows sublinearly with Mdot at high Mdot;
- H/R increases but should not become pathological for the tested range;
- non-Keplerian effects become important only near the inner region;
- no radial projection fold should block connection to R_out in the standard benchmark.
```

Quantities to record:

```text
Mdot/Mdot_Edd
alpha
R_son/r_g
l0/(r_g c)
max H/R
min tau
L_rad/L_Edd
integrated Qadv/Qvisc
Omega/Omega_K range
number of sonic crossings
number of R-turnarounds in desingularized phase-space flow
physical residual
```

Required output:

```text
outputs/tables/slim_benchmark_pseudo_newtonian_sequence.md
outputs/figures/slim_benchmark_pseudo_newtonian_profiles.png
outputs/figures/slim_benchmark_advective_fraction_vs_mdot.png
outputs/figures/slim_benchmark_luminosity_vs_mdot.png
```

Acceptance criteria:

```text
solutions converge for at least Mdot/Mdot_Edd = 0.03, 0.1, 0.3, 1, 3
Mdot=10 should be attempted but may be flagged if H/R or tau leaves model validity
physical residual <= few x 1e-6 to 1e-5
R_son and l0 vary smoothly with Mdot
desingularized flow does not produce a blocking R-fold before R_out
```

If this benchmark succeeds:

```text
The IMRI fold is likely caused by minidisk-specific boundary/closure physics.
```

If this benchmark fails with a similar fold near \(R\sim6r_g\):

```text
The current equations, stress closure, or numerical formulation are still not
consistent with known no-wind slim disk behavior.
```

---

## Benchmark 3: compare qualitatively to Sadowski-like Schwarzschild solutions

This is not meant to be a high-precision GR reproduction at first. It is a
topological and qualitative comparison.

Use:

```text
spin a = 0 equivalent
alpha = 0.01
Mdot/Mdot_Edd = 0.1, 1, 3, 10
```

If the current code is pseudo-Newtonian:

```text
Do not expect exact agreement with relativistic Sadowski profiles.
Compare only qualitative structure and trends.
```

Qualitative expectations:

```text
- global transonic solution exists;
- sonic-point regularity determines l0;
- wrong l0 misses the sonic point or follows an incorrect branch;
- inner angular momentum and radial velocity depart from thin-disk behavior;
- advective cooling becomes important with increasing Mdot;
- high-Mdot luminosity increases more slowly than linearly with Mdot.
```

Optional reference-curve comparison:

```text
Create literature/reference_curves/README.md.
Digitize selected curves from Sadowski 2009 or Sadowski 2011 manually.
Store as CSV:
    reference_curves/sadowski2009_figure_l_profile.csv
    reference_curves/sadowski2009_figure_vr_profile.csv
    reference_curves/sadowski2009_flux_profile.csv
```

Do not automate figure digitization unless a human has checked the curve labels
and units.

Required output:

```text
outputs/tables/slim_benchmark_sadowski_qualitative.md
outputs/figures/slim_benchmark_sadowski_comparison.png
```

Acceptance:

```text
qualitative trends match;
profiles are smooth and transonic;
no unphysical projection fold appears in the single-BH benchmark.
```

---

# 4. Equations to implement for the pseudo-Newtonian benchmark

Use the current code conventions where possible.

## 4.1 Units

Use dimensionless internal units for the solver:

```math
r_g = GM/c^2,
```

```math
x=\ln(R/r_g).
```

Use:

```math
\dot M_{\rm Edd}
=
\frac{L_{\rm Edd}}{0.1c^2}
```

for the code convention, but always write this convention into every output
table.

Also include conversion to the Abramowicz 1988 convention if needed, because
some older papers use \(\dot M_E=L_E/c^2\) or other efficiency-dependent
definitions.

## 4.2 Paczynski-Wiita potential

```math
\Phi(R)=-\frac{GM}{R-2r_g}.
```

```math
\Omega_K^2
=
\frac{GM}{R(R-2r_g)^2}.
```

```math
R_{\rm ISCO}=6r_g.
```

## 4.3 Continuity

```math
\dot M=2\pi R\Sigma u,
```

where:

```math
u=-v_R>0.
```

## 4.4 Vertical structure

```math
\Sigma=2\rho H.
```

```math
P=P_{\rm gas}+P_{\rm rad}.
```

```math
P_{\rm gas}=\rho\mathcal R T,
```

```math
P_{\rm rad}=\frac{a_{\rm r}T^4}{3}.
```

Use the same vertical hydrostatic closure already used by the IMRI solver.

## 4.5 Angular momentum

For the primary benchmark, use the same algebraic stress closure as the current
transonic code:

```math
W_{R\phi}=\alpha \Pi
```

or the current generalized form:

```math
W_{R\phi}
=
C_\alpha \alpha\,2H\,P_{\rm gas}^{\mu}P_{\rm tot}^{1-\mu}.
```

Baseline:

```text
mu = 0
C_alpha = 1
alpha = 0.01
```

Angular momentum integral:

```math
\dot M(l-l_0)=2\pi R^2 W_{R\phi}.
```

Here \(l_0\) is the eigenvalue.

## 4.6 Radial momentum

Use the same local formulation as the transonic solver:

```math
u\frac{du}{dR}
=
R(\Omega^2-\Omega_K^2)
-
\frac{1}{\Sigma}\frac{d\Pi}{dR}.
```

## 4.7 Energy

No wind:

```math
Q_{\rm visc}=Q_{\rm rad}+Q_{\rm adv}.
```

Use:

```math
Q_{\rm adv}
=
\Sigma v_R T\frac{ds}{dR}.
```

Use the existing first-law entropy expression:

```math
T\frac{ds}{dR}
=
\frac{de}{dR}
-
\frac{P}{\rho^2}\frac{d\rho}{dR}.
```

## 4.8 Radiative cooling

Use electron-scattering diffusion first:

```math
Q_{\rm rad}
=
\frac{16\sigma_{\rm SB}T^4}{3\kappa\Sigma}.
```

This is the two-face convention used in the current code.

Later add free-free opacity or effective optical-depth bridging only after the
benchmark passes.

---

# 5. Solver strategy

Use the current transonic machinery, but isolate it from the IMRI-specific
geometry.

## 5.1 Separate module

Add:

```text
src/imri_qpe/layer3_minidisk_1d/slim_benchmark.py
```

This module should not import stream/tide/wind/minidisk source terms.

Suggested dataclass:

```python
@dataclass(frozen=True)
class StandardSlimBenchmarkParams:
    M_g: float
    mdot_edd_ratio: float
    alpha: float = 0.01
    mu_stress: float = 0.0
    stress_factor: float = 1.0
    R_out_rg: float = 1.0e4
    potential: str = "paczynski_wiita"
    opacity: str = "electron_scattering"
    eddington_eta: float = 0.1
    no_wind: bool = True
```

## 5.2 Initial solution

Begin at:

```text
Mdot/Mdot_Edd = 1e-4
```

using the thin disk analytic solution as the initial guess.

Then continue:

```text
1e-4 -> 3e-4 -> 1e-3 -> 3e-3 -> 1e-2 -> 3e-2
-> 0.1 -> 0.3 -> 1 -> 3 -> 10
```

Use pseudo-arclength only if ordinary continuation stalls.

## 5.3 Sonic regularity

For the standard benchmark, first use the simplest robust method that works:

```text
global relaxation with sonic regularity;
no finite-buffer minidisk sonic patch;
no dynamic patch tied to old IMRI branch.
```

If the existing phase-space/desingularized method is used, it should produce a
monotonic outward branch for the standard disk. If it produces an \(R\)-fold
near \(6r_g\) even in this benchmark, that is a major red flag.

## 5.4 Outer boundary

Use a true asymptotic outer boundary:

```text
R_out = 1e4 r_g or 1e5 r_g
```

At the outer boundary require:

```text
Omega/Omega_K close to 1
Q_adv/Q_visc close to 0
Q_visc = Q_rad
```

Do not use the IMRI finite-radius slope-matching closure for the first
benchmark.

## 5.5 Residuals and diagnostics

For every solution output:

```text
D,C1,C2,K
smin/smax(A)
smin/smax(B)
R_son
l0
physical residual
global energy residual
desingularized R-fold status
outer boundary residuals
```

---

# 6. Reference tests

## Test T1: wrong-eigenvalue behavior

For a fixed \(\dot M\), perturb \(l_0\) around the converged value:

```text
l0 + 1e-3
l0 - 1e-3
l0 + 3e-3
l0 - 3e-3
```

Expected:

```text
wrong l0 either misses the sonic point or follows an incorrect branch.
```

This is an important qualitative test because published slim-disk work
emphasizes that \(l_0\) is selected by global sonic regularity.

Output:

```text
outputs/tables/slim_benchmark_wrong_l0_test.md
outputs/figures/slim_benchmark_wrong_l0_test.png
```

## Test T2: luminosity saturation / advection

Compute:

```math
L_{\rm rad}
=
\int 2\pi R Q_{\rm rad}\,dR.
```

Compute:

```math
L_{\rm visc}
=
\int 2\pi R Q_{\rm visc}\,dR.
```

Compute:

```math
f_{\rm adv}
=
\frac{\int 2\pi R Q_{\rm adv}\,dR}
{\int 2\pi R Q_{\rm visc}\,dR}.
```

Expected:

```text
L_rad grows more slowly than Mdot at high Mdot;
f_adv increases with Mdot.
```

Output:

```text
outputs/tables/slim_benchmark_luminosity_advection.md
outputs/figures/slim_benchmark_luminosity_advection.png
```

## Test T3: radial structure profiles

For each Mdot, plot:

```text
Sigma(R)
T(R)
u/c
Omega/Omega_K
H/R
Qrad/Qvisc
Qadv/Qvisc
l/l_K
```

Output:

```text
outputs/figures/slim_benchmark_radial_profiles_mdot_*.png
```

## Test T4: no-fold test

Run the desingularized phase-space flow along the converged solution and verify:

```text
no blocking R-fold before R_out
```

A local wiggle is acceptable only if the final solution remains a single-valued
radial disk profile and the BVP crosses it properly.

Output:

```text
outputs/tables/slim_benchmark_phase_space_fold_audit.md
```

---

# 7. Published comparison layer

## 7.1 Without digitized curves

Compare these qualitative statements:

```text
- slim disk is advective and optically thick;
- transonic eigenvalue problem determines l0;
- low-Mdot limit resembles thin disk;
- high-Mdot disks advect more heat;
- inner profiles are non-Keplerian;
- relaxation/global solution exists.
```

This comparison is acceptable for first validation.

## 7.2 With digitized curves

If reference curves are manually digitized, add CSV files:

```text
literature/reference_curves/sadowski2009_l_profile.csv
literature/reference_curves/sadowski2009_radial_velocity.csv
literature/reference_curves/sadowski2009_flux.csv
```

Include metadata:

```text
paper
figure
spin
alpha
Mdot or luminosity convention
axis units
digitization tool
date
human checker
```

Then compare:

```text
profile shape
inner radius behavior
sonic location order
relative trend with Mdot
```

Do not require exact agreement unless the same GR equations and units have been
implemented.

---

# 8. Decision tree

## Case A: benchmark passes

If the standard no-wind slim benchmark passes through \(\dot M/\dot M_{\rm Edd}
\sim1\) and higher without a blocking fold:

```text
The IMRI/minidisk no-wind failure is likely caused by minidisk-specific
boundary geometry or closure, not by the core transonic machinery.
```

Then proceed to:

```text
prescribed wind topology scan in the IMRI model.
```

## Case B: benchmark fails with same \(R\simeq6r_g\) fold

If the standard single-BH benchmark also folds near \(6r_g\):

```text
The current equations, stress closure, sonic regularity implementation, or
phase-space continuation are inconsistent with published no-wind slim disks.
```

Then fix the benchmark before adding wind.

Likely suspects:

```text
stress normalization
two-face vs one-face flux convention
radial momentum sign
angular momentum integral sign
pseudo-Newtonian derivative
vertical pressure integral
entropy/advection sign
sonic compatibility scaling
far boundary condition
```

## Case C: benchmark passes at low Mdot but fails at high Mdot only

Check:

```text
H/R
tau_eff
Q_adv/Qvisc
Omega/Omega_K
```

If the model leaves the validity regime, flag the high-Mdot benchmark as
outside the no-wind height-integrated assumptions.

Then proceed cautiously to wind.

## Case D: benchmark only works with a different stress closure

If:

```text
mu = 0 fails
mu = 1 works
```

or vice versa, the IMRI problem should scan stress closure before wind.

---

# 9. Concrete scripts to add

```text
scripts/run_standard_slim_benchmark_thin_limit.py
scripts/run_standard_slim_benchmark_sequence.py
scripts/run_standard_slim_wrong_l0_test.py
scripts/run_standard_slim_luminosity_advection.py
scripts/run_standard_slim_phase_space_fold_audit.py
scripts/plot_standard_slim_benchmark_profiles.py
```

Outputs:

```text
outputs/tables/slim_benchmark_sanity_tests.md
outputs/tables/slim_benchmark_thin_limit.md
outputs/tables/slim_benchmark_pseudo_newtonian_sequence.md
outputs/tables/slim_benchmark_wrong_l0_test.md
outputs/tables/slim_benchmark_luminosity_advection.md
outputs/tables/slim_benchmark_phase_space_fold_audit.md

outputs/figures/slim_benchmark_thin_limit_profiles.png
outputs/figures/slim_benchmark_pseudo_newtonian_profiles.png
outputs/figures/slim_benchmark_advective_fraction_vs_mdot.png
outputs/figures/slim_benchmark_luminosity_vs_mdot.png
outputs/figures/slim_benchmark_wrong_l0_test.png
```

---

# 10. Suggested code structure

```text
src/imri_qpe/layer3_minidisk_1d/slim_benchmark.py
src/imri_qpe/layer3_minidisk_1d/slim_benchmark_reference.py
tests/test_standard_slim_benchmark.py
```

Functions:

```python
def make_standard_slim_params(...):
    ...

def initial_thin_disk_guess(params):
    ...

def solve_standard_slim_disk(params, initial_guess=None):
    ...

def continue_standard_slim_sequence(mdot_values, params):
    ...

def thin_disk_analytic_profiles(params):
    ...

def compare_to_thin_disk(profile, analytic):
    ...

def slim_benchmark_diagnostics(profile):
    ...

def write_slim_benchmark_tables(results):
    ...
```

---

# 11. Codex implementation order

## Sprint 1: low-Mdot thin benchmark

Implement:

```text
Mdot = 1e-4, 1e-3, 1e-2
```

and pass the thin disk checks.

Do not proceed until this works.

## Sprint 2: no-wind slim sequence

Continue:

```text
0.03, 0.1, 0.3, 1, 3
```

and produce the main sequence table.

## Sprint 3: high-Mdot extension

Try:

```text
Mdot = 10
```

only after the lower sequence works.

## Sprint 4: published qualitative comparison

Add plots and optional digitized references.

## Sprint 5: return to IMRI decision

Only then decide whether to add wind to the IMRI model.

---

# 12. Compact Codex prompt

```text
Before adding wind to the IMRI minidisk model, benchmark the transonic solver
against a standard single-BH no-wind slim disk.

Implement a separate standard_slim_benchmark module with Paczynski-Wiita
potential, no stream/tide/wind, constant Mdot, outer thin-disk boundary at
Rout=1e4 rg, and the same gas+radiation, radial momentum, angular momentum,
and energy equations as the current solver.

Run:
    Mdot/Mdot_Edd = 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1, 3, 10
    alpha = 0.01
    no wind

First recover the thin disk:
    Omega/OmegaK ~ 1
    Qadv/Qvisc << 1
    Qvisc ~ Qrad
    nuSigma matches analytic thin disk
    no R-fold before Rout

Then recover slim-disk trends:
    smooth transonic solution exists
    l0 is an eigenvalue
    wrong l0 misses sonic regularity
    advective fraction increases with Mdot
    luminosity grows sublinearly with Mdot
    no blocking radial projection fold before Rout

If the standard benchmark also folds near 6 rg, stop and fix the transonic
formulation. If the benchmark passes but the IMRI minidisk fails, then the IMRI
failure is physically meaningful and the next step is a wind topology scan.
```

---

# 13. Bottom line

This benchmark is the cleanest way to decide whether the current obstruction
is:

```text
a real IMRI/minidisk no-wind physical obstruction
```

or

```text
a remaining defect in the transonic solver.
```

Do this before adding wind.
