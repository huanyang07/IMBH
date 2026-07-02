# Codex Plan: High-\(\dot M\) No-Wind Benchmark, Corrected Stream Loading, and Finite-Minidisk Limit-Cycle Roadmap

Repository target:

```text
https://github.com/huanyang07/IMBH
```

Purpose:

```text
Update the implementation plan after realizing that the current stream-fed
run at fixed Mdot_inner/Edd = 1 is not the same physical test as the
Xue/Sadowski-type high-state slim-disk limit cycle.
```

The central point is:

```text
The current stream-source run puffs a local outer annulus but does not drive
the inner disk into a high-Mdot advective state, because Mdot_inner is fixed.
```

Therefore, before adding wind as the next physics layer, Codex should first
test whether the solver can recover the strongly advective no-wind upper branch
in a standard slim-disk benchmark and then test whether a finite stream-fed
minidisk can access that branch through mass loading.

---

# 1. Physical interpretation to preserve

The target QPE mechanism is conceptually close to the radiation-pressure/slim-disk limit cycle:

```text
cool low state
-> stream-fed mass accumulation
-> radiation-pressure unstable transition
-> jump to hot advective high-Mdot branch
-> rapid draining
-> return to cool state
-> reload
```

In symbols:

```math
\dot M_{\rm cap} > \dot M_{\rm low}
```

during loading, so the disk mass grows:

```math
\frac{dM_d}{dt}
\simeq
\dot M_{\rm cap}-\dot M_{\rm low}.
```

At ignition:

```math
\Sigma \to \Sigma_{\max}
```

and the disk jumps to a high state with:

```math
\dot M_{\rm high} \gg \dot M_{\rm cap}.
```

During the burst:

```math
\frac{dM_d}{dt}
\simeq
\dot M_{\rm cap}-\dot M_{\rm high}<0.
```

The burst ends when the disk drops to:

```math
\Sigma \to \Sigma_{\min}.
```

This is the desired limit-cycle picture.

However, the current stream-fed steady run is not yet testing this, because it
keeps:

```text
Mdot_inner/Edd = 1
```

fixed, and adds stream heating/forcing mainly at:

```text
R ~ 100 rg
```

for the current \(R_{\rm out}=300r_g\), \(R_{\rm inj}/R_{\rm out}=0.4\) setup.

That setup tests:

```text
outer stream heating at fixed inner accretion rate,
```

not:

```text
inner high-Mdot advective outburst.
```

---

# 2. Updated high-level sequence

The revised plan is:

```text
A. Standard high-Mdot no-wind benchmark.
B. Correct source/sink bookkeeping.
C. Corrected no-wind stream-loading stress test.
D. Finite-minidisk equilibrium and stability map.
E. Wind only if no-wind high branch is absent or physically incomplete.
F. Time-dependent limit-cycle simulation only after branch/stability map.
```

Do not jump directly from the current \(Mdot_inner/Edd=1\) stream-heated
annulus result to a full wind model.

---

# 3. Sprint A: standard high-\(\dot M\) no-wind benchmark

## 3.1 Motivation

The standard no-wind benchmark has recovered a mild slim disk through
near-Eddington conditions. But the Xue/Sadowski high state is expected to be a
stronger advective state, often associated with transient inner accretion rates
above the mean supply.

Therefore Codex should push the standard single-BH no-wind benchmark to:

```text
Mdot/Edd = 2, 3, 5, 10
```

before interpreting the low advective fraction in the current stream-fed
experiment.

## 3.2 Configuration

Use the mature standard slim setup:

```text
potential = Paczynski-Wiita
outer closure = pressure-supported
adaptive outer mesh = yes
no stream
no wind
constant Mdot
```

Recommended ladder:

```text
0.5 -> 0.7 -> 0.9 -> 1.0 -> 1.5 -> 2 -> 3 -> 5 -> 10
```

Use small adaptive log-\(\dot M\) steps, with strict/scout classification.

## 3.3 Required diagnostics

Do not rely only on the signed global advective fraction.

For each solution compute:

```math
f_{\rm adv,global}
=
\frac{\int Q_{\rm adv}\,2\pi R\,dR}
{\int Q_{\rm visc}\,2\pi R\,dR}.
```

Also compute:

```math
f_{\rm adv,inner}
=
\frac{\int_{R<20r_g} Q_{\rm adv}\,2\pi R\,dR}
{\int_{R<20r_g} Q_{\rm visc}\,2\pi R\,dR}.
```

And:

```math
f_{\rm adv,pos}
=
\frac{\int \max(Q_{\rm adv},0)\,2\pi R\,dR}
{\int Q_{\rm visc}\,2\pi R\,dR}.
```

Also track:

```text
Lrad/LEdd
Lvisc/LEdd
max H/R
min tau_eff or tau_es
Rson
lambda0/lK_ISCO
number of radial folds
dominant residual block
```

Why:

```text
The signed global integral can hide local inner advection because outer
radiative regions dilute the integral and sign changes can cancel.
```

## 3.4 Success criteria

A standard high-\(\dot M\) no-wind upper branch is considered recovered if:

```text
1. smooth transonic solutions exist at Mdot/Edd = 2, 3, and preferably 5;
2. f_adv_inner grows with Mdot;
3. Lrad grows sublinearly with Mdot;
4. no unphysical radial projection fold blocks the solution before Rout;
5. H/R remains within the intended height-integrated validity regime;
6. residuals and sonic diagnostics pass mesh checks.
```

A strong warning flag:

```text
If Mdot/Edd = 5--10 still has f_adv_inner << 0.1 and no luminosity saturation,
audit Qadv sign, entropy-gradient calculation, stress normalization, and
Eddington-rate convention.
```

## 3.5 New scripts

```text
scripts/run_standard_slim_high_mdot_no_wind_ladder.py
scripts/run_standard_slim_high_mdot_advection_diagnostics.py
scripts/run_standard_slim_high_mdot_residual_profile.py
```

## 3.6 New outputs

```text
outputs/tables/slim_benchmark_high_mdot_no_wind_ladder.md
outputs/tables/slim_benchmark_high_mdot_advection_diagnostics.md
outputs/tables/slim_benchmark_high_mdot_residual_profiles.md

outputs/figures/slim_benchmark_high_mdot_advective_fraction.png
outputs/figures/slim_benchmark_high_mdot_luminosity.png
outputs/figures/slim_benchmark_high_mdot_profiles.png
```

---

# 4. Sprint B: correct source/sink bookkeeping

## 4.1 Problem

The current stream-source run uses a radial profile in which:

```text
Mdot_out/Mdot_in > 1
```

as the stream strength increases.

With inward-positive accretion rate:

```math
\dot M=-2\pi R\Sigma v_R>0,
```

steady continuity with stream source \(S_\Sigma\) and wind sink
\(\dot\Sigma_w\) is:

```math
\frac{d\dot M}{d\ln R}
=
\dot M'_w-\dot M'_s,
```

where:

```math
\dot M'_s = 2\pi R^2 S_\Sigma \ge 0,
```

```math
\dot M'_w = 2\pi R^2 \dot\Sigma_w \ge 0.
```

Therefore:

```text
positive dMdot/dlnR corresponds to wind/recycling loss dominating,
not to a conservative stream source.
```

A pure no-wind stream source has:

```math
\dot M'_w=0,
```

so:

```math
\frac{d\dot M}{d\ln R}=-\dot M'_s<0
```

in the source annulus.

That means the inner accretion rate is larger than the accretion rate outside
the source annulus.

## 4.2 Implementation requirement

Separate these concepts explicitly:

```text
stream_source_prime(logR)  # positive mass added to disk per dlnR
wind_sink_prime(logR)      # positive mass removed from disk per dlnR
mdot_profile_from_source_sink(logR)
```

Do not use one parameter such as `stream_mass_fraction` to mean both stream
addition and outward mass excess.

## 4.3 Mass budget diagnostic

For every source/sink run, output:

```math
\Delta \dot M_{\rm budget}
=
\dot M_{\rm out}
-
\dot M_{\rm in}
-
\int(\dot M'_w-\dot M'_s)d\ln R.
```

Acceptance:

```text
relative mass budget error < 1e-6 for prescribed profiles
relative mass budget error < 1e-3 for coupled wind/source solves
```

## 4.4 Stream heating convention audit

The current stream heating should be audited for one-face versus two-face
normalization.

Use the convention:

```text
Qvisc, Qrad, Qadv, Qstream, Qwind are all two-face vertically integrated
surface energy rates.
```

Then require:

```math
P_{\rm stream}
=
\int 2\pi R Q_{\rm stream}\,dR.
```

Add a table:

```text
outputs/tables/stream_source_sink_bookkeeping_audit.md
```

---

# 5. Sprint C: corrected no-wind stream-loading stress test

## 5.1 Purpose

Test whether a physically signed stream source, possibly with higher inner
accretion rate, can access the no-wind advective branch before adding wind.

The current test does not answer this because it fixes:

```text
Mdot_inner/Edd = 1
```

and deposits heat in the outer annulus.

## 5.2 Configuration

Use physical minidisk radius:

```text
Rout = 300 rg
```

Use pressure-supported outer closure and slope-Picard as already validated.

Run:

```text
Mdot_inner/Edd = 1, 2, 3, 5
```

Stream source strength:

```text
f_s = 0, 0.1, 0.3
```

Stream injection location:

```text
Rinj/Rout = 0.2, 0.3, 0.4
```

Heating efficiency:

```text
eta_heat = 0, 1, 3, 10
```

Angular momentum offset:

```text
delta_l = -0.1, 0, +0.1
```

Use true source sign:

```text
Mdot_inner > Mdot_outer
```

for no-wind stream addition.

## 5.3 Diagnostics

For each model output:

```text
Mdot_inner/Edd
Mdot_outer/Edd
total stream source rate
Rinj
eta_heat
delta_l
Rson
lambda0
max H/R
max Qstream/Qvisc
int Qstream/Qvisc
f_adv_global
f_adv_inner
f_adv_pos
Lrad/LEdd
residual
dominant block
```

## 5.4 Success criteria

No-wind stream loading is promising if:

```text
f_adv_inner increases strongly with Mdot_inner;
Lrad grows sublinearly at high Mdot_inner;
the solution remains smooth and mesh-converged;
stream source does not merely puff a local annulus.
```

No-wind stream loading is not enough if:

```text
even at Mdot_inner/Edd = 3--5, f_adv_inner remains small
and the disk remains radiative except for a local stream-heated bump.
```

---

# 6. Sprint D: finite-minidisk equilibrium map

This is the first actual limit-cycle step.

## 6.1 Why this is needed

A limit cycle is not proven by finding one steady hot branch.

We need an equilibrium family in a loading variable such as:

```text
disk mass
outer surface density
Mdot_cap
source normalization
```

The steady solver can find stable and unstable branches. Stability must be
diagnosed separately.

## 6.2 Suggested control parameters

For a first equilibrium map, use:

```text
Rout = 300 rg
M2 fixed
alpha fixed
stream annulus fixed
no wind initially
```

Vary:

```text
Mdisk proxy or Sigma_out proxy
```

Operationally, Codex can vary the normalization of the stream-fed source or
the target surface-density/entropy normalization and record the resulting
inner accretion rate.

Suggested scanned variable:

```text
outer entropy/surface-density parameter theta_load
```

or:

```text
source normalization Mdot_cap
```

But output the physically meaningful quantity:

```math
M_d =
\int 2\pi R\Sigma\,dR.
```

## 6.3 Required outputs

For each equilibrium:

```text
M_d
Mdot_inner
Mdot_cap or source normalization
Sigma_out
Rson
lambda0
max H/R
Lrad
f_adv_inner
f_adv_global
residual
```

Plot:

```text
Mdot_inner versus M_d
Mdot_inner versus Sigma_out
Lrad versus M_d
f_adv_inner versus M_d
```

## 6.4 Stability labels

Local thermal stability diagnostic:

```math
\left.
\frac{\partial(Q^+-Q^-)}
{\partial T}
\right|_\Sigma.
```

Use:

```text
stable if derivative < 0
unstable if derivative > 0
```

where:

```math
Q^- = Qrad + Qadv
```

in no-wind tests, and later:

```math
Q^- = Qrad + Qadv + Qwind.
```

Also perform a global perturbation test for selected equilibria:

```text
increase T by 1 percent in the suspected unstable region
decrease T by 1 percent
re-polish or integrate a local thermal relaxation
record whether perturbation grows or decays
```

## 6.5 Goal

Identify:

```text
cool stable branch
unstable middle/radiation-pressure branch
hot advective branch
upper and lower turning points
```

The needed quantities for the limit cycle are:

```math
\Sigma_{\max},\qquad \Sigma_{\min},
```

or globally:

```math
M_{d,\max},\qquad M_{d,\min}.
```

Then:

```math
\Delta M_{\rm cyc}
=
M_{d,\max}-M_{d,\min}.
```

---

# 7. Sprint E: wind only after the no-wind high branch test

Add wind if either condition is true:

```text
1. standard high-Mdot no-wind branch becomes advective but finite stream-fed
   no-wind minidisk cannot access it;

2. finite stream-fed no-wind high branch exists but is too radiative / too
   luminous / physically super-Eddington without mass loss.
```

Use minimal wind in two stages.

## 7.1 Prescribed wind topology scan

Purpose:

```text
Does inward mass loss change the branch topology in the right direction?
```

Use:

```math
\frac{d\dot M}{d\ln R}
=
\dot M'_w-\dot M'_s.
```

Scan prescribed wind profiles:

```text
f_w = 0, 0.3, 1, 3
R_w/Rout = 0.2, 0.4, 0.7
log width = 0.2, 0.4
```

Evaluate:

```text
f_adv_inner
f_nonrad = (Qadv+Qwind)/Qvisc
Lrad saturation
Mdot_out/Mdot_in
fold removal if any
```

## 7.2 Energy-limited wind

Energy equation:

```math
Q_{\rm visc}+Q_{\rm stream}
=
Q_{\rm rad}
+
Q_{\rm adv}
+
Q_{\rm wind}.
```

Use:

```math
Q_{\rm wind}
=
\epsilon_w
\left[
Q_{\rm heat}
-
Q_{\rm adv}
-
Q_{\rm Edd,z}
\right]_+.
```

where:

```math
Q_{\rm heat}=Q_{\rm visc}+Q_{\rm stream},
```

and:

```math
Q_{\rm Edd,z}
=
\frac{2c\Omega_K^2H}{\kappa}.
```

Wind mass loss per log radius:

```math
\dot M'_w =
\frac{2\pi R^2Q_{\rm wind}}{E_w}.
```

Use:

```math
E_w =
\chi_{\rm esc}\frac{GM}{2R}
+
\frac12 v_\infty^2.
```

Initial scan:

```text
epsilon_w = 0.1, 0.3, 1
chi_esc = 1
v_inf = 0
l_w = l
```

Later angular momentum lever arm:

```text
l_w = lambda_w l
lambda_w = 1.2, 1.5, 2
```

---

# 8. Updated implementation priority

Use this exact sequence:

```text
1. Standard high-Mdot no-wind benchmark to Mdot/Edd = 2,3,5,10.
2. Source/sink sign and energy bookkeeping audit.
3. Corrected no-wind stream-loading stress test with true source sign and
   Mdot_inner = 1,2,3,5.
4. Finite minidisk equilibrium map and stability labels.
5. Wind topology scan only if no-wind high branch is absent/incomplete.
6. Time-dependent limit-cycle simulation only after equilibrium branches and
   stability are known.
```

---

# 9. Concrete scripts to add

## Standard high-\(\dot M\)

```text
scripts/run_standard_slim_high_mdot_no_wind_ladder.py
scripts/run_standard_slim_high_mdot_advection_diagnostics.py
```

Outputs:

```text
outputs/tables/slim_benchmark_high_mdot_no_wind_ladder.md
outputs/tables/slim_benchmark_high_mdot_advection_diagnostics.md
```

## Bookkeeping

```text
scripts/run_stream_source_sink_bookkeeping_audit.py
```

Output:

```text
outputs/tables/stream_source_sink_bookkeeping_audit.md
```

## Corrected no-wind source tests

```text
scripts/run_stream_corrected_nowind_source_stress_test.py
```

Output:

```text
outputs/tables/stream_corrected_nowind_source_stress_test.md
```

## Equilibrium map

```text
scripts/run_finite_minidisk_equilibrium_map.py
scripts/run_finite_minidisk_stability_labels.py
```

Outputs:

```text
outputs/tables/finite_minidisk_equilibrium_map.md
outputs/tables/finite_minidisk_stability_labels.md
outputs/figures/finite_minidisk_s_curve.png
```

## Wind, later

```text
scripts/run_stream_prescribed_wind_topology_scan.py
scripts/run_stream_energy_limited_wind_scan.py
```

---

# 10. Acceptance criteria before claiming progress toward a limit cycle

Do not claim a limit cycle just from a hot steady branch.

Require:

```text
1. an equilibrium map with at least two stable branches or a transient hot
   draining state;
2. a thermally/viscously unstable region or demonstrated runaway;
3. upper and lower transition thresholds;
4. an estimate of Delta M_cyc;
5. a time-dependent run that cycles.
```

The first necessary output is:

```text
finite_minidisk_s_curve.png
```

showing:

```text
Mdot_inner versus Mdisk or Sigma_out,
with stable/unstable labels.
```

---

# 11. Compact Codex prompt

```text
Update the plan after realizing the current f_m stream-source test at fixed
Mdot_inner/Edd=1 is not the Xue/Sadowski high state. It only heats/puffs an
outer annulus near R~100 rg and does not drive high inner Mdot.

Next:
1. First push the standard no-wind benchmark to Mdot/Edd=2,3,5,10 using the
   pressure-supported closure and adaptive mesh. Diagnose local inner advection:
       f_adv_inner = int_{R<20rg} Qadv / int_{R<20rg} Qvisc
       f_adv_pos = int max(Qadv,0) / int Qvisc
       Lrad/LEdd
   Do not rely only on signed global Qadv/Qvisc.

2. Fix source/sink bookkeeping:
       dMdot/dlnR = Mdot_wind_prime - Mdot_stream_prime.
   A no-wind stream source must make Mdot_inner > Mdot_outer.

3. Run corrected no-wind stream-loading tests at Rout=300 rg with
       Mdot_inner/Edd = 1,2,3,5
       f_s = 0.1,0.3
       Rinj/Rout = 0.2,0.3,0.4
       eta_heat = 0,1,3,10
       delta_l = -0.1,0,+0.1.

4. Build a finite-minidisk equilibrium map parameterized by Mdisk or Sigma_out.
   Output Mdot_inner(Mdisk), Lrad(Mdisk), f_adv_inner(Mdisk), and stability
   labels from d(Q+ - Q-)/dT|Sigma.

5. Add wind only after no-wind high branch topology is tested. Start with a
   prescribed wind topology scan, then energy-limited wind with
       Qvisc+Qstream = Qrad+Qadv+Qwind.
```

---

# 12. Bottom line

The current stream-fed no-wind run does not contradict the Xue/Sadowski
limit-cycle picture. It has not generated the corresponding high state because
the inner accretion rate was held fixed near Eddington.

The next test is:

```text
Can the code recover a strongly advective high-Mdot no-wind branch when
Mdot_inner is actually driven above Eddington?
```

Then:

```text
Can a finite stream-fed minidisk access that branch through mass loading?
```

Only then should wind become the primary physics layer.
