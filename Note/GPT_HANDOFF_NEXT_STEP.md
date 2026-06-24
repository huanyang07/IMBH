# GPT Handoff: IMRI QPE Minidisk Advective Branch Next Step

Status note: this handoff is now superseded by the repaired Sprint-B isolated
solver results. Its main failure table described a pre-repair
solver/convention artifact near `Mdot/Mdot_Edd = 0.03`. Current results are in
`outputs/tables/isolated_slim_branch_summary.md` and show a valid reduced
thin/moderate advective sequence through `Mdot/Mdot_Edd ~= 10`, with the QPE
target still outside the nearly Keplerian model's validity because `H/R`
becomes large. The next step is a transonic slim-disk solver with radial
momentum and sonic regularity.

This is a focused handoff packet for getting scientific/numerical advice on
the next step of the IMRI QPE minidisk model.

## 1. Project Goal

We are testing whether a circumsecondary minidisk around an IMBH embedded in a
TDE disk can produce QPE-like limit cycles.

Fiducial system:

```text
M_SMBH = 1e6 Msun
M2     = 1e4 Msun
a      = 50 r_g,Smbh
R_u    = 0.3 R_H
```

The one-zone target burst rate from the current model is:

```text
Mdot_burst ~= 0.02445 Msun/yr
Mdot_burst/Mdot_Edd,2 ~= 94
```

## 2. Current Scientific Status

The early local S-curve calculation produced a stable hot branch only after
adding an imposed local advective term:

```text
Q_adv = xi * Mdot * P / (2 pi R^2 rho)
```

That was encouraging but not physically sufficient, because `xi` was an input.
The key requirement is to compute:

```text
Q_adv = Sigma v_R T ds/dR
xi_eff = - R rho/P * T ds/dR
```

from a radial disk solution.

## 3. What Has Been Implemented

### 3.1 Entropy advection

File:

```text
src/imri_qpe/layer3_minidisk_1d/entropy_advection.py
```

Implemented:

```text
T ds/dR = de/dR - P/rho^2 d rho/dR
```

for a gas+radiation mixture with:

```text
P = P_gas + P_rad
e = P_gas/((gamma_gas - 1) rho) + a_rad T^4/rho
```

Also implemented an independent log-gradient formula:

```text
T ds/dR =
Rgas T [ 1/(gamma-1) dlnT/dR - dlnrho/dR ]
+ 4 a_rad T^4/rho [ dlnT/dR - (1/3) dlnrho/dR ]
```

Manufactured tests pass for gas-isentropic and radiation-isentropic profiles.

### 3.2 Global audit of stitched local hot roots

File:

```text
src/imri_qpe/layer3_minidisk_1d/global_slim.py
```

The global evaluator computes:

```text
v_R from a reduced Keplerian angular-momentum equation
Mdot = -2 pi R Sigma v_R
Q_adv = Sigma v_R T ds/dR
Q_rad
Q_wind
energy residual = Q_visc - Q_rad - Q_adv - Q_wind
```

Output summary:

```text
outputs/tables/global_slim_audit_hardened.md
outputs/figures/global_slim_audit_hardened.png
```

Hardened audit result:

| N | stencil | signed | L1 | median xi_eff | int Qadv/Qvisc |
|---:|---|---:|---:|---:|---:|
| 12 | limited | -4.62 | 4.62 | 2.16 | 5.62 |
| 24 | limited | -4.80 | 4.80 | 2.56 | 5.67 |
| 48 | limited | -5.13 | 5.13 | 2.59 | 6.05 |
| 48 | centered | -5.39 | 5.39 | 2.71 | 6.19 |

Interpretation:

```text
The stitched local xi=0.3 hot roots are not a self-consistent global solution.
This is not just signed cancellation, one bad boundary cell, or the slope limiter.
```

### 3.3 Fixed-Sigma relaxation cross-check

The code can relax `T(R)` at fixed `Sigma(R)` to close the formal energy
residual, but this is not a valid disk solution.

Result:

```text
signed residual ~= 0
L1 residual     ~= 0
median xi_eff   ~= 2.69
int Qadv/Qvisc  ~= -25.5
Mdot range      ~= -0.214 to 0.187 Msun/yr
max H/R         ~= 1.18
```

Interpretation:

```text
Energy balance can be forced at fixed Sigma, but the result is not a physical
steady hot branch because mass/angular-momentum consistency and radial
regularity are not solved.
```

### 3.4 Sprint B isolated no-wind benchmark

File:

```text
src/imri_qpe/layer3_minidisk_1d/isolated_slim_solver.py
```

This benchmark turns off:

```text
stream source
tidal torque
wind
```

It imposes constant `Mdot`, assumes nearly Keplerian rotation, solves the
steady angular-momentum closure for `Sigma(R)`, and relaxes `T(R)` against:

```text
Q_visc = Q_rad + Q_adv
```

Angular-momentum closure used:

```text
Mdot (l - l_in) = 2 pi R^2 W
W = (3/2) nu Sigma Omega_K
nu = alpha H^2 Omega_K
```

Radial velocity:

```text
v_R = - Mdot / (2 pi R Sigma)
```

Output summary:

```text
outputs/tables/isolated_slim_branch_summary.md
outputs/figures/isolated_slim_branch_continuation.png
```

Pre-repair continuation result:

| Mdot/Mdot_Edd | converged | L1 | max residual | int Qadv/Qvisc | max H/R | message |
|---:|:---:|---:|---:|---:|---:|---|
| 0.01 | yes | 3.386e-04 | 3.617e-04 | -1.965e-05 | 4.393e-03 | converged |
| 0.03 | no | 0.472 | 0.385 | -2.633e-04 | 8.235e-03 | maximum iterations reached |
| 0.1 | no | nan | nan | nan | nan | no angular-momentum Sigma root at cell 1 |
| 0.3 | no | 1.08 | 0.994 | -0.0514 | 0.1 | maximum iterations reached |
| 1 | no | 0.818 | 0.997 | -0.0513 | 0.1 | maximum iterations reached |
| 3 | no | 0.97 | 0.998 | -0.0248 | 0.0698 | maximum iterations reached |
| 10 | no | 0.996 | 0.995 | -7.402e-04 | 0.0148 | line search failed |
| 30 | no | nan | nan | nan | nan | no angular-momentum Sigma root at cell 1 |
| 100 | no | 1 | 0.999 | -1.224e-03 | 0.019 | line search failed |

Interpretation:

```text
Before the repair, the isolated nearly Keplerian no-wind benchmark found the
low-rate thin disk but did not find a clean no-wind advective branch toward the
QPE target Mdot/Mdot_Edd ~= 94. This table is superseded by the repaired
summary in outputs/tables/isolated_slim_branch_summary.md.
```

## 4. Important Caveats

The current isolated solver is not a full slim-disk solver.

It does **not** solve:

```text
radial momentum
non-Keplerian Omega(R)
sonic point regularity
transonic inner boundary condition
eigenvalue/inner torque condition
```

It assumes:

```text
Omega = Omega_K
W = (3/2) nu Sigma Omega_K
constant imposed Mdot
Keplerian angular-momentum integral
```

The failure of this benchmark may therefore mean one of two things:

```text
1. the physical no-wind isolated branch really is absent under these assumptions;
2. the assumptions are too restrictive and a faithful slim disk requires radial
   momentum and sonic regularity.
```

## 5. Files To Read

Please read these first:

```text
README.md
Note/CODEX_GLOBAL_SLIM_NEXT_STEPS.md
outputs/tables/global_slim_audit_hardened.md
outputs/tables/isolated_slim_branch_summary.md
src/imri_qpe/layer3_minidisk_1d/isolated_slim_solver.py
src/imri_qpe/layer3_minidisk_1d/global_slim.py
src/imri_qpe/layer3_minidisk_1d/entropy_advection.py
```

Helpful figures:

```text
outputs/figures/global_slim_audit_hardened.png
outputs/figures/isolated_slim_branch_continuation.png
```

Relevant literature in `Literature/` if needed:

```text
Literature/1988ApJ...332..646A          # Abramowicz et al. slim disks
Literature/Pan_2022_ApJL_928_L18.pdf    # QPE disk-instability context
Literature/2211.00704v1.pdf             # TDE/QPE disk advection/magnetic pressure
Literature/Jiang_2013_ApJ_778_65.pdf    # radiation-MHD stability cautions
Literature/Hirose_2009_ApJ_691_16.pdf   # radiation-pressure stability cautions
```

## 6. Questions For GPT

Please advise on the next scientific/numerical step.

Specific questions:

1. Is the Sprint-B failure likely caused by missing radial momentum and sonic
   regularity, or by a mistake/over-restriction in the Keplerian angular
   momentum closure?

2. Is the isolated benchmark currently implemented in
   `isolated_slim_solver.py` a valid reduced slim-disk test, or is it too
   restrictive to expect an advective branch?

3. What is the minimum faithful isolated slim-disk solver we should implement
   next?

4. Should the next solver be a boundary-value problem in

   ```text
   Sigma(R), T(R), v_R(R), Omega(R)
   ```

   with radial momentum included?

5. What equations should be solved for a one-temperature vertically integrated
   slim disk around the IMBH?

6. What inner and outer boundary conditions are appropriate?

   In particular:

   ```text
   inner sonic point or ISCO condition?
   zero torque or eigenvalue torque?
   outer fixed Mdot and angular momentum?
   outer temperature or entropy condition?
   ```

7. Are there simpler diagnostic tests before implementing the full transonic
   solver?

8. Should stream feeding, tidal truncation, and wind remain off until the
   isolated solver passes?

9. If a no-wind isolated branch is absent, should the next step be:

   ```text
   A. full transonic slim disk,
   B. stream-fed/tidally truncated disk,
   C. wind-regulated disk,
   D. time-dependent high-state calculation without requiring a steady branch?
   ```

10. What would be clear go/no-go criteria after the next implementation?

## 7. Desired Answer Format

Please respond with:

```text
1. Diagnosis of the current failure.
2. The next solver equations to implement.
3. Boundary conditions.
4. Numerical method recommendation.
5. Minimal validation tests.
6. Whether to delay stream/tide/wind.
7. Any literature equations or references we should follow exactly.
```
