# Codex Mdot=5 Shen-Diagnostic Sprint Results

Date: 2026-07-05

This sprint implements the first diagnostic layer requested after the Shen &
Matzner update.  It does not change the solver equations.  It freezes the main
Mdot=5 anchors into one audit table, computes source-corrected Shen-style slope
diagnostics, and post-processes the energy-wind checkpoints for the launch
energy required to match target power-law wind slopes.

New script:

```text
scripts/audit_mdot5_shen_wind_diagnostics.py
```

New outputs:

```text
outputs/tables/m5_wind_shen_budget_diagnostics.md
outputs/tables/m5_wind_shen_budget_diagnostics.json
outputs/tables/m5_wind_shen_slope_profiles.json
```

## Main Results

The no-wind `Mdot_inner/Edd=5`, `f_s=0.80` anchor remains strict:

```text
final_full = 2.783e-10
f_adv_global = 0.49897
f_adv_inner = 0.47158
Lrad/LEdd = 1.29956
max H/R = 0.31521
Rson = 4.361 rg
```

The strongest current energy-only wind checkpoint, `eta_6p425`, is also strict:

```text
final_full = 2.473e-10
Qwind/Qvisc = 0.82285
Lrad/LEdd = 0.52225
f_adv_global = -0.00666
f_adv_inner = -0.15201
max H/R = 0.13548
Rson = 5.269 rg
```

Interpreting this same energy sink as an escape-energy wind still implies a
large mass reservoir requirement:

```text
implied Mwind/Mdot_inner = 3.5855
required Mdot_outer/Mdot_inner = 3.7855
```

The Shen-equivalent cumulative slope of this implied wind is moderate:

```text
s_equiv_implied_etaesc1 = 0.362
```

So the issue is not that a Shen-style wind profile is impossible.  The issue is
that the current BVP has not solved the required local mass loading and outer
reservoir supply self-consistently.

## Power-Law Bridge Check

The prescribed power-law mass bridge remains a clean calibration target.

At `zeta=0.10`:

```text
final_full = 3.490e-08
Mdot_outer/Mdot_inner = 0.55855
current wind sink/Mdot_inner = 0.35865
s_equiv_current_mass_profile = 0.07382
s_fit_tilde_current_mdot_profile = 0.07391
Lrad/LEdd = 0.58279
f_adv_global = 0.00112
Rson = 5.232 rg
```

The fitted source-corrected slope matches the imposed power-law slope, so the
diagnostic is behaving sensibly.

However, the actual `Qwind(R)` in that state, if converted with
`Ew=GM/(2R)`, would imply a much larger mass loss:

```text
implied etaesc1 Mwind/Mdot_inner = 4.473
required Mdot_outer/Mdot_inner = 5.032
s_equiv_implied_etaesc1 = 0.410
```

This confirms that the prescribed bridge is useful but not a physical local
wind solution.

## Launch-Energy Calibration

For `eta_6p425`, the required launch-energy multiplier to realize target
Shen-style slopes is:

```text
target s=0.10:
    median eta_E_req = 13.16
    p10/p90 = 5.59 / 14.72
    median v_inf/c = 0.457

target s=0.30:
    median eta_E_req = 4.39
    p10/p90 = 1.86 / 4.91
    median v_inf/c = 0.264
```

This is encouraging.  A moderate Shen slope around `s~0.3` does not require an
absurd local launch energy; an `Ew = eta_E GM/(2R)` closure with `eta_E` of a
few, or a terminal-speed closure, is a plausible next physical parameterization.

For the weaker `epsilon_w=0.98` checkpoint:

```text
s_equiv_implied_etaesc1 = 0.148
target s=0.10 median eta_E_req = 3.40
target s=0.30 median eta_E_req = 1.13
```

That checkpoint is closer to a weak/moderate wind calibration point.

## Local Mdot(R) BVP Status

The prototype local `Mdot(R)` BVP remains non-production:

```text
zeta=0.03, eta_E=33.333, N=96:
    final_full = 4.574e-04
    mass_residual_max = 4.574e-04

zeta=0.03, eta_E=33.333, N=128:
    final_full = 4.055e-03
    mass_residual_max = 4.055e-03
```

The N=128 result is worse than N=96, so the current prototype is not
mesh-converged.  The next solver work should target local-mass residual
localization, scaling, and staged continuation rather than stronger wind
coupling.

## Interpretation

The project now has three clean tiers:

```text
1. No-wind Mdot=5 stream-fed slim state:
   strict and physically advective.

2. Energy-only wind-cooled Mdot=5 state:
   strict and numerically real, but not mass-loaded.

3. Prescribed Shen-style power-law mass bridge:
   useful calibration family, but not a local Qwind/Ew solution.
```

The Shen prior changes the interpretation of the power-law bridge: it is a
reasonable calibration target, not an arbitrary hack.  But the physical branch
still requires a strict local `Mdot(R)` BVP and eventually wind angular-momentum
coupling plus reservoir-controlled outer boundaries.

## Recommended Next Step

Implement the production local `Mdot(R)` BVP diagnostics and continuation:

```text
1. Add residual localization for the local mass residual:
   R_M(R), interval_E(R), Qwind/Qvisc, Mwind_prime/Mdot,
   stream_prime/Mdot, Mdot_tilde/Mdot_inner, s_eff_tilde(R),
   and Jacobian row/column norms.

2. Start from the prescribed zeta=0.03 bridge.

3. Continue local BVP in launch energy:
   eta_E = 100 -> 60 -> 40 -> 33.333.

4. Require:
   final_full <= 1e-5 exploratory,
   mass residual not localized in one unresolved cell,
   and N=96/128/192 trend improving.

5. Only then move to:
   zeta=0.05 with eta_E~20,
   zeta=0.10 with eta_E~10,
   and physical closure scans eta_E = 1, 3, 10, 30.
```

Do not add wind angular momentum yet.  First make the mass/energy local BVP
strict at the calibrated Shen target.
