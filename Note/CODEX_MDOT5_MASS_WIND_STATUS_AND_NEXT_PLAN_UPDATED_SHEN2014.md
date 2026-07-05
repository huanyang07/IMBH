# Codex Handoff: Mdot=5 Mass-Loaded Wind Branch, Shen–Matzner Power-Law Prior, and Next Plan

Date: 2026-07-05

Repository context: `huanyang07/IMBH` on `main`.

This is an updated version of the previous Mdot=5 mass-wind handoff. It adds the Shen & Matzner (2014, ApJ 784, 87) analytic prior for wind-driven radial accretion profiles, especially the prescription

```math
\dot M_{\rm acc}(R) \propto R^s,
\qquad 0 \le s \le 1,
```

for windy advective disks. The key update is: **the current prescribed power-law mass-wind bridge is no longer just a numerical hack; it is a useful literature-motivated calibration family.** It should still not be the final physical solution. The final model should solve `Mdot(R)` locally from the wind energy closure.

Relevant files already produced / reviewed:

```text
Note/CODEX_MDOT5_ENERGY_WIND_VALIDATION_RESULTS.md
Note/CODEX_MDOT5_POWERLAW_MASS_WIND_BRIDGE_RESULTS.md
Note/CODEX_MDOT5_WIND_POWERLAW_SLOPE_RESULTS.md
outputs/tables/m5_energy_wind_mesh_validation_summary.md
outputs/tables/m5_energy_wind_implied_mass_coupled_budget.md
outputs/tables/m5_energy_wind_powerlaw_slope_diagnostics.json
src/imri_qpe/layer3_minidisk_1d/winds.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
scripts/run_mdot5_powerlaw_mass_wind_pilot.py
scripts/run_mdot5_local_mdot_bvp_pilot.py
```

Additional literature context added here:

```text
Shen & Matzner 2014, ApJ, 784, 87, especially Section 4 and Appendix A.
```

---

## 1. Executive assessment

The project has made a real numerical advance, but the main bottleneck has shifted.

Previously, the key question was:

```text
Can the finite stream-fed Mdot_inner/Edd=5 minidisk access a high-Mdot branch?
```

The answer is now **yes**. The no-wind compact-source `Mdot_inner/Edd=5`, `f_s=0.80` branch exists and is a strong steady slim-disk/high-Mdot anchor.

The current question is now:

```text
Can the wind-regulated upper state be made physically self-consistent once wind
energy, wind mass loss, wind angular momentum, and the outer reservoir supply
are coupled in the same BVP?
```

The latest results show three separate levels of evidence:

```text
Level 1: Energy-only wind cooling branch
    Numerically robust and mesh validated.
    But wind removes energy without mass or angular momentum.

Level 2: Prescribed power-law mass-wind bridge
    Successful up to zeta=0.10 of the full implied energy-wind mass loss.
    Useful as a calibration/debug bridge.
    Now has literature support from Shen & Matzner-style Mdot(R) ∝ R^s winds.
    But wind mass profile is prescribed, not locally solved from Qwind/Ew.

Level 3: Fully local Mdot(R) mass-coupled wind BVP
    Prototype exists.
    Not yet production quality; residuals are still too high and N behavior is not tight.
```

Bottom line:

```text
The energy-only wind branch is now a well-supported wind-cooling solution family.
The prescribed power-law mass-wind bridge shows that adding some wind mass loss
does not immediately destroy the transonic branch.
Shen & Matzner 2014 makes a power-law Mdot(R) family a defensible calibration prior.
But the physical mass-loaded, angular-momentum-coupled wind-regulated high state
is not yet demonstrated.
```

Do not chase `epsilon_w -> 1` or `zeta -> 1` as the next trophy. The next decisive step is to turn the prototype local `Mdot(R)` wind BVP into a robust production solver and close the mass/energy/angular-momentum/reservoir budgets.

---

## 2. Current status in numbers

### 2.1 Standard high-Mdot backbone

The standard no-wind slim disk was previously recovered to `Mdot/Edd=5`:

```text
Mdot/Edd = 5
residual = 2.293e-6
f_adv_global = 0.4534
f_adv_inner(R<20rg) = 0.4666
Lrad/LEdd = 1.541
max H/R = 0.3164
Rson = 4.360 rg
```

This remains the clean reference proving the solver can recover a true high-Mdot advective slim branch.

### 2.2 Finite stream-fed Mdot=5 no-wind / no-heating anchor

The finite stream-fed compact-source branch at `Mdot_inner/Edd=5`, `f_s=0.80` is a real high-Mdot anchor:

```text
Mdot_inner/Edd = 5
Rout ~ 300-335 rg class
f_s = 0.80
eta_heat = 0
wind = 0
f_adv_global ~ 0.499
f_adv_inner ~ 0.472
Lrad/LEdd ~ 1.300
max H/R ~ 0.315
Rson ~ 4.36 rg
Mdot_outer/Mdot_inner ~ 0.20
```

This is a steady high-Mdot/slim anchor, not yet a full QPE high state.

### 2.3 Energy-only wind branch

The energy-only wind branch is now numerically strong. The latest validation freezes anchors at `epsilon_w=0, 0.50, 0.80, 0.98, 0.997` and `eta=6.20, 6.30, 6.35`, and then manually extends to `eta=6.425`.

Representative frozen anchors:

| state | residual | Qwind/Qvisc | Lrad/LEdd | f_adv_global | f_adv_inner | max H/R | Rson/rg |
|---|---:|---:|---:|---:|---:|---:|---:|
| epsilon_w=0 | 2.783e-10 | 0 | 1.2996 | 0.4990 | 0.4716 | 0.3152 | 4.361 |
| epsilon_w=0.98 | 2.157e-10 | 0.1976 | 1.1429 | 0.3710 | 0.3140 | 0.2805 | 4.462 |
| epsilon_w=0.997 | 1.206e-09 | 0.6898 | 0.6931 | 0.0614 | -0.0932 | 0.1789 | 4.935 |
| eta=6.20 | 3.677e-10 | 0.7834 | 0.5811 | 0.0107 | -0.1468 | 0.1510 | 5.141 |
| eta=6.35 | 1.593e-10 | 0.8110 | 0.5412 | -0.0019 | -0.1515 | 0.1406 | 5.227 |
| eta=6.425 | ~2.47e-10 | 0.8228 | 0.5222 | -0.0067 | not quoted in note | not quoted in note | 5.269 |

Interpretation:

```text
As wind cooling strengthens:
    Lrad decreases,
    H/R decreases,
    Rson moves outward,
    f_adv_global falls toward zero/slightly negative.
```

This is not an advective-dominated hot state at high wind strength. It is better called a **wind-cooled high-Mdot steady state**.

### 2.4 Mesh validation of energy-only wind

Mesh validation passed at `N=768, 896, 1024` for:

```text
epsilon_w = 0.98
eta = 6.20
eta = 6.35
```

The physical diagnostics are essentially stable across N. This is strong evidence that the energy-only wind cooling branch is numerically real.

### 2.5 Main physical caveat: implied mass loss is huge

Post-processing the energy-only branch with:

```text
E_w = GM/(2R)
l_w = l
Mdot_wind_prime = 2 pi R^2 Qwind / E_w
```

implies very large mass loss:

| state | Qwind/Qvisc | implied Mwind/Mdot_inner | current Mdot_outer/Mdot_inner | required Mdot_outer/Mdot_inner |
|---|---:|---:|---:|---:|
| epsilon_w=0.98 | 0.198 | 0.864 | 0.20 | 1.064 |
| epsilon_w=0.997 | 0.690 | 2.967 | 0.20 | 3.167 |
| eta=6.20 | 0.783 | 3.386 | 0.20 | 3.586 |
| eta=6.35 | 0.811 | 3.523 | 0.20 | 3.723 |
| eta=6.425 | 0.823 | 3.586 | 0.20 | 3.786 |

This is the key physical issue.

At fixed `f_s=0.80`, the no-wind stream source gives `Mdot_outer/Mdot_inner=0.20`. But if the energy sink is interpreted as an escape-energy wind, the outer boundary would need to supply several times the inner accretion rate.

Therefore:

```text
The high-eta energy-only wind branch is not self-consistent as a mass-loaded
wind unless the reservoir/outer boundary and Mdot(R) field are changed.
```

### 2.6 Prescribed power-law mass-wind bridge

Codex implemented a prescribed cumulative power-law wind sink:

```text
Mdot(R) / Mdot_inner
  = 1 + f_w * powerlaw_wind_shape(R) - f_stream * stream_shape(R)
```

where the wind sink fraction is:

```text
wind_sink_fraction = zeta * implied_full_energy_wind_mass_loss
```

using the `eta=6.425` energy-wind checkpoint as reference. For that reference:

```text
implied Mwind/Mdot_inner = 3.58551
```

Thus:

```text
zeta=0.10 -> wind_sink_fraction = 0.358551
```

The adaptive prescribed bridge reached `zeta=0.10` at `N=896`:

```text
zeta = 0.10
wind_sink_fraction = 0.358551
Mdot_outer/Mdot_inner = 0.558551
final residual = 3.490e-8
Lrad/LEdd = 0.582786
f_adv_global = 0.001115
Rson = 5.23164 rg
```

Mesh spot checks at `zeta=0.03`, `0.05`, and `0.10` are encouraging. The N=640/768 residuals at `zeta=0.05` and `zeta=0.10` are looser than the N=896 anchor but still physics-gated, with physical diagnostics drifting weakly.

Interpretation:

```text
The prescribed power-law wind-mass bridge is a useful calibration/debug target.
It now has a clear analytic prior from windy advective disk theory.
But it is not the final physical wind branch, because the wind mass profile is
imposed rather than solved locally from Qwind/Ew.
```

### 2.7 Local Mdot(R) BVP prototype

Codex added a prototype third-field BVP with node-wise `logMdot_i` and a local mass-continuity residual:

```text
dMdot/dlnR = wind_prime - source_prime
```

with

```text
wind_prime = 2 pi R^2 Qwind / E_w
E_w = eta_E GM/(2R)
```

The prototype is not yet production quality. Calibration results:

| bridge target | N | eta_E | initial residual | final residual | Mdot_outer/Mdot_inner | Lrad/LEdd | f_adv_global | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zeta=0.03 | 96 | 33.333 | 0.578 | 4.574e-4 | 0.307304 | 0.553070 | 0.002176 | 5.26372 |
| zeta=0.03 | 128 | 33.333 | 0.379 | 4.055e-3 | 0.305616 | 0.553521 | 0.000565 | 5.26589 |
| zeta=0.05 | 96 | 20 | 0.538 | 7.523e-3 | 0.378536 | 0.583849 | 0.002699 | 5.25314 |
| zeta=0.10 | 96 | 10 | 0.461 | 2.303e-2 | 0.559040 | 0.605375 | 0.013074 | 5.23509 |

The important lesson is that the calibrated sequence `eta_E ~ 1/zeta` is a good debug path. The physical unscaled `E_w = GM/(2R)` imposes the full post-processed mass loss and is too aggressive for a first local-BVP solve.

---

## 3. New literature addendum: Shen & Matzner 2014 power-law windy disks

### 3.1 Why the paper matters here

Shen & Matzner (2014) model evolving TDE disks and explicitly include wind losses in the early advective stage. In their advective-disk treatment, they use the standard windy accretion prescription:

```math
\dot M_{\rm acc}(R) \propto R^s,
\qquad 0 \le s \le 1.
```

They describe:

```text
s = 0: no wind
s = 1: strong mass loss
```

This is directly relevant to the current Codex power-law mass-wind bridge.

The important conclusion for this project is:

```text
A power-law Mdot(R) profile is a defensible analytic calibration family.
It should be used to calibrate and validate the local wind closure.
It should not be imposed forever as the final answer.
```

### 3.2 Key equations from Shen & Matzner-style windy disk theory

Shen & Matzner write mass and angular-momentum conservation for a disk that loses mass in a wind. The wind removes mass at rate `Sigma_dot_w` and angular momentum with a lever arm parameter `f_j`.

A useful local prescription in their Appendix A is:

```math
\dot\Sigma_w = sK\frac{\Sigma\nu}{R^2},
```

with

```math
K = \frac{3}{2}\frac{1+2s}{1-2s(f_j-1)}.
```

For smooth steady flows, avoid the wind-induced instability boundary:

```math
f_j < 1 + \frac{1}{2s}.
```

For the no-lever-arm baseline `f_j = 1`, and the advective Shakura-Sunyaev case `n = 1/2`, their self-similar solution reduces to the familiar scaling:

```math
\eta = \frac{4(1+s)}{3}
```

for central accretion decay in the wind-free fallback comparison. This is not something Codex needs to reproduce immediately, but it is useful context: wind mass loading and angular-momentum loading are tightly linked.

Implication for the current project:

```text
Begin with l_w = l or f_j = 1.
Only scan lever arms after the local mass-coupled wind BVP is strict.
Avoid aggressive lever arms that approach the wind-induced instability boundary.
```

### 3.3 How to compare our stream-fed disk to a Shen-style power law

Our model is not a pure wind-only disk. It has a localized compact stream source. The continuity equation is:

```math
\frac{d\dot M}{d\ln R}
=
\dot M'_w - \dot M'_s.
```

Therefore the raw `Mdot(R)` profile contains both wind and stream-source structure. Do **not** fit a global Shen-style slope directly across the compact source annulus.

Define a source-corrected cumulative profile:

```math
\widetilde{\dot M}(R)
=
\dot M(R)
+
\int_{R_{\rm in}}^R \dot M'_s\,d\ln R.
```

Then:

```math
\frac{d\widetilde{\dot M}}{d\ln R}
=
\dot M'_w.
```

This `Mdot_tilde(R)` is the correct object to compare with a wind-only power law:

```math
\widetilde{\dot M}(R)
\simeq
\dot M_{\rm inner}\left(\frac{R}{R_{\rm ref}}\right)^s.
```

Recommended diagnostics:

```text
s_eff_raw(R) = dlnMdot/dlnR
    use only in source-free zones.

Mdot_tilde(R) = Mdot(R) + integral_inner_to_R Mdot_stream_prime dlnR

s_eff_tilde(R) = dlnMdot_tilde/dlnR
    this is the main diagnostic for comparison to Shen-style wind profiles.

s_fit_tilde = fitted slope of ln(Mdot_tilde/Mdot_inner) vs ln(R/R_ref)
    fit only over wind-active, source-free, non-boundary intervals.
```

Practical exclusion zones:

```text
exclude sonic/inner regularity buffer if noisy;
exclude the compact stream source annulus;
exclude the outermost boundary buffer if outer residuals are active;
fit over the clean wind-active region.
```

### 3.4 Useful scale conversion for the current Mdot=5 branch

For the current branch:

```text
Mdot_inner/Edd = 5
Rout ≈ 335 rg
Rson ≈ 4.5-5.3 rg depending on wind strength
```

A rough radial lever arm is:

```math
\frac{R_{\rm out}}{R_{\rm in}}
\sim
\frac{335}{5}
\sim 67.
```

If the cumulative wind mass loss is `Mwind/Mdot_inner`, a rough equivalent Shen-style slope is:

```math
s_{\rm equiv}
\approx
\frac{\ln(1+M_w/\dot M_{\rm inner})}
{\ln(R_{\rm out}/R_{\rm in})}.
```

Approximate values:

| cumulative wind loss | equivalent `s` over `Rout/Rin ~ 67` | interpretation |
|---:|---:|---|
| `Mwind/Mdot_inner = 0.3` | `s ~ 0.06` | weak wind |
| `Mwind/Mdot_inner = 1.0` | `s ~ 0.16` | moderate wind |
| `Mwind/Mdot_inner = 3.5` | `s ~ 0.36` | moderate-to-strong wind |
| `Mwind/Mdot_inner = 10` | `s ~ 0.57` | strong wind |

This is important: the high-eta energy-only branch, if interpreted as `Mwind/Mdot_inner ~ 3.5`, corresponds to approximately:

```text
s_equiv ~ 0.35-0.36
```

That is not absurd under a Shen-style `0 <= s <= 1` prior. It is a substantial but plausible windy advective-disk slope. The issue is not primarily that the implied wind mass is mathematically impossible. The issue is that the current BVP has not yet solved the required mass and reservoir supply self-consistently.

### 3.5 What this means for `E_w`

The local mass-loaded wind closure is:

```math
\dot M'_w
=
\frac{2\pi R^2 Q_{\rm wind}}{E_w}.
```

If we target a Shen-style corrected profile:

```math
\widetilde{\dot M}(R)
=
\dot M_{\rm inner}
\left(\frac{R}{R_{\rm ref}}\right)^s,
```

then the target local wind source is approximately:

```math
\dot M'_w(R) = s\,\widetilde{\dot M}(R),
```

and the required launch energy is:

```math
E_{w,\rm req}(R)
=
\frac{2\pi R^2 Q_{\rm wind}(R)}
{s\,\widetilde{\dot M}(R)}.
```

Codex should post-process the current energy-wind checkpoints and output:

```text
Ew_req(R)
eta_E_req(R) = Ew_req / [GM/(2R)]
v_inf_req(R) = sqrt(2 Ew_req)
```

for target slopes:

```text
s = 0.05, 0.10, 0.20, 0.30, 0.50
```

Interpretation:

```text
eta_E_req ~ 1-30 and smooth:
    plausible.

v_inf_req ~ 0.05c-0.3c and smooth:
    plausible for super-Eddington/TDE-like winds.

eta_E_req << 1:
    suspicious; wind mass loss is cheaper than local escape energy.

eta_E_req >> 50 or very spiky:
    target Mdot(R) is probably incompatible with the local Qwind distribution.

negative or non-smooth Ew_req:
    likely residual, activation, or source/boundary artifact.
```

### 3.6 How to use Shen & Matzner without making the model circular

Do **not** do this:

```text
choose desired Mdot(R)
invert for arbitrary Ew(R)
claim the solved Mdot(R) agrees with the literature
```

That would make the model circular.

Do this instead:

```text
1. Use Shen-style Mdot ∝ R^s as a low-dimensional validation/calibration family.
2. Compute Ew_req(R) for a few target s values.
3. Fit Ew_req only with a physically bounded low-parameter closure.
4. Solve Mdot(R) locally in the BVP.
5. Compare the resulting source-corrected Mdot_tilde(R) to the Shen-style s range.
```

Recommended physical `E_w` closures:

```text
A. Escape-multiple closure:
   Ew = eta_E GM/(2R)
   eta_E = 1, 3, 10, 30

B. Terminal-speed closure:
   Ew = xi_esc GM/(2R) + 0.5 v_inf^2
   xi_esc = 1 initially
   v_inf = 0.05c, 0.10c, 0.15c, 0.30c

C. Smooth 2-parameter eta_E(R):
   only if A/B fail;
   must be smooth, monotonic or weakly curved, and physically bounded.
```

Reason for terminal-speed closure:

```text
At R ~ 300 rg, GM/(2R) = c^2/600.
A wind with v_inf = 0.15c has kinetic energy ~0.01125 c^2,
which is ~6.75 times GM/(2R).
So eta_E of a few to a few tens is not crazy at large radius.
Escape-only eta_E=1 may overestimate mass loading.
```

### 3.7 Suggested prior ranges for `s`

Use broad categories, not a single forced value:

```text
weak wind:
    s_eff ~ 0.05-0.15

moderate wind:
    s_eff ~ 0.15-0.40

strong wind:
    s_eff ~ 0.40-1.00
```

For the current Mdot=5 branch, likely useful targets are:

```text
s = 0.10, 0.20, 0.30, 0.50
```

Interpretation of existing checkpoints:

```text
prescribed bridge zeta=0.10:
    Mwind/Mdot_inner ~ 0.359
    s_equiv ~ 0.07
    weak-wind calibration point.

energy-only high-eta endpoint interpreted with Ew=GM/(2R):
    Mwind/Mdot_inner ~ 3.5-3.6
    s_equiv ~ 0.36
    moderate-to-strong wind calibration point.
```

---

## 4. What we understand now

### 4.1 The high-Mdot finite stream-fed branch exists

The finite stream-fed `Mdot_inner/Edd=5` branch exists and connects smoothly to the standard high-Mdot slim backbone.

### 4.2 Conservative stream heating was not the missing trigger

Previous stream-heating ladders showed that adding conservative stream heating increases luminosity slightly and thickens the disk slightly, but does not produce a new stronger advective branch.

### 4.3 Energy-only wind cooling is numerically robust

The wind-aware `interval_E` Jacobian and mesh validation show that the energy-only wind branch is not a numerical mirage.

### 4.4 Strong wind cooling can replace advection

At high wind efficiency, `Qwind` carries much of the energy that no-wind slim disks would otherwise carry partly through advection. Therefore low or mildly negative `f_adv_global` is not automatically a failure.

The correct interpretation is:

```text
High eta branch = wind-cooled high-Mdot state,
not advective-dominated hot state.
```

### 4.5 The physical bottleneck is mass loading and reservoir supply

The high-eta energy-only wind branch, interpreted as an escape-energy wind, requires `Mdot_outer/Mdot_inner ~ 3.7-3.8`, not `0.2`. That is not a small perturbation.

So the main problem is no longer residual convergence. It is physical closure:

```text
mass + energy + angular momentum + reservoir supply must be solved together.
```

### 4.6 Shen & Matzner supports the power-law bridge as a calibration prior

The prescribed bridge should not be dismissed as arbitrary. The power-law form is close to a standard analytic windy-disk prescription.

But the proper hierarchy remains:

```text
prescribed power-law bridge:
    calibration/debug target

local Mdot(R) BVP:
    physical solution target

source-corrected fitted s_eff:
    validation diagnostic
```

---

## 5. What we do not understand yet

1. **Whether a fully local mass-loaded wind BVP exists at high zeta.**
   The prescribed power-law bridge reaches `zeta=0.10`, but the true local BVP is not yet tight.

2. **What launch energy is physically appropriate.**
   `E_w = GM/(2R)` is a useful baseline, but real winds may include enthalpy, terminal kinetic energy, radiation work, magnetic/turbulent help, and angular-momentum work.

3. **What Shen-style slope the solved model naturally chooses.**
   The model should not impose one exact `s`; it should solve `Mdot(R)` and then report `s_eff`.

4. **Whether wind angular-momentum loss changes the branch.**
   The current bridge is not yet a full angular-momentum-coupled wind model.

5. **Whether the outer reservoir can supply the required mass.**
   If the high-wind state requires `Mdot_outer/Mdot_inner > 3`, then the external stream/reservoir must be modeled differently from the fixed `f_s=0.80`, fixed `Mdot_inner` setup.

6. **Whether the wind-regulated branch participates in a limit cycle.**
   A steady branch is not enough. The project still needs an equilibrium/stability map and eventually a time-dependent reload/drain cycle.

---

## 6. Main problem right now

The main problem is:

```text
The project has a numerically robust energy-wind branch and a promising
Shen-motivated prescribed mass-wind bridge, but not yet a self-consistent local
mass/energy/angular-momentum wind branch with a reservoir boundary.
```

The next sprint should not focus on pushing `epsilon_w` closer to one. It should focus on making the local mass-coupled wind BVP robust and physically interpretable.

---

## 7. Concrete next plan

### Phase 0 — Freeze and protect anchors

Create regression anchors for:

```text
A. No-wind Mdot=5, f_s=0.80 compact-source branch
B. Energy-only wind states:
   epsilon_w = 0.98
   epsilon_w = 0.997
   eta = 6.20
   eta = 6.35
   eta = 6.425
C. Prescribed power-law mass-wind bridge:
   zeta = 0.03
   zeta = 0.05
   zeta = 0.10
D. Local Mdot(R) BVP prototype:
   zeta = 0.03, eta_E = 33.333
```

Each anchor should store:

```text
full residual
physical_E
mass-continuity residual
energy budget residual
angular-momentum budget residual if wind AM loss is enabled
Mdot_outer/Mdot_inner
wind_sink_integral/Mdot_inner
source_integral/Mdot_inner
Qwind/Qvisc
Lrad/LEdd
f_adv_global
f_adv_inner
f_adv_pos
max H/R
Rson
peak residual locations
checkpoint path
s_eff_raw and s_eff_tilde if Mdot(R) is available
```

### Phase 1 — Make budget audits first-class diagnostics

For every wind run, output one compact budget table:

```text
Mass:
    Mdot_outer - Mdot_inner - Mwind + Mstream

Energy:
    integral(Qvisc + Qstream + Qtide)
  - integral(Qrad + Qadv + Qwind)

Angular momentum:
    advective AM flux + viscous torque + stream AM source - wind AM sink
```

Use both signed and positive-only versions where relevant.

Acceptance:

```text
mass budget error <= 1e-4 preferred
energy budget error <= few x 1e-4 preferred
AM budget error tracked, then tightened once wind AM is enabled
```

### Phase 2 — Add Shen-style source-corrected slope diagnostics

Implement diagnostics:

```text
Mdot_tilde(R) = Mdot(R) + integral_inner_to_R Mdot_stream_prime dlnR
s_eff_raw(R) = dlnMdot/dlnR
s_eff_tilde(R) = dlnMdot_tilde/dlnR
s_fit_tilde over clean wind-active/source-free intervals
```

Output files:

```text
outputs/tables/m5_wind_shen_slope_diagnostics.json
outputs/tables/m5_wind_shen_slope_summary.md
```

Include:

```text
fit radial range
excluded intervals
cumulative Mwind/Mdot_inner
s_equiv from total wind loss
s_fit_tilde
median/percentile s_eff_tilde
source contamination flags
outer-boundary contamination flags
```

Acceptance:

```text
s_eff_tilde smooth in clean wind-active zones;
fit stable under N changes;
fit not dominated by the stream source annulus;
fit not dominated by the outer boundary buffer.
```

### Phase 3 — Post-process current energy-wind checkpoints for required Ew(R)

For each current energy-wind checkpoint:

```text
epsilon_w = 0.98
epsilon_w = 0.997
eta = 6.20
eta = 6.35
eta = 6.425
```

For target slopes:

```text
s = 0.05, 0.10, 0.20, 0.30, 0.50
```

compute:

```math
E_{w,req}(R)
=
\frac{2\pi R^2 Q_{wind}(R)}{s\,\widetilde{\dot M}(R)}
```

and output:

```text
eta_E_req(R) = E_w_req / [GM/(2R)]
v_inf_req(R) = sqrt(2 E_w_req)
median eta_E_req
10/90 percentile eta_E_req
smoothness metric
spike count
fraction of intervals with eta_E_req < 1
fraction of intervals with eta_E_req > 50
```

This diagnostic decides whether the current `Qwind(R)` distribution is compatible with a Shen-style power-law mass profile.

### Phase 4 — Keep the prescribed bridge, but label it correctly

Continue using the prescribed power-law wind bridge as a debug/calibration target. Do not call it a physical solution.

Recommended prescribed bridge targets:

```text
zeta = 0.03
zeta = 0.05
zeta = 0.10
optional: zeta = 0.15, 0.20 only as code stress tests
```

For each bridge checkpoint, report:

```text
s_equiv
s_fit_tilde if applicable
Mwind/Mdot_inner
Mdot_outer/Mdot_inner
energy budget
mass budget
residual localization
```

Do not spend much effort going to `zeta=1` with the prescribed bridge. That would mostly prove that an imposed `Mdot(R)` profile can be solved around, not that the physical wind branch exists.

### Phase 5 — Upgrade the local Mdot(R) BVP into production machinery

This is the most important next step.

Move the third unknown field `logMdot_i` and local continuity residual into the production square-Newton/collocation framework.

Use the normalized residual form:

```text
d ln Mdot / d ln R - (wind_prime - source_prime)/Mdot = 0
```

rather than an unscaled `dMdot` residual. This should improve conditioning across orders of magnitude in mass flux.

Recommended state vector:

```text
z = [logu_i, logT_i, logMdot_i, lambda0, maybe outer/reservoir parameter]
```

Recommended local continuity residual:

```text
R_M_i = D_x(logMdot)_i - (Mdot_wind_prime_i - Mdot_stream_prime_i)/Mdot_i
```

Recommended Jacobian pieces:

```text
dR_M / dlogMdot
dR_M / dlogT via Qwind
dR_M / dlogu via H, Qadv, Qwind
dR_E / dlogMdot because Sigma and Qadv depend on Mdot
dR_Omega / dlogMdot because viscous stress/angular momentum depend on Mdot
```

### Phase 6 — Continue in calibrated launch energy, not directly in physical wind strength

The calibration result shows that:

```text
eta_E ~ 1/zeta
```

is the right debug path.

For the local BVP, use a two-parameter continuation:

```text
zeta: 0.01 -> 0.03 -> 0.05 -> 0.10
eta_E: large -> calibrated target -> physical target
```

Suggested path:

```text
1. Start from prescribed zeta=0.03 bridge.
2. Solve local BVP with eta_E = 100.
3. Continue eta_E down to 33.333.
4. Then continue zeta to 0.05 with eta_E ~ 20.
5. Then continue zeta to 0.10 with eta_E ~ 10.
6. Only after those are strict, explore eta_E below the calibrated value.
```

Acceptance for a local BVP checkpoint:

```text
final_full <= 1e-6 preferred, <= 1e-5 exploratory
mass residual localized and small everywhere
N=96/128/192/256 trend improves, not worsens
Mdot(R) matches prescribed bridge at calibration target within ~1-2%
physical diagnostics stable with N
s_eff_tilde stable and plausible
```

### Phase 7 — Test physical Ew closures motivated by Shen-style calibration

After `Ew_req(R)` post-processing, test low-dimensional closures:

```text
A. Ew = eta_E GM/(2R)
   eta_E = 1, 3, 10, 30

B. Ew = xi_esc GM/(2R) + 0.5 v_inf^2
   xi_esc = 1
   v_inf = 0.05c, 0.10c, 0.15c, 0.30c

C. Smooth two-parameter eta_E(R)
   only if A/B fail.
```

For each solved local BVP, report:

```text
Mwind/Mdot_inner
Mdot_outer/Mdot_inner
s_equiv
s_fit_tilde
eta_E effective distribution if variable
v_inf implied if applicable
Qwind/Qvisc
Lrad/LEdd
f_adv_global/inner
Rson
max H/R
mass/energy budgets
```

The goal is not exact agreement with any one literature profile. The goal is:

```text
solved Mdot_tilde(R) lies in a plausible Shen-style s range
and uses a physically interpretable Ew closure.
```

### Phase 8 — Add residual localization for the local mass equation

For the local BVP prototype, dump profiles of:

```text
R_mid/rg
R_M = local mass-continuity residual
interval_E
interval_R
interval_Omega
Qwind/Qvisc
Mdot_wind_prime/Mdot
Mdot_stream_prime/Mdot
logMdot slope
Mdot(R)/Mdot_inner
Mdot_tilde(R)/Mdot_inner
s_eff_tilde(R)
H/R
Qadv/Qvisc
Jacobian row norms
Jacobian column norms
```

Classify failures:

```text
If R_M peaks near wind activation front:
    improve wind activation derivatives / smoothing / mesh.

If R_M peaks near outer edge:
    revisit reservoir boundary and outer closure.

If R_M peaks near source annulus:
    improve source correction / source-wind overlap / local mesh.

If interval_E peaks while R_M is small:
    energy Jacobian/scaling still limiting.

If residuals are broad:
    row/column scaling and variable scaling are the likely bottleneck.
```

### Phase 9 — Introduce wind angular momentum loss after the local mass BVP is strict

Once local mass coupling works, add angular momentum removal:

```text
l_w = lambda_w l
lambda_w = 1.0, 1.2, 1.5, 2.0
```

The angular-momentum equation must include the wind sink consistently. If wind removes angular momentum, it should also affect the required viscous torque and possibly the sonic structure.

Do not scan aggressive lever arms until the `lambda_w=1` case is robust.

Tie to Shen & Matzner:

```text
Their Appendix A warns about wind-induced instability for too large a lever arm.
Use lambda_w=1 as the baseline.
Only move toward lambda_w>1 after mass/energy closure is strict.
Track whether the chosen lever arm approaches a wind-induced instability-like limit.
```

### Phase 10 — Decide what the outer reservoir is

At high wind loading, fixed `Mdot_inner` + fixed `f_s=0.80` is no longer a realistic steady reservoir model.

For a physical wind-regulated high state, the BVP should eventually allow one of these reservoir controls:

```text
A. fixed external stream supply Mdot_stream_supply;
B. fixed outer reservoir surface density / entropy;
C. fixed disk mass Mdisk;
D. fixed outer pressure/entropy plus free Mdot_inner eigenvalue.
```

The most important transition is:

```text
from imposed Mdot_inner
    to Mdot_inner determined by reservoir loading / outer boundary.
```

Without this, the model cannot yet say whether the high-wind state is naturally reached in a QPE cycle.

### Phase 11 — Build the equilibrium/stability map

After mass-coupled wind is robust, build steady sequences parameterized by reservoir loading:

```text
Mdisk or Sigma_out
external stream supply
outer entropy/load parameter
```

Output:

```text
Mdot_inner
Mdot_outer
Mdot_wind
Lrad
Qwind/Qvisc
f_adv_global
f_adv_inner
max H/R
Rson
s_fit_tilde
thermal stability label
```

Look for:

```text
low stable branch
unstable middle branch
wind-regulated or advective upper branch
upper and lower turning points
hysteresis
```

Only then does the project really test the QPE limit-cycle idea.

---

## 8. How to interpret negative/small advection at high wind

Small or mildly negative `f_adv_global` at high wind is not automatically a bug.

It can mean:

```text
wind cooling is carrying energy that no-wind slim disks would otherwise radiate
or advect.
```

However, it is physical only if:

```text
1. local energy residual is strict;
2. mass loss implied by Qwind/Ew is included in Mdot(R);
3. angular momentum loss is included or shown negligible;
4. entropy/advection profiles are smooth and mesh stable;
5. the outer reservoir can supply the required Mdot_outer;
6. the solved source-corrected Mdot_tilde(R) has a plausible Shen-style slope.
```

Until then, call high-eta results:

```text
candidate wind-cooled high-Mdot energy solutions
```

not:

```text
certified physical mass-loaded wind-regulated QPE high state.
```

---

## 9. Acceptance criteria before claiming a physical wind-regulated high branch

Require all of the following:

```text
1. Local Mdot(R) BVP strict convergence:
       final_full <= few x 1e-6 at N >= 512/768.

2. Mass budget closure:
       Mdot_outer/Mdot_inner = 1 - f_stream + f_wind
       to <= 1e-4 to 1e-3.

3. Energy budget closure:
       integral(Qvisc + Qstream + Qtide)
       = integral(Qrad + Qadv + Qwind)
       within tolerance.

4. Angular momentum budget closure:
       include l_w = lambda_w l or justify l_w=l as baseline.

5. Mesh robustness:
       N ladder at least 512/768/896 or 640/768/896,
       physical diagnostics stable to ~1-2%.

6. Closure robustness:
       chi_edd and activation width scans do not qualitatively change the branch.

7. Source/wind geometry robustness:
       reasonable source and wind shapes give same qualitative state.

8. Shen-style slope plausibility:
       source-corrected s_fit_tilde lies in a plausible range,
       roughly 0.05-1 depending on wind strength,
       and is not an artifact of the source annulus or outer boundary.

9. Launch energy plausibility:
       Ew closure is physically interpretable;
       eta_E_req is not <<1, not wildly spiky, and preferably of order 1-30.

10. Reservoir consistency:
       required Mdot_outer is compatible with chosen external supply/reservoir.

11. Stability relevance:
       branch appears in an equilibrium map with a plausible transition from a low state.
```

---

## 10. Immediate Codex task list

Pasteable implementation prompt:

```text
Current status:
- Energy-only Mdot_inner/Edd=5, f_s=0.80 wind branch is numerically robust.
- Mesh validation passed at N=768/896/1024 for epsilon_w=0.98, eta=6.20, eta=6.35.
- Eta continuation reached eta=6.425 with Qwind/Qvisc~0.823, Lrad/LEdd~0.522, f_adv_global~-0.0067.
- But energy-only wind has wind_sink_integral=0, so it is not a physical mass-loaded wind branch.
- Interpreting the high-eta energy sink as escape-energy wind implies Mwind/Mdot_inner~3.6 and required Mdot_outer/Mdot_inner~3.8, while the current model has only Mdot_outer/Mdot_inner=0.2.
- A prescribed power-law mass-coupled bridge from the eta=6.425 endpoint reaches zeta=0.10 at N=896 with wind_sink_fraction~0.3586 and Mdot_outer/Mdot_inner~0.5586.
- This prescribed power-law bridge is now literature-motivated by Shen & Matzner 2014, who use Mdot_acc(R) ∝ R^s with 0<=s<=1 for windy advective disks.
- Still, the prescribed bridge is only a calibration/debug target, not the final local wind solution.
- A prototype local Mdot(R) BVP exists but has residual floors ~4e-4 to 2e-2 at low N and is not production quality.

Do not chase epsilon_w->1 or zeta->1 next.

Next tasks:

1. Freeze anchors:
   - no-wind Mdot=5 f_s=0.80;
   - energy-only wind: epsilon_w=0.98, 0.997, eta=6.20, 6.35, 6.425;
   - prescribed power-law bridge: zeta=0.03, 0.05, 0.10;
   - local Mdot BVP prototype: zeta=0.03, eta_E=33.333.

2. Add first-class budget audits for every wind run:
   - mass budget;
   - energy budget;
   - angular-momentum budget;
   - peak residual localization.

3. Add Shen-style slope diagnostics:
   - Mdot_tilde(R) = Mdot(R) + integral_inner_to_R Mdot_stream_prime dlnR;
   - s_eff_raw(R) = dlnMdot/dlnR in source-free zones;
   - s_eff_tilde(R) = dlnMdot_tilde/dlnR;
   - s_fit_tilde over clean wind-active/source-free intervals.

4. Post-process energy-wind checkpoints for target Shen slopes:
   - s = 0.05, 0.10, 0.20, 0.30, 0.50;
   - compute Ew_req(R) = 2*pi*R^2*Qwind / [s*Mdot_tilde(R)];
   - output eta_E_req = Ew_req/[GM/(2R)] and v_inf_req = sqrt(2Ew_req);
   - flag eta_E_req << 1, eta_E_req >> 50, and spiky/non-smooth profiles.

5. Keep prescribed power-law bridge only as a calibration/debug target.
   Do not call it physical.
   Report s_equiv and s_fit_tilde for each zeta checkpoint.

6. Promote the local Mdot(R) BVP into production square Newton:
   - add logMdot_i as a third field;
   - use normalized residual
         dlnMdot/dlnR - (wind_prime - source_prime)/Mdot = 0;
   - include Jacobian terms linking logMdot to energy, angular momentum,
     Sigma, Qadv, Qwind, and outer closure.

7. Continue the local BVP in calibrated launch energy:
   - start zeta=0.03, eta_E=100 -> 33.333;
   - then zeta=0.05, eta_E~20;
   - then zeta=0.10, eta_E~10;
   - only afterward push toward physical eta_E=1 if justified.

8. Test physical Ew closures:
   A. Ew = eta_E GM/(2R), eta_E = 1, 3, 10, 30.
   B. Ew = xi_esc GM/(2R) + 0.5 v_inf^2,
      v_inf = 0.05c, 0.10c, 0.15c, 0.30c.
   C. Smooth two-parameter eta_E(R), only if A/B fail.

9. Add residual localization for the local mass residual:
   - R_M(R), interval_E(R), Mdot_wind_prime/Mdot,
     Mdot_stream_prime/Mdot, logMdot slope, Mdot_tilde/Mdot_inner,
     s_eff_tilde(R), Qwind/Qvisc, row/column Jacobian norms.

10. After local mass coupling is strict, add wind angular momentum:
    - l_w = lambda_w l with lambda_w = 1.0 first, then 1.2, 1.5, 2.0.
    - Avoid aggressive lever arms until the lambda_w=1 branch is robust.
    - Track Shen/Matzner-style wind-induced-instability constraints.

11. Replace fixed-Mdot_inner interpretation with reservoir-controlled sequences:
    - fixed external stream supply or fixed Mdisk/Sigma_out;
    - solve for Mdot_inner as part of the equilibrium branch.

12. Then build the equilibrium/stability map:
    - low branch, unstable branch, wind/advective upper branch,
      turning points, hysteresis, and eventually time-dependent cycle.
```

---

## 11. Bottom line

The latest update is good news, and the Shen & Matzner paper makes the power-law bridge more defensible.

```text
Solved / mostly solved:
    high-Mdot finite stream-fed branch exists;
    energy-only wind cooling branch is numerically robust;
    prescribed power-law mass bridge works to modest coupling;
    power-law Mdot(R) is supported as an analytic calibration prior.

Not solved:
    fully local mass-loaded wind BVP;
    angular-momentum-coupled wind;
    reservoir-consistent outer boundary;
    equilibrium/stability map;
    QPE limit cycle.
```

The next principled move is:

```text
Use Shen & Matzner-style Mdot ∝ R^s profiles to calibrate and diagnose the wind
launch energy, but let the production BVP solve Mdot(R) locally from Qwind/Ew.
Then embed the mass/energy/angular-momentum-coupled wind branch in a
reservoir-controlled equilibrium map.
```

That is the bridge from “numerically interesting wind-cooled steady solution” to “physically credible QPE high state.”
