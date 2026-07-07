# Codex Handoff — Source-Element Production Formulation for `eta_E=90`

**Project:** `huanyang07/IMBH`  
**GitHub commit under review:** `3d5fea8`  
**Primary notes/files:**  
- `Note/CODEX_MDOT5_SOURCE_MICRODOMAIN_RESULTS.md`
- `Note/GPT_PROMPT_SOURCE_ELEMENT_NEXT_STEP.md`
- `scripts/run_mdot5_local_mdot_eta_continuation.py`
- `outputs/tables/m5_local_mdot_eta90_source_buffer_*`
- `outputs/tables/m5_local_mdot_eta90_source_element_refine2_*`
- corresponding checkpoints under `outputs/checkpoints/`

## 0. Executive decision

The current `eta_E=90` branch is **not source-annulus certified**.

The latest source-buffer / source-element sprint was useful because it made the defect more honest, but it did **not** solve it. The best N251 source-element result improves the production residual only slightly and worsens the source-band audit. This means the next step should **not** be lower `eta_E`, stronger wind, more global node insertion, or another blind remap.

The next concrete formulation should be:

> **Promote source-band extra radial/energy rows into a local sparse source-element production solve, replace endpoint-linear quarter-point interpolation with true polynomial/Lobatto source-element states, and add conservative finite-volume mass, energy, and angular-momentum rows inside the source annulus.**

Use a **group-balanced penalty continuation / filter merit** so that the solve cannot trade a smaller mass residual for a hidden source-band energy residual.

---

## 1. Current target

```text
Mdot_inner/Edd = 5
Rout = 335 rg
Rinj = 240 rg
f_s = 0.80
eta_E = 90
source = compact source annulus
wind = local-Mdot mass-loaded formulation
```

The goal is to certify the `eta_E=90` source annulus before continuing to:

```text
eta_E = 80, 70, 60
```

Do **not** lower `eta_E` until the `eta_E=90` source annulus passes the certification gates in Section 9.

---

## 2. Current numerical status

From Codex’s latest prompt:

```text
Old N201 halo-4 source-domain checkpoint:
    production residual      ~ 3.929e-2
    source-band audit        ~ 3.088e-2

Best N251 source-element/refine2 result:
    production residual      ~ 3.752e-2
    source-band audit        ~ 3.612e-2
```

More detailed current best N251-style state:

```text
final_full                  ~ 3.751866e-2
interval_mass_residual_max  ~ 3.751866e-2
interval_R                  ~ 2.696455e-2
interval_E                  ~ 2.427828e-2
source extra energy         ~ 3.612137e-2
source extra radial         ~ 3.287952e-2
```

Interpretation:

```text
- Source-element internal nodes exist now.
- Source-buffer DeltaM variables exist now.
- Sparse/audit bookkeeping is improved.
- But the source annulus is still not satisfying mass conservation,
  radial balance, and energy balance simultaneously.
- N251 is not a certified improvement over N201.
```

This is **not** physical branch loss yet. It is a source-annulus representation/formulation problem.

---

## 3. Main diagnosis

The source annulus currently has a coupled defect involving:

```text
mass conservation,
radial momentum consistency,
energy consistency,
source-edge/interface compatibility,
and probably angular-momentum compatibility.
```

The latest source-element insertion alone did not fix the defect because the source-band residual evaluation is still too close to an endpoint-linear interval representation. In the current code path, quarter-point source-band rows effectively use something like:

```python
g = (y_right - y_left) / dx
yq = (1 - fraction) * y_left + fraction * y_right
```

Then the residual is evaluated at `yq`.

That means the source-band audit is still asking a piecewise-linear interval state to satisfy a steep, source-loaded ODE across the compact annulus. Adding more nodes helps only a little because the formulation still lets different constraint groups compete rather than enforcing a consistent local source element.

The key correction is:

> The same internal source-element state variables must be used by the mass, radial, energy, angular-momentum, and interface rows.

---

## 4. Direct answers to the current Codex questions

### 4.1 Should source-band extra rows become production rows with sparse/local Jacobian?

**Yes, but locally and carefully.**

Do **not** simply append many global rectangular rows and hope global Newton handles it. Instead:

```text
- make source-band radial/energy extra rows production rows
  inside a local source-element least-squares or sparse Newton solve;

- keep the global ordinary residual active outside the source annulus;

- keep the source-band audit rows as final diagnostics even after they become
  part of the local production solve.
```

This creates a controlled local augmented problem:

```text
ordinary production rows outside source band
+ source-element collocation rows inside source band
+ FV mass/energy/angular-momentum rows
+ interface compatibility rows
```

### 4.2 What weighting avoids trading mass residual against hidden energy residual?

Use a **group-balanced penalty continuation with filter acceptance**.

Define residual groups:

```text
G_P  = ordinary production rows outside source band
G_M  = finite-volume mass rows in source band
G_R  = source-band radial collocation rows
G_E  = source-band energy collocation rows
G_J  = source-band angular-momentum rows
G_FVE = finite-volume energy rows
G_I  = interface compatibility rows
```

For each group, report unweighted norms:

```math
\rho_P = ||r_P||_\infty,
\quad
\rho_M = ||r_M||_\infty,
\quad
\rho_R = ||r_R||_\infty,
\quad
\rho_E = ||r_E||_\infty,
\quad
\rho_J = ||r_J||_\infty,
\quad
\rho_I = ||r_I||_\infty.
```

Use a penalty ladder:

```text
gamma = 0.03, 0.10, 0.30, 1.0, 3.0, 10.0
```

Example merit:

```math
\Phi_\gamma =
\rho_P^2
+ w_M^2 \rho_M^2
+ \gamma^2
  \left(
      w_R^2\rho_R^2
    + w_E^2\rho_E^2
    + w_J^2\rho_J^2
    + w_{FVE}^2\rho_{FVE}^2
    + w_I^2\rho_I^2
  \right).
```

But the important part is the **filter rule**:

```text
Accept a local/global step only if:
    production residual does not worsen beyond tolerance;
    finite-volume mass residual does not worsen;
    source-band radial residual does not worsen;
    source-band energy residual does not worsen;
    source-band angular-momentum residual does not worsen;
    interface residual does not worsen;
    and total weighted merit decreases.
```

This prevents the optimizer from “buying” a lower mass residual by hiding a higher source-band energy defect, which is exactly the failure mode seen so far.

### 4.3 Rectangular least squares or square formulation?

For the next sprint: **use rectangular least squares locally in the source annulus.**

Rationale:

```text
- The problem is diagnostic/certification, not elegance.
- The source annulus has more physical consistency checks than the current
  square formulation can satisfy.
- Rectangular LS will honestly reveal whether the available local degrees of
  freedom can satisfy mass, radial, energy, angular momentum, and interface
  constraints simultaneously.
```

For the eventual production solver, a square formulation is still possible, but it should be built by **replacing** inadequate rows with conservative/source-element rows, not by hiding the source-band audit.

Recommended hierarchy:

```text
1. Local rectangular source-element least-squares for certification.
2. If successful, decide whether to keep it as production.
3. If square form is desired, construct it by replacing old source-band rows
   with the finite-element/FV rows, not by dropping the hard audits.
```

### 4.4 What exact unknowns and residual rows should be added next?

Use:

```math
x = \ln R,
\qquad
z = (U,\Theta,m) = (\ln u,\ln T,\ln\dot M).
```

Add true source-element internal states at Lobatto-like or quarter nodes:

```text
a = 0, 1/4, 1/2, 3/4, 1
```

For each source element, add unknowns:

```text
U_a
Theta_a
m_a
```

Optionally add source-element flux variables:

```text
DeltaM_i
DeltaE_i
DeltaJ_i
```

where `i` indexes source-element subintervals.

The source-element state should be represented by polynomial interpolation:

```math
z(x) = \sum_a L_a(x) z_a,
```

```math
z'(x) = \frac{1}{h}\sum_a L'_a(x) z_a.
```

This replaces endpoint-linear quarter-point reconstruction.

Residual rows to add are listed in Section 6.

### 4.5 Should finite-volume energy and angular-momentum balance be introduced inside the source annulus before lowering eta_E?

**Yes.**

The source annulus is where the stream source, local wind mass loss, energy loss, and angular-momentum injection all overlap. It is not enough to enforce only a pointwise differential energy row or only an integrated mass row.

Before lowering `eta_E`, require the `eta_E=90` annulus to satisfy:

```text
finite-volume mass;
pointwise/collocation radial;
pointwise/collocation energy;
finite-volume energy;
finite-volume angular momentum;
interface compatibility.
```

Finite-volume energy should **not** replace the differential/source-band energy audit. Use both.

---

## 5. Recommended source-element unknown layout

### 5.1 Local source-element block

For each source-band macro-element `[x_L, x_R]`, define nodes:

```text
x_0 = x_L
x_1 = x_L + 0.25 h
x_2 = x_L + 0.50 h
x_3 = x_L + 0.75 h
x_4 = x_R
h = x_R - x_L
```

Unknowns:

```text
U_0, U_1, U_2, U_3, U_4
Theta_0, Theta_1, Theta_2, Theta_3, Theta_4
m_0, m_1, m_2, m_3, m_4
```

Optional conservative flux increments:

```text
DeltaM
DeltaE
DeltaJ
```

### 5.2 Interface variables

At the source-block inner and outer boundaries, either:

```text
Option A: share nodes with the global grid.
```

or:

```text
Option B: duplicate nodes and enforce explicit compatibility rows.
```

Given the current interface trouble, **Option B is safer** for the next sprint:

```math
z_{\rm source}(x_a)-z_{\rm global}(x_a)=0,
```

```math
z_{\rm source}(x_b)-z_{\rm global}(x_b)=0.
```

Also enforce flux compatibility:

```math
\dot M_{\rm source}(x_a)-\dot M_{\rm global}(x_a)=0,
```

```math
\dot M_{\rm source}(x_b)-\dot M_{\rm global}(x_b)=0.
```

Use explicit compatibility rows rather than weak anchors for the certification run.

---

## 6. Residual rows to add

### 6.1 Radial momentum collocation rows

At source-element collocation points `q`, e.g.

```text
q = 1/4, 1/2, 3/4
```

evaluate:

```math
r_R(q)
=
\frac{
U'(x_q) - F_R(x_q, z_q)
}{
S_R(x_q)
}.
```

Here `F_R` is the radial-momentum derivative implied by the existing transonic/slim local equations, and `S_R` is the physical scaling already used by the solver.

### 6.2 Energy collocation rows

At the same points:

```math
r_E(q)
=
\frac{
\Theta'(x_q) - F_E(x_q, z_q)
}{
S_E(x_q)
}.
```

This is the production version of the source-band extra energy audit. It must use the polynomial source-element state, not endpoint-linear interpolation.

### 6.3 Finite-volume mass conservation rows

Use actual `Mdot`, not only `m = ln Mdot`:

```math
\dot M_{R} - \dot M_{L}
-
\int_{x_L}^{x_R}
\left(
\dot M'_w - \dot M'_s
\right) dx
=
0.
```

Normalized:

```math
r_M^{FV}
=
\frac{
\dot M_R - \dot M_L
-
\int_{x_L}^{x_R}
(\dot M'_w-\dot M'_s) dx
}{
\dot M_{\rm inner}
}.
```

Keep the sign convention:

```math
\frac{d\dot M}{d\ln R}
=
\dot M'_w-\dot M'_s.
```

So stream source alone makes `Mdot` decrease outward; wind mass loss makes `Mdot` increase outward.

### 6.4 Finite-volume energy balance rows

Inside each source element, add:

```math
r_E^{FV}
=
\frac{
\int_{x_L}^{x_R}
2\pi R^2
\left(
Q_{\rm visc}
+ Q_{\rm stream}
+ Q_{\rm tide}
- Q_{\rm rad}
- Q_{\rm adv}
- Q_{\rm wind}
\right)
dx
}{
\int_{x_L}^{x_R}
2\pi R^2
\left(
|Q_{\rm visc}|
+ |Q_{\rm stream}|
+ |Q_{\rm tide}|
+ |Q_{\rm rad}|
+ |Q_{\rm adv}|
+ |Q_{\rm wind}|
\right)
dx + \epsilon
}.
```

Use it as a production row in the local source-element LS. Still keep the pointwise/collocation energy row as a separate gate.

### 6.5 Finite-volume angular-momentum rows

Inside each source element:

```math
r_J^{FV}
=
\frac{
\left[
\dot M l - G
\right]_R
-
\left[
\dot M l - G
\right]_L
-
\int_{x_L}^{x_R}
\left(
\dot M'_w l_w
-
\dot M'_s l_s
+
\tau_s
\right)dx
}{
J_{\rm scale}
}.
```

Here:

```text
l      = disk specific angular momentum
G      = viscous torque flux
l_s    = stream specific angular momentum
l_w    = wind specific angular momentum
tau_s  = explicit stream torque term, if present
```

Start with the current settings:

```text
l_w = l
current torque_delta_l_fraction handling unchanged
```

Do not add new wind angular-momentum physics yet. This row is a budget/compatibility row for the current closure.

### 6.6 Interface compatibility rows

At source-block edges:

```math
r_I^z = z_{\rm source}-z_{\rm global}.
```

Also include flux compatibility:

```math
r_I^{\dot M}
=
\frac{
\dot M_{\rm source}-\dot M_{\rm global}
}{
\dot M_{\rm inner}
}.
```

If angular-momentum flux is duplicated across the interface, include:

```math
r_I^J
=
\frac{
(\dot M l - G)_{\rm source}
-
(\dot M l - G)_{\rm global}
}{
J_{\rm scale}
}.
```

---

## 7. Algorithmic plan

### Step 1 — Freeze current checkpoints

Freeze these as regression anchors:

```text
N201 halo-4 source-domain:
    production ~3.929e-2
    source audit ~3.088e-2

N251 source-element/refine2 best:
    production ~3.752e-2
    source audit ~3.612e-2
```

The new formulation must beat both **simultaneously**.

### Step 2 — Add local source-element LS mode

Suggested flags:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_LS=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_EXTRA_ROWS=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_EXTRA_AUDIT_ONLY=0
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_POLY=lobatto4
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_FV_MASS=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_FV_ENERGY=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_FV_AM=1
```

The solve should initially be **local**, not fully global:

```text
active variables:
    source-element internal U/Theta/m
    source-block boundary copies
    DeltaM, DeltaE, DeltaJ if implemented
    maybe lambda0 only in later stage
```

Outside source-band nodes should be fixed or weakly anchored during the first local solve.

### Step 3 — Replace endpoint-linear source-band rows

Replace:

```text
endpoint slope + linear yq at quarter points
```

with:

```math
z_q = \sum_a L_a(q) z_a,
```

```math
z'_q = h^{-1}\sum_a L'_a(q) z_a.
```

Then compute radial/energy residuals with the same `F_R`, `F_E`, scaling, and physical terms used by the global solver.

This is the most important implementation change.

### Step 4 — Penalty continuation

Run a penalty ladder:

```text
gamma = 0.03
gamma = 0.10
gamma = 0.30
gamma = 1.0
gamma = 3.0
gamma = 10.0
```

At each `gamma`, polish until the local merit stops improving.

Apply filter acceptance:

```text
accept only if:
    production residual not worse;
    FV mass not worse;
    source radial not worse;
    source energy not worse;
    FV energy not worse;
    FV angular momentum not worse;
    interface rows not worse;
    total merit lower.
```

### Step 5 — Release boundaries and do global polish

Only after local source-element LS reaches at least exploratory quality:

```text
source_band_extra <= 3e-5
interval_mass <= 3e-6
```

release:

```text
source-block boundary states,
nearby halo nodes,
lambda0 if needed,
global state.
```

Run a sparse/damped global polish with source-band production rows still active.

Do **not** let the global polish disable the hard source-band rows.

### Step 6 — Re-audit with all old diagnostics

After the global polish, re-run:

```text
original production residual
source-band extra audit
split_differential
split_rms_differential
rectangular source-band rows
finite-volume mass audit
finite-volume energy audit
finite-volume angular-momentum audit
old midpoint differential audit
```

The old hidden-defect diagnostics must become small, not merely replaced by new rows.

---

## 8. Diagnostics to add

For each source element and each source-element collocation point, output:

```text
R
x = lnR
element id
local coordinate q
U, Theta, m
U_prime_poly
Theta_prime_poly
m_prime_poly
F_R
F_E
F_M
r_R_collocation
r_E_collocation
r_M_collocation if used
source_prime/Mdot
wind_prime/Mdot
Qvisc
Qstream
Qtide
Qrad
Qadv
Qwind
Qwind/Qvisc
Qadv/Qvisc
FV mass residual
FV energy residual
FV angular momentum residual
interface residuals
Jacobian row norm
Jacobian column norm by variable group
```

Also output group norms:

```text
rho_P
rho_M
rho_R
rho_E
rho_J
rho_FVE
rho_I
total weighted merit
unweighted max residual
peak residual radius
peak residual group
```

Include before/after comparison for every penalty stage.

---

## 9. Acceptance criteria before lowering `eta_E`

### 9.1 Hard numerical gates

For `eta_E=90`, require:

```text
production residual              <= 1e-5
interval_mass_residual_max       <= 3e-6
source_band_extra_max            <= 3e-5 exploratory, <=1e-5 certified
source_band_extra_energy_max     <= 3e-5 exploratory, <=1e-5 certified
source_band_extra_radial_max     <= 3e-5 exploratory, <=1e-5 certified
FV energy residual max           <= 3e-5 exploratory, <=1e-5 certified
FV angular-momentum residual max <= 3e-5 exploratory, <=1e-5 certified
interface residual max           <= 1e-5
```

### 9.2 Robustness gates

```text
N201/N251/N301 or equivalent source-element refinements agree.
h-refinement and p-refinement in the source element agree.
No single source-edge cell dominates.
Peak residual does not simply move from one source edge to another.
Old split/rectangular audits no longer show O(1e-2) defects.
Source normalization is preserved.
Mass/source budget closes.
```

### 9.3 Physical stability gates

```text
Mdot_outer/Mdot_inner stable to <= 1e-4 absolute.
Lrad/LEdd stable to <= 0.1–0.3%.
Rson stable to <= 1e-3–1e-2 rg.
f_adv_global stable to <= 1%.
No new sonic-point defect.
No new outer-buffer residual wall.
No H/R or thermodynamic discontinuity inside the source annulus.
```

Only after these pass should Codex continue:

```text
eta_E = 80, 70, 60
```

using the exact same certified source-element formulation.

---

## 10. Rejection criterion for the current closure

If the new source-element formulation still stalls at `O(1e-2)` after:

```text
- production source-band rows;
- polynomial/Lobatto source-element interpolation;
- FV mass;
- FV energy;
- FV angular momentum;
- interface compatibility;
- group-balanced penalty continuation;
- h-refinement and p-refinement;
```

then reject the `eta_E=90` compact-source branch **for the current local-Mdot closure**.

That would not mean the high-`Mdot` branch is gone. It would mean:

```text
This compact source + local mass-loaded wind closure cannot be represented
consistently at eta_E=90 without changing the physical source/wind/boundary
formulation.
```

Possible physical/formulation changes after rejection would include:

```text
- smoother/wider source annulus;
- modified source angular-momentum deposition;
- explicit reservoir/source two-domain model;
- different local wind launch-energy closure;
- wind angular-momentum coupling.
```

But do not move to these until the source-element certification attempt has honestly failed.

---

## 11. Shen & Matzner / power-law wind context

Shen & Matzner-style windy advective disks use a radial accretion profile of the form:

```math
\dot M_{\rm acc}(R) \propto R^s,
\qquad 0 \le s \le 1.
```

This remains a good **calibration/validation prior** for the mass-loaded wind branch, especially for checking the solved source-corrected:

```math
s_{\rm eff}(R)
=
\frac{d\ln \widetilde{\dot M}}{d\ln R}.
```

But it is **not** the immediate numerical fix. The current bottleneck is source-annulus representation. Do not tune `s`, `E_w`, or wind angular momentum to hide an `O(1e-2)` source-band residual.

Once `eta_E=90` is certified, resume the Shen-calibrated wind branch program:

```text
- report s_eff_raw and s_eff_tilde;
- compute Ew_req(R) for target s = 0.1, 0.2, 0.3, 0.5;
- compare eta_E_req and v_inf_req against physical wind priors;
- continue eta_E lower only with the certified source-element formulation.
```

---

## 12. Codex-ready short prompt

```text
Current state at commit 3d5fea8:
- Target: Mdot_inner/Edd=5, Rout=335 rg, Rinj=240 rg,
  f_s=0.80, eta_E=90, compact source annulus, local-Mdot mass-loaded wind.
- eta_E=90 is not source-band certified.
- Old N201 halo-4 source-domain checkpoint:
    production ~3.929e-2
    source-band audit ~3.088e-2
- Best N251 source-element/refine2 result:
    production ~3.752e-2
    source-band audit ~3.612e-2
- Therefore the N251 source-element result is not a certified improvement.

Diagnosis:
- This is not physical branch loss.
- More nodes alone are not enough.
- The compact source annulus still has a coupled mass/radial/energy/interface
  compatibility defect.
- Current source-band quarter-point rows still rely too much on endpoint-linear
  interval representation.
- The same internal source-element states must satisfy mass, radial, energy,
  angular-momentum, and interface rows.

Next implementation:
1. Freeze N201 and N251 bad-but-informative anchors.
2. Add local sparse source-element least-squares mode.
3. Make source-band extra radial/energy rows production rows inside that local
   source-element solve.
4. Replace endpoint-linear quarter-point residuals with polynomial/Lobatto
   source-element interpolation:
       z_q = sum_a L_a(q) z_a
       z'_q = h^-1 sum_a L'_a(q) z_a
5. Keep/upgrade finite-volume mass rows:
       Mdot_R - Mdot_L - int(Mwind_prime - Mstream_prime)dlnR = 0
6. Add finite-volume energy rows:
       int 2*pi*R^2*(Qvisc+Qstream+Qtide-Qrad-Qadv-Qwind)dlnR = 0
7. Add finite-volume angular-momentum rows:
       [Mdot*l - G]_R - [Mdot*l - G]_L
       - int(Mwind_prime*l_w - Mstream_prime*l_s + tau_s)dlnR = 0
8. Add explicit source-block interface compatibility rows for state and fluxes.
9. Use group-balanced penalty continuation:
       gamma = 0.03, 0.10, 0.30, 1.0, 3.0, 10.0
   with filter acceptance:
       mass not worse,
       source radial not worse,
       source energy not worse,
       FV energy not worse,
       FV angular momentum not worse,
       interface not worse,
       production not worse.
10. Release boundaries and globally polish only after local source-element LS
    reaches exploratory quality.
11. Certify eta_E=90 only if:
       production <= 1e-5
       interval_mass <= 3e-6
       source_band_extra <= 3e-5 exploratory, <=1e-5 certified
       FV energy/J <= 3e-5 exploratory, <=1e-5 certified
       interface <= 1e-5
       physical diagnostics stable.
12. Do not lower eta_E to 80/70/60 until eta_E=90 passes these gates.
```

---

## 13. Bottom line

The next sprint should **not** chase `eta_E` lower and should **not** add new wind complexity.

The correct next move is:

```text
local sparse source-element LS
+ production source-band radial/energy rows
+ polynomial/Lobatto internal states
+ FV mass, energy, and angular-momentum rows
+ explicit interface compatibility
+ group-balanced penalty continuation
```

If that succeeds, `eta_E=90` becomes a real certified local-Mdot mass-loaded wind checkpoint. If it fails at `O(1e-2)` even after h/p source-element refinement and balanced constraints, then the current compact-source/local-wind closure should be rejected at `eta_E=90` and the physical source/wind/boundary formulation should be revisited.
