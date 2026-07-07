# Codex Handoff — Source-Plus-Buffer Interface Formulation: Evaluation and Next Plan

Date: 2026-07-06  
Repository/commit reviewed: `huanyang07/IMBH@99e1fdd`  
Main note reviewed: `Note/CODEX_SOURCE_PLUS_BUFFER_INTERFACE_RESULTS.md`  
Primary script: `scripts/run_mdot5_local_mdot_eta_continuation.py`

## 1. Executive summary

The new source-plus-buffer interface formulation is an important implementation step, but it has **not yet solved the source-annulus representation defect**.

The key result is:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
Rinj = 240 rg
f_s = 0.80
eta_E = 100
compact C2 source
torque_delta_l_fraction = +0.005
N = 164
```

The augmented production-polish run is strict globally:

```text
output:
  outputs/tables/m5_source_plus_buffer_production_eta100_N164_bandonly_nfev8.json

final_full = 9.354e-6
accepted strict = yes
source_plus_buffer_production_applied = true
alpha = 2.4414e-4
```

but the actual source-plus-buffer compatibility barely moves:

```text
selected source-plus-buffer residual:
  0.0484758 -> 0.0484633

energy compatibility:
  0.00712424 -> 0.00712378

source-band extra:
  0.1593905 -> 0.1593848
```

The undamped augmented candidate would improve the source-band residual much more:

```text
candidate_selected ~ 0.02295
candidate_source_band_extra ~ 0.13741
```

but it breaks the global differential BVP badly:

```text
candidate_full ~ 1.523e-2
```

So the current result says:

> The new compatibility machinery works numerically, but useful source-band correction directions are almost orthogonal to, or directly in conflict with, the old strict global differential residual.

Do **not** loosen the strict guard just to accept larger local improvements. That would only hide the defect.

The next step should be a **band-local residual replacement / homotopy formulation** where the old pointwise differential rows in the source-plus-buffer band are gradually replaced by mathematically equivalent integrated/cumulative rows. This is better than simply appending more rows to the old production residual.

---

## 2. Current status of the project

The broader project has already established several important anchors:

```text
1. Standard no-wind high-Mdot slim branch exists to Mdot/Edd = 5.
2. Finite stream-fed Mdot_inner/Edd = 5 branch exists in no-wind / energy-wind forms.
3. Energy-only wind continuation is numerically robust but not physically complete because it removes energy without solving mass loss.
4. Local-Mdot mass-loaded wind is the right physical direction.
5. The current blocker is source-annulus representation inside the compact C2 stream source band.
```

The previous source-element LS pilot already showed that the old endpoint-linear source-band audit was under-resolving the problem. Polynomial source-element diagnostics exposed much larger hidden defects, especially in energy. The new source-plus-buffer formulation continues that story: it gives a more compatible way to compare endpoint increments, source-interface finite-volume integrals, source-element polynomial/Simpson integrals, and production mass rows, but the state still cannot be corrected without violating the old differential BVP.

Therefore:

```text
eta_E = 100:
    globally strict under old differential residual,
    not source-plus-buffer/source-element compatible.

eta_E = 90:
    should not be retried seriously until eta_E=100 is source-band certified.

eta_E = 80,70,60:
    do not continue yet.
```

---

## 3. Main diagnosis

This is not a physical branch loss.

It is also not primarily a mesh-count problem anymore. The pathology has survived:

```text
- source micro-domain attempts,
- source-element node refinement,
- local LS diagnostics,
- source-plus-buffer interface variables,
- augmented production polish.
```

The strongest diagnosis is:

> The old production residual and the new source-plus-buffer compatibility residuals are not yet representing the same source-band equations in the same variables.

The current augmented polish appends source-plus-buffer rows to the old production residual. That is useful as an audit, but it asks one state vector to satisfy two incompatible discretizations at the source annulus.

This is why the line search behaves like this:

```text
large alpha:
    improves source-band compatibility,
    destroys old global differential residual.

small alpha:
    preserves old global residual,
    barely improves source-band compatibility.
```

That is a structural signal. It is not just poor damping.

---

## 4. Answer to the candidate next steps

### 4.1 Analytic/local derivatives for source-interface and source-element energy rows

**Yes, implement them, but do not expect this alone to solve the floor.**

Analytic or semi-analytic local derivatives are needed because the source-band rows are numerically delicate and the finite-difference Jacobian may be giving a noisy correction direction. The note says the optional hybrid Jacobian currently injects exact analytic entries for cumulative mass/energy increment columns, but the path was too slow for interactive validation. That is still worth improving.

However, the current line-search pattern is too systematic to blame only on noisy derivatives. The undamped source-band direction improves the local rows but violates the old global residual by orders of magnitude. Better derivatives may make the direction cleaner, but they will not fix an inconsistent residual formulation.

Recommendation:

```text
Implement analytic/local derivatives as Step 1,
but treat them as enabling infrastructure, not the final fix.
```

### 4.2 Row-local production residual pre-polish using only source-band/base rows

**Yes, as a diagnostic and preconditioner. Not as certification.**

A row-local pre-polish should be used to ask:

```text
Can the source-band rows be reduced if we stop letting far-field global rows dominate the local solve?
```

But the line search must not certify the result unless the final global residual, under the chosen production formulation, is also strict.

Recommendation:

```text
Use row-local pre-polish with source-band/base rows,
then global polish using the same active source-band formulation.
Do not accept a local-only improvement as a physical checkpoint.
```

### 4.3 True global residual reformulation with cumulative increment variables

**Yes. This is the main next step.**

The current implementation added cumulative endpoint variables:

```text
mass_cum
energy_cum
```

and source-plus-buffer rows. But they are still appended to the old production residual. The next step should make them part of the **actual production formulation** in the source-plus-buffer band.

The key idea:

```text
Outside the source-plus-buffer band:
    keep the original differential residual.

Inside the source-plus-buffer band:
    replace old pointwise mass/energy rows with integrated/cumulative rows,
    using a homotopy parameter chi from old to new.
```

This avoids the current trap where the solver is required to satisfy two discretizations simultaneously.

### 4.4 Better scaling / normalization of source-band energy compatibility

**Yes, but only after a representation identity audit.**

The current selected source-plus-buffer residual is around `0.048`, while energy compatibility is around `0.007`, and source-band extra is around `0.159`. These numbers are not obviously commensurate. Before changing weights blindly, Codex should output the dimensional numerators and denominators used in every normalization.

Recommendation:

```text
First audit normalizations and row units.
Then equilibrate rows by physical scale or row-Jacobian norm.
Always report unweighted residual groups as final diagnostics.
```

### 4.5 Alternative source annulus collocation or finite-volume formulation

**Yes. The best version is a mixed integral/collocation formulation.**

The source annulus is where mass, energy, angular momentum, and wind/source terms overlap. It should not be treated purely by pointwise midpoint residuals. But a broad integrated residual can hide localized defects.

Recommended compromise:

```text
- Use integrated endpoint-compatible rows for mass, radial momentum, and energy.
- Use source-element polynomial or Lobatto quadrature inside the band.
- Keep pointwise/high-order source-band rows as audits and, later, optional production rows.
- Add finite-volume angular-momentum rows before lowering eta_E.
```

---

## 5. Most important formulation correction

The current `energy_cum` concept needs a careful check.

If `energy_cum` represents a physical heat/cool numerator integral, then setting it to close locally may not be equivalent to the old differential temperature/entropy equation. A source annulus with advection does **not** necessarily satisfy a naive local integral condition of the form:

```text
integral(Qvisc + Qstream - Qrad - Qadv - Qwind) = 0
```

unless the advective flux terms and endpoint entropy/enthalpy changes are represented consistently.

A safer production energy residual is the **integral form of the same ODE** used by the old differential residual.

Let

```math
x = \ln R,
\qquad
z = (U,\Theta,m) = (\ln u,\ln T,\ln \dot M).
```

Suppose the original ODE system is:

```math
U' = F_R(x,z),
```

```math
\Theta' = F_E(x,z),
```

```math
m' = F_M(x,z)
     = \frac{\dot M'_w - \dot M'_s}{\dot M}.
```

Then inside each source-band interval `[x_i,x_{i+1}]`, use endpoint-compatible integral rows:

```math
R^R_i
=
(U_{i+1}-U_i)
-
\int_{x_i}^{x_{i+1}} F_R(x,z(x))\,dx,
```

```math
R^E_i
=
(\Theta_{i+1}-\Theta_i)
-
\int_{x_i}^{x_{i+1}} F_E(x,z(x))\,dx.
```

These rows are directly equivalent to the original differential residual in the continuum limit. They are much less ambiguous than an energy numerator row that is not tied to endpoint `logT` or entropy.

The physical energy numerator should still be audited:

```math
N^E_i
=
\int_{x_i}^{x_{i+1}}
2\pi R^2
\left(
Q_{visc}+Q_{stream}-Q_{rad}-Q_{adv}-Q_{wind}
\right)dx.
```

But before using `N^E_i` as a production row, Codex should verify that it is mathematically equivalent to the `F_E` residual under the same variables and normalization.

---

## 6. Recommended new production formulation

### 6.1 Band-local replacement homotopy

Define the source-plus-buffer band `B` as the compact source support plus a small buffer on both sides. Outside `B`, keep the old production residual.

Inside `B`, define a homotopy residual:

```math
R_B(\chi)
=
(1-\chi) R_{old,B}
+
\chi R_{new,B},
\qquad 0\le \chi\le 1.
```

Here `R_new,B` contains the source-plus-buffer integral/cumulative rows, not merely extra rows appended to the old rows.

Suggested ladder:

```text
chi = 0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.00
```

At each `chi`, solve to strict tolerance under the **homotopy residual**. Keep the old differential residual as an audit, but do not let it veto every movement if it is exactly the representation being replaced.

This is the clean way to determine whether the old differential residual is a real physical constraint or a source-band discretization artifact.

### 6.2 Unknowns

Use the global state plus band-local cumulative increments.

For each selected band node:

```text
logu_i
logT_i
logMdot_i
```

For each source-plus-buffer interval:

```text
DeltaM_i        mass increment
DeltaU_i        radial/logu increment, optional
DeltaTheta_i    energy/logT increment
DeltaJ_i        angular-momentum flux increment, optional first as audit
```

The current `mass_cum` / `energy_cum` can remain useful, but the energy variable should be tied explicitly to an endpoint-compatible temperature/entropy increment, not only to an energy numerator.

### 6.3 Mass rows

Use inward-positive accretion rate:

```math
\frac{d\dot M}{d\ln R}=\dot M'_w-\dot M'_s.
```

For each interval:

```math
R^{M,endpoint}_i
=
\dot M_{i+1}-\dot M_i-\Delta M_i,
```

```math
R^{M,source}_i
=
\Delta M_i
-
\int_{x_i}^{x_{i+1}}
(\dot M'_w-\dot M'_s)\,dx.
```

Use exact stream-source integrals where possible and polynomial/quadrature wind integrals at the same quadrature points used for the energy/radial rows.

### 6.4 Radial rows

Use integral radial rows in the band:

```math
R^R_i
=
(U_{i+1}-U_i)
-
\int_{x_i}^{x_{i+1}} F_R(x,z(x))\,dx.
```

Use Lobatto or Simpson quadrature. Also audit the pointwise residual at:

```text
q = 1/4, 1/2, 3/4
```

### 6.5 Energy rows

Use an endpoint-compatible ODE-integral row:

```math
R^E_i
=
(\Theta_{i+1}-\Theta_i)
-
\int_{x_i}^{x_{i+1}} F_E(x,z(x))\,dx.
```

Add physical energy numerator audits:

```math
N^E_{interface,i}
=
\int_{x_i}^{x_{i+1}}
2\pi R^2
(Q_{visc}+Q_{stream}-Q_{rad}-Q_{adv}-Q_{wind})\,dx
```

using the source-interface view, and

```math
N^E_{element,i}
=
\int_{x_i}^{x_{i+1}}
2\pi R^2
(Q_{visc}+Q_{stream}-Q_{rad}-Q_{adv}-Q_{wind})\,dx
```

using the source-element polynomial/Simpson view.

Then enforce or audit compatibility:

```math
R^{E,compat}_i
=
N^E_{interface,i}-N^E_{element,i}.
```

But do **not** set `energy_cum = 0` as a hard production row unless the advective endpoint flux is included consistently.

### 6.6 Angular momentum rows

Before lowering `eta_E`, add angular-momentum balance in the source annulus.

Define the viscous angular-momentum flux:

```math
F_J = \dot M l - G,
```

where `G` is the viscous torque flux. Then use:

```math
R^J_i
=
[F_J]_{i+1}-[F_J]_i
-
\int_{x_i}^{x_{i+1}}
(\dot M'_w l_w-\dot M'_s l_s+\tau_s)\,dx.
```

At first, this can be an audit row. Promote it to production if it is comparable to the mass/energy source defects. The source annulus is exactly where angular-momentum incompatibility can masquerade as radial or energy residual.

---

## 7. Derivative and scaling plan

### 7.1 Analytic/local Jacobian entries

Add local analytic or semi-analytic derivatives for the band rows with respect to:

```text
logu_i
logT_i
logMdot_i
DeltaM_i
DeltaTheta_i
DeltaJ_i
```

At minimum, add exact derivatives for:

```text
mass endpoint rows wrt logMdot and DeltaM
mass source rows wrt DeltaM
energy endpoint rows wrt logT and DeltaTheta
radial endpoint rows wrt logu and DeltaU
compatibility rows wrt cumulative increment variables
```

For the expensive physics terms, use local central or complex-step finite differences until full analytic derivatives are available.

### 7.2 Directional derivative audit

For each candidate correction direction `delta`, verify:

```math
\frac{R(z+\epsilon\delta)-R(z)}{\epsilon}
\approx
J\delta
```

for:

```text
epsilon = 1e-4, 1e-5, 1e-6, 1e-7
```

Report relative error by residual group:

```text
old production band
new mass rows
new radial rows
new energy rows
energy compatibility rows
physical energy numerator audits
```

If the direction derivative is poor for source-band energy rows, do not interpret line-search failure physically.

### 7.3 Scaling

Report both weighted and unweighted norms.

Suggested row groups:

```text
G_old_outside
G_old_band
G_mass_endpoint
G_mass_source
G_radial_integral
G_energy_ode_integral
G_energy_physical_compat
G_angular_momentum
G_interface
```

Use row scaling such as:

```math
\tilde r_i = \frac{r_i}{\max(S_i,\epsilon)}
```

where `S_i` is either a physical scale or a row-Jacobian norm. Do not let the optimizer reduce one group by hiding another. Use a filter:

```text
accept if:
    primary homotopy residual decreases,
    mass group does not worsen beyond cap,
    energy compatibility does not worsen beyond cap,
    physical diagnostics remain stable.
```

---

## 8. Minimal diagnostic experiments

### Experiment A — no-source/no-wind identity test

Use the source-plus-buffer code on a known strict branch but turn off stream and wind:

```text
Mdot_stream_prime = 0
Mdot_wind_prime = 0
Qstream = 0
Qwind = 0
```

Expected result:

```text
old differential rows strict
new integral rows strict
mass cumulative rows zero
energy compatibility rows zero
source-element polynomial rows small
```

If this fails, the source-plus-buffer machinery is internally inconsistent.

### Experiment B — weak compact source test

Use the compact source but weak source fraction:

```text
f_s = 0.05, 0.10, 0.20
eta_E = infinity or wind off
```

Expected result:

```text
source-plus-buffer defects grow smoothly with source strength,
not jump to O(0.05-0.1).
```

### Experiment C — eta_E=100 homotopy

Use the current strict checkpoint:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
Rinj = 240 rg
f_s = 0.80
eta_E = 100
N = 164
```

Run the band-replacement homotopy:

```text
chi = 0.00 -> 1.00
```

Acceptance for each homotopy stage:

```text
homotopy residual <= 1e-5
mass residual <= 3e-6
physical diagnostics stable
source-plus-buffer selected residual decreases monotonically or nearly so
```

### Experiment D — old-vs-new residual contradiction test

At `chi=1`, if the new source-plus-buffer residual is strict but the old source-band differential residual is not, run representation audits:

```text
old midpoint residual
split residual
Simpson residual
source-element polynomial residual
finite-volume mass/energy/angular momentum
```

Interpretation:

```text
If all high-order/integral representations agree and only old midpoint rows fail:
    old rows are representation artifacts in the source band.

If high-order/integral rows disagree with each other:
    source formulation is still inconsistent.

If all rows remain large:
    current eta_E/source closure may be physically incompatible.
```

---

## 9. Acceptance criteria before retrying eta_E=90

First certify `eta_E=100` under the source-plus-buffer formulation.

Exploratory acceptance:

```text
homotopy/new production residual <= 1e-5
mass endpoint/source rows <= 3e-6
energy ODE-integral rows <= 3e-5
radial ODE-integral rows <= 3e-5
energy compatibility <= 3e-5
source-element poly audits <= 3e-5 to 1e-4
physical energy numerator audit understood
```

Certified acceptance:

```text
homotopy/new production residual <= 1e-5
mass endpoint/source rows <= 1e-6 to 3e-6
energy ODE-integral rows <= 1e-5
radial ODE-integral rows <= 1e-5
energy compatibility <= 1e-5
source-element poly audits <= 1e-5
FV angular momentum <= 1e-5 to 3e-5
```

Physical stability gates:

```text
Mdot_outer/Mdot_inner stable to <=1e-4 absolute
Lrad/LEdd stable to <=0.1-0.3 percent
Rson stable to <=1e-3 to 1e-2 rg
f_adv_global stable to <=1 percent
no new sonic defect
no new outer-buffer wall
no residual peak jumping between source edges
```

Only after this should Codex retry:

```text
eta_E = 90
```

Then apply the same gates. Only after `eta_E=90` is source-band certified should Codex continue:

```text
eta_E = 80, 70, 60
```

---

## 10. Concrete implementation order

### Step 1 — Freeze current 99e1fdd anchors

Freeze these outputs:

```text
m5_source_plus_buffer_eta100_N164_smoke
m5_source_plus_buffer_eta100_N164_bandonly_state0
m5_source_plus_buffer_eta100_N164_bandonly_prodM
m5_source_plus_buffer_eta100_N164_bandonly_prodM_preserve
m5_source_plus_buffer_production_eta100_N164_bandonly_nfev8
```

Track:

```text
final_full
selected source-plus-buffer residual
source-band extra
mass group
energy group
energy compatibility
candidate selected
candidate full
accepted alpha
Mdot_outer/Mdot_inner
Lrad/LEdd
Rson
```

### Step 2 — Add residual identity audit

For every source-plus-buffer interval, output:

```text
R_left, R_mid, R_right
old differential mass/radial/energy residuals
integrated ODE mass/radial/energy residuals
source-interface FV mass/energy numerator
source-element polynomial FV mass/energy numerator
mass_cum
energy_cum or DeltaTheta
normalization denominators
quadrature weights
Qvisc, Qstream, Qrad, Qadv, Qwind
Mdot_wind_prime, Mdot_stream_prime
```

The purpose is to identify exactly which two views disagree.

### Step 3 — Add endpoint-compatible energy increment

Add a `DeltaTheta` or entropy increment variable and rows:

```math
\Theta_{i+1}-\Theta_i-\Delta\Theta_i=0,
```

```math
\Delta\Theta_i-\int_i^{i+1}F_E(x,z(x))dx=0.
```

Keep physical energy numerator rows as audits until equivalence is proven.

### Step 4 — Implement band-replacement homotopy

Add flags such as:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_REPLACE_BAND=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_CHI=...
```

Inside the source-plus-buffer band, scale down old pointwise mass/energy rows and scale up new integrated rows.

Outside the band, keep the old residual unchanged.

### Step 5 — Add analytic/semi-analytic Jacobian entries

Start with exact derivatives for auxiliary increment rows and local central/complex-step derivatives for expensive physics terms.

Keep a `J delta` directional derivative audit.

### Step 6 — Row-local pre-polish under the new residual

Run a local source-band pre-polish using the homotopy residual, not merely appended source rows. Then release to global polish with the same rows active.

### Step 7 — Add FV angular-momentum audit/rows

Add angular-momentum balance inside the source annulus before lowering `eta_E`.

### Step 8 — Certify eta_E=100, then eta_E=90

Do not continue lower until both pass the source-band gates.

---

## 11. Codex-ready prompt

```text
Current state at commit 99e1fdd:
- A source-plus-buffer interface formulation has been implemented in
  scripts/run_mdot5_local_mdot_eta_continuation.py.
- It adds local block variables logu/logT/logMdot and cumulative mass/energy
  increments mass_cum/energy_cum.
- It compares endpoint increments, source-interface FV integrals,
  source-element polynomial/Simpson integrals, and production mass rows.
- It also adds an augmented production polish mode with default variable mode
  SOURCE_PLUS_BUFFER_PRODUCTION_VARIABLE_MODE=band.

Validation case:
- Mdot_inner/Edd = 5
- Rout = 335 rg
- Rinj = 240 rg
- f_s = 0.80
- compact C2 source
- torque_delta_l_fraction = +0.005
- eta_E = 100
- N = 164

Key output:
outputs/tables/m5_source_plus_buffer_production_eta100_N164_bandonly_nfev8.json

Result:
- final_full = 9.354e-6, strict
- source_plus_buffer_production_applied = true
- alpha = 2.4414e-4
- selected source-plus-buffer residual improves only
    0.0484758 -> 0.0484633
- energy compatibility improves only
    0.00712424 -> 0.00712378
- undamped candidate would improve source-band selected residual to ~0.02295
  but gives candidate_full ~1.523e-2, so strict line search rejects it.

Interpretation:
- The source-plus-buffer machinery works numerically.
- It still does not solve the source-band representation defect.
- Useful local compatibility directions strongly conflict with the old strict
  global differential residual.
- Do not loosen the strict guard just to accept source-band improvements.

Recommended next implementation:
1. Add a residual identity audit comparing:
   - old differential rows;
   - source-interface FV integrals;
   - source-element polynomial/Simpson integrals;
   - endpoint-compatible integral ODE rows;
   - physical energy numerator audits.

2. Add endpoint-compatible energy increment variables. Do not use energy_cum as
   a hard local energy-balance closure unless it is tied to endpoint entropy or
   logT change. Use rows:
       Theta_{i+1} - Theta_i - DeltaTheta_i = 0
       DeltaTheta_i - int_i^{i+1} F_E(x,z) dx = 0

3. Implement source-band residual replacement/homotopy:
       R_B(chi) = (1-chi) R_old_band + chi R_new_band
   with chi = 0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0.
   Outside the band, keep the old residual.
   Inside the band, replace old pointwise mass/energy rows with integrated
   cumulative rows instead of appending more rows to the old system.

4. Add analytic or semi-analytic local Jacobian entries for source-plus-buffer
   rows, especially energy rows, and run a J*delta directional derivative audit.

5. Use row-local pre-polish only under the new homotopy residual, then release
   to global polish with the same source-band formulation active.

6. Add finite-volume angular-momentum audit/rows in the source annulus before
   lowering eta_E.

7. Certification before retrying eta_E=90:
   - eta_E=100 must pass source-plus-buffer representation gates first.
   - homotopy/new production residual <= 1e-5;
   - mass rows <= 3e-6;
   - energy/radial integral rows <= 3e-5 exploratory, <=1e-5 certified;
   - energy compatibility <= 3e-5 exploratory, <=1e-5 certified;
   - FV angular momentum <= 3e-5 exploratory;
   - Mdot_outer/Mdot_inner, Lrad/LEdd, Rson stable;
   - no new sonic or outer-buffer residual wall.

8. Only after eta_E=100 and eta_E=90 are source-band certified should eta_E be
   lowered to 80, 70, or 60.
```

---

## 12. Bottom line

The latest commit did the right thing by implementing a compatible source-plus-buffer layer and a guarded augmented production polish. The result is useful because it shows the defect more clearly.

But the key lesson is:

```text
Appending source-plus-buffer compatibility rows to the old strict differential
residual is not enough. The source band needs a replacement/homotopy residual
that uses endpoint-compatible integral mass/radial/energy equations.
```

The best next step is:

```text
residual identity audit
+ endpoint-compatible energy increment
+ band-local residual replacement homotopy
+ analytic/semi-analytic local Jacobian
+ FV angular-momentum audit
```

Do not lower `eta_E` until `eta_E=100`, then `eta_E=90`, pass this source-band certification.
