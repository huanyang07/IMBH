# Codex Handoff: Source-Element LS Pilot — Updated Plan for eta_E=90 Source-Annulus Certification

Date: 2026-07-06  
Target commit context: latest Codex update after `CODEX_SOURCE_ELEMENT_LS_PILOT_RESULTS.md`  
Repository: `huanyang07/IMBH`

---

## 0. Executive Summary

The new source-element least-squares (LS) pilot is a **diagnostic success but not yet a production success**.

It confirms that the current `eta_E=90` compact source-annulus solution is **not certified**. The old endpoint-linear source-band audit was still under-resolving the hidden defect. The new polynomial source-element audit exposes a much larger energy defect inside the compact source annulus.

The current LS implementation improves some high-order source-element residual groups when the filter is relaxed, but it worsens the old endpoint source-band audit and leaves the production residual essentially unchanged. Therefore:

```text
Do not lower eta_E.
Do not add new wind complexity.
Do not certify N201 or N251.
Do not just run more LS iterations.
```

The next implementation should be a **true mixed source-element block**:

```text
local element states
+ explicit source-block interface rows
+ flux variables DeltaM / DeltaE / DeltaJ
+ polynomial radial/energy collocation rows
+ finite-volume mass/energy/angular-momentum rows
+ high-order rows as local production rows
+ old endpoint rows retained as audits
```

Final certification should require **all** residual representations to agree: production differential residual, polynomial source-element rows, finite-volume balances, and legacy source-band audits.

---

## 1. Current Target

```text
Mdot_inner/Edd = 5
Rout = 335 rg
Rinj = 240 rg
f_s = 0.80
eta_E = 90
compact source annulus
local-Mdot mass-loaded wind formulation
```

The purpose of this sprint is not to discover a new physical wind branch. It is to decide whether the `eta_E=90` local-Mdot mass-loaded wind checkpoint is a real solution once the compact source annulus is resolved honestly.

---

## 2. What the LS Pilot Implemented

Codex added a disabled-by-default local source-element LS mode in:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

Relevant flags:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_LS=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_LS_GAMMAS=...
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_LS_FV_MASS=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_LS_FV_ENERGY=1
```

Implemented pieces:

```text
- polynomial source-element state/slope evaluation from local 5-node Lagrange stencils;
- pointwise local-Mdot parameters so physics routines see polynomial logMdot and derivative;
- polynomial radial and energy collocation rows;
- finite-volume mass rows using polynomial wind quadrature plus exact stream-source integrals;
- finite-volume energy rows using the same Qvisc/Qstream/Qrad/Qadv/Qwind convention as the differential residual;
- sparse finite-difference Jacobian pattern for local source-element variables;
- gamma continuation and filter acceptance;
- table output for LS group norms.
```

The strict default filter is:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_LS_FILTER_TOL=0.0
```

so an LS step is rejected if it reduces one residual group by worsening another.

---

## 3. New Numerical Evidence

### 3.1 N201 strict pilot

Input checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta90_Nsrc32_source_domain_qm_halo4_seed/stage_00_etaE_90_N201.npz
```

Output table:

```text
outputs/tables/m5_local_mdot_eta90_source_element_ls_poly_strict_N201.json
```

Result:

```text
source_element_ls_applied      False
production final_full          3.9291186e-2
old source_band_extra          3.0880927e-2
poly radial defect             5.9573522e-2
poly energy defect             2.5540674e-1
poly FV mass defect            2.3627608e-2
poly FV energy defect          3.8012403e-2
```

Interpretation:

```text
The polynomial source-element energy defect is far larger than the old endpoint-linear source-band audit.
Therefore the previous audit was still under-resolving the hidden source-annulus defect.
```

### 3.2 N201 relaxed diagnostic

Output table:

```text
outputs/tables/m5_local_mdot_eta90_source_element_ls_poly_pilot_N201.json
```

Used:

```text
SOURCE_ELEMENT_LS_FILTER_TOL=0.02
```

Result:

```text
source_element_ls_applied      True
poly energy defect             2.5540674e-1 -> 2.3792226e-1
poly FV mass defect            2.3627608e-2 -> 2.2149053e-2
poly FV energy defect          3.8012403e-2 -> 3.7588393e-2
old source_band_extra          3.0880927e-2 -> 3.1208403e-2
production full                unchanged at 3.9291186e-2
```

Interpretation:

```text
Relaxed LS improves high-order polynomial/FV groups,
but worsens the old endpoint source-band audit and does not improve production full.
Diagnostic only.
```

### 3.3 N251 strict pilot

Input checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta90_source_element_refine2_global_domain2_eta90/stage_00_etaE_90_N251.npz
```

Output table:

```text
outputs/tables/m5_local_mdot_eta90_source_element_ls_poly_strict_N251.json
```

Result:

```text
source_element_ls_applied      False
production final_full          3.7518660e-2
old source_band_extra          3.6121368e-2
poly radial defect             5.0695160e-2
poly energy defect             1.8723939e-1
poly FV mass defect            1.2499953e-2
poly FV energy defect          7.9317703e-2
```

Interpretation:

```text
N251 reduces poly radial, poly energy, and FV mass relative to N201.
But it worsens FV energy and worsens the old endpoint source-band audit.
Therefore N251 is informative but not certified.
```

### 3.4 N251 relaxed diagnostic

Output table:

```text
outputs/tables/m5_local_mdot_eta90_source_element_ls_poly_relaxed_N251.json
```

Used:

```text
SOURCE_ELEMENT_LS_FILTER_TOL=0.02
```

Result:

```text
source_element_ls_applied      True
poly energy defect             1.8723939e-1 -> 1.8059725e-1
poly FV mass defect            1.2499953e-2 -> 1.2394751e-2
poly FV energy defect          7.9317703e-2 -> 7.2188748e-2
old source_band_extra          3.6121368e-2 -> 3.6775257e-2
production full                unchanged at 3.7518660e-2
```

Interpretation:

```text
Same pattern as N201:
high-order LS groups improve,
legacy source-band audit worsens,
production residual does not improve.
```

---

## 4. Updated Diagnosis

The new source-element LS mode is doing something useful: it exposes a larger hidden compact-source-annulus energy defect than the older endpoint-linear audit.

But the current variable layout and residual coupling are not yet the right production formulation.

Current status:

```text
eta_E=90 source annulus remains uncertified.

The polynomial LS audit is more honest than the old endpoint-linear source-band audit.

The current LS descent direction trades residual groups:
    polynomial energy/FV defects improve,
    old endpoint audit worsens,
    production full remains unchanged.

The failure is not simply lack of nodes.
The failure is a mixed representation/compatibility problem inside the compact source annulus.
```

The most likely issue is that source-band residuals are still not being represented with a **single consistent local element state and flux basis** that production rows, polynomial rows, FV rows, and interface rows all share.

---

## 5. What Changed Relative to the Previous Plan

Previous recommendation:

```text
Promote source-band rows into a local production LS solve.
```

Updated recommendation:

```text
Still promote source-band rows, but not by only turning on the current LS rows globally.

First build a true mixed source-element block:
    element-local states,
    explicit source-block interfaces,
    flux variables,
    polynomial collocation rows,
    FV mass/energy/angular-momentum rows,
    staged rectangular LS locally,
    old endpoint audits as guardrails rather than hard vetoes during development.
```

The current 5-node Lagrange source-element audit should remain, but the production formulation must be made more internally consistent.

---

## 6. Revised Filter Strategy

The strict filter with

```text
SOURCE_ELEMENT_LS_FILTER_TOL=0.0
```

is too blunt for development because the old endpoint audit is now known to be lower-order and under-resolved.

Use a **hierarchical filter**.

### 6.1 Primary groups

These must improve during local source-element development:

```text
poly_E
poly_R
FV_M
FV_E
eventually FV_J
```

### 6.2 Guardrail groups

These may not blow up, but they should not veto every high-order improvement:

```text
old endpoint-linear source_band_extra
original production residual
mass budget
global diagnostics: Mdot_outer/Mdot_inner, Lrad, Rson, f_adv
```

### 6.3 Exploratory acceptance rule

Accept a local exploratory step if:

```text
max(poly_R, poly_E, FV_M, FV_E) decreases by >= 10-20%;

old_source_band_extra <= max(1.10 * old_initial, old_initial + 5e-3);

production_full <= 1.10 * production_initial;

Mdot_outer/Mdot_inner, Lrad/LEdd, Rson remain essentially unchanged.
```

### 6.4 Final certification rule

Final certification remains strict:

```text
production_full <= 1e-5
poly_R <= 1e-5 to 3e-5
poly_E <= 1e-5 to 3e-5
FV_M <= 1e-5 to 3e-5
FV_E <= 1e-5 to 3e-5
FV_J <= 1e-5 to 3e-5
old endpoint source_band_extra <= 1e-5 to 3e-5
```

So:

```text
During development:
    do not let old lower-order audit veto every high-order improvement.

For certification:
    all audits must agree.
```

---

## 7. Recommended Numerical Formulation

Use:

```math
x = \ln R,
\qquad
z = (U,\Theta,m) = (\ln u,\ln T,\ln \dot M).
```

For each source element \(e=[x_L,x_R]\), add true local element nodes, e.g.

```text
q = 0, 1/4, 1/2, 3/4, 1
```

with unknowns:

```text
U_a, Theta_a, m_a  for each local node a
```

Use element-local interpolation:

```math
z(x) = \sum_a L_a(x) z_a,
```

```math
z'(x) = \frac{1}{h_e}\sum_a L'_a(x) z_a,
```

where \(h_e=x_R-x_L\).

This should replace any endpoint-linear quarter-point reconstruction as the production representation inside the source band.

---

## 8. Unknowns to Add Next

For each source element, add:

```text
local states:
    U_a = log u at element node a
    Theta_a = log T at element node a
    m_a = log Mdot at element node a

flux variables:
    DeltaM_e
    DeltaE_e
    DeltaJ_e
```

At the source-block boundaries, duplicate interface states:

```text
z_global_left
z_source_left

z_source_right
z_global_right
```

This allows explicit compatibility rows rather than weak anchors.

Optional, later:

```text
lambda0 or local torque correction inside the source block,
only if angular-momentum FV rows show a systematic mismatch.
```

---

## 9. Residual Rows to Add Next

### 9.1 Polynomial radial collocation rows

At source-element quadrature/collocation points \(x_q\):

```math
r_R(x_q)
=
\frac{U'(x_q)-F_R(x_q,z_q)}{S_R(x_q)}.
```

### 9.2 Polynomial differential-energy rows

```math
r_E(x_q)
=
\frac{\Theta'(x_q)-F_E(x_q,z_q)}{S_E(x_q)}.
```

Use at least:

```text
q = 1/4, 1/2, 3/4
```

or a Gauss-Lobatto/Lobatto set.

### 9.3 Finite-volume mass rows

Use the physical sign convention:

```math
\frac{d\dot M}{d\ln R}
=
\dot M'_w-\dot M'_s.
```

Rows:

```math
r_{\Delta M,e}
=
\Delta M_e
-
\int_{x_L}^{x_R}
(\dot M'_w-\dot M'_s)\,dx
=0,
```

```math
r_{M,e}
=
\dot M_R-\dot M_L-\Delta M_e
=0.
```

### 9.4 Finite-volume energy rows

Use the same convention as the differential energy residual:

```math
r_{\Delta E,e}
=
\Delta E_e
-
\int_{x_L}^{x_R}
2\pi R^2
\left(
Q_{\rm visc}
+
Q_{\rm stream}
-
Q_{\rm rad}
-
Q_{\rm adv}
-
Q_{\rm wind}
\right)
dx
=0.
```

Then either:

```math
\Delta E_e = 0
```

for local steady balance, or connect \(\Delta E_e\) to an explicitly defined advective/enthalpy flux difference if the code has that flux available.

Important: before trusting this row, run a differential/FV identity audit. If the FV energy row and differential energy row are not representing the same equation, they will fight each other.

### 9.5 Finite-volume angular-momentum rows

Inside the source annulus:

```math
r_{\Delta J,e}
=
\Delta J_e
-
\int_{x_L}^{x_R}
\left(
\dot M'_w l_w
-
\dot M'_s l_s
+
\tau_s
\right)
dx
=0,
```

```math
r_{J,e}
=
\left[\dot M l-G\right]_R
-
\left[\dot M l-G\right]_L
-
\Delta J_e
=0.
```

Here:

```text
G = viscous torque flux
l_s = stream specific angular momentum
l_w = wind specific angular momentum
tau_s = explicit stream torque, if present
```

If torque is currently encoded via `torque_delta_l_fraction`, map that convention consistently into \(l_s\) or \(\tau_s\).

### 9.6 Interface compatibility rows

At the source-block boundaries:

```math
r_{\rm int,L}=z_{\rm source}(x_a)-z_{\rm global}(x_a)=0,
```

```math
r_{\rm int,R}=z_{\rm source}(x_b)-z_{\rm global}(x_b)=0.
```

Also enforce flux compatibility:

```math
\dot M_{\rm source}(x_a)-\dot M_{\rm global}(x_a)=0,
```

```math
\dot M_{\rm source}(x_b)-\dot M_{\rm global}(x_b)=0.
```

If angular-momentum flux is available, add:

```math
(\dot M l-G)_{\rm source}-(\dot M l-G)_{\rm global}=0.
```

---

## 10. Differential/FV Consistency Audit

The pilot shows that N251 improves polynomial radial, polynomial energy, and FV mass relative to N201, but worsens FV energy. That suggests the FV energy row may not yet be consistent with the polynomial differential energy row.

Before using FV energy as a hard production row, run this audit per source element:

```math
I_E^{\rm diff}
=
\int r_E^{\rm poly}(x)\,w_E(x)\,dx,
```

```math
I_E^{\rm FV}
=
\int
2\pi R^2
\left(
Q_{\rm visc}
+
Q_{\rm stream}
-
Q_{\rm rad}
-
Q_{\rm adv}
-
Q_{\rm wind}
\right)
dx.
```

Output:

```text
I_E_diff
I_E_FV
I_E_FV - I_E_diff
normalization factors
Qvisc contribution
Qstream contribution
Qrad contribution
Qadv contribution
Qwind contribution
Mdot at quadrature points
dMdot/dlnR at quadrature points
stream_source_prime
wind_prime
```

If \(I_E^{\rm diff}\) and \(I_E^{\rm FV}\) do not agree qualitatively, fix the normalization/equation identity before treating both as production constraints.

---

## 11. Minimal Identity Tests

Before a big global rewrite, run three local source-element identity tests.

### Test 1: No-source/no-wind identity

Use the source-element machinery on a known strict `Mdot=5` no-wind branch, but set inside the source block:

```text
Mdot_stream_prime = 0
Mdot_wind_prime = 0
Qstream = 0
Qwind = 0
```

Expected:

```text
poly_R, poly_E, FV_M, FV_E all small.
```

If not, the source-element machinery itself is wrong.

### Test 2: Source-only weak amplitude

Use the compact source shape, but reduce integrated source fraction:

```text
f_s_test = 0.05, 0.10, 0.20
wind off or eta_E effectively infinite
```

Expected:

```text
poly defects grow smoothly with f_s,
not jump immediately to O(0.1).
```

### Test 3: Source plus weak wind

Run:

```text
eta_E = 100 first,
then eta_E = 90,
same source-element formulation.
```

Expected:

```text
eta_E=100 should be easier than eta_E=90.
If both show O(0.1) polynomial energy defects,
the problem is source-element representation, not eta_E.
```

---

## 12. Staged Algorithmic Plan

### Phase 1 — Freeze LS diagnostic anchors

Freeze:

```text
N201 strict
N201 relaxed
N251 strict
N251 relaxed
```

Record:

```text
production_full
old_source_band_extra
poly_R
poly_E
FV_M
FV_E
Mdot_outer/Mdot_inner
Lrad/LEdd
Rson
f_adv_global
peak residual radius
```

These are the baseline failure cases.

### Phase 2 — Reclassify old endpoint rows

Use old endpoint source-band rows as:

```text
guardrail during development,
hard audit during final certification.
```

Do not use them as an absolute no-worse veto while trying to reduce the new high-order polynomial/FV defects.

### Phase 3 — Build true element-local source states

Move from global-stencil source-element evaluation to actual element-local unknowns.

First implementation:

```text
nodes per source element:
    q = 0, 1/4, 1/2, 3/4, 1

unknowns per node:
    logu
    logT
    logMdot

flux variables:
    DeltaM_e
    DeltaE_e
    DeltaJ_e
```

### Phase 4 — Add explicit source-block interface rows

Duplicate source-block boundary states.

Add state and flux compatibility rows.

This is mandatory because the current LS directions appear to improve internal source rows without improving production/full residuals.

### Phase 5 — Use mixed collocation + FV rows

Local source-element production rows should include:

```text
polynomial radial collocation
polynomial differential energy
finite-volume mass
finite-volume energy
finite-volume angular momentum
interface compatibility
```

Old endpoint rows remain as audits.

### Phase 6 — Staged rectangular LS locally

Use rectangular LS inside the source block first.

Penalty ladder:

```text
gamma_poly = 0.03, 0.10, 0.30, 1.0, 3.0, 10.0
```

At each stage:

```text
primary high-order groups must improve;
legacy audit must stay within guardrail cap;
production full must stay within guardrail cap;
global physics must remain stable.
```

### Phase 7 — Global polish only after local improvement

Do not global-polish a step that only trades residuals.

Local release criteria:

```text
poly_E reduced by factor >= 3 from initial;
poly_R reduced by factor >= 2;
FV_M <= initial and preferably < 1e-2;
FV_E <= initial and preferably < 1e-2;
old_source_band_extra not worse by more than guardrail cap;
production_full not worse by more than guardrail cap.
```

Then release to global polish with source-element rows still active.

### Phase 8 — Certification gate before lowering eta_E

Do not lower \(\eta_E\) until `eta_E=90` passes the certification gate in Section 13.

---

## 13. Acceptance Criteria

### Exploratory improvement

A local source-element step can be called an exploratory improvement if:

```text
max(poly_R, poly_E, FV_M, FV_E) decreases by >= 10-20%;
old_source_band_extra <= max(1.10 * old_initial, old_initial + 5e-3);
production_full <= 1.10 * production_initial;
Mdot_outer/Mdot_inner stable to <=1e-4 to 1e-3;
Lrad/LEdd stable to <=0.3%;
Rson stable to <=1e-2 rg;
no new sonic or outer-buffer wall.
```

### Local-release improvement

Release from local source-element solve to global polish only if:

```text
poly_E reduced by factor >= 3;
poly_R reduced by factor >= 2;
FV_M not worse and preferably < 1e-2;
FV_E not worse and preferably < 1e-2;
old_source_band_extra within guardrail cap;
production_full within guardrail cap;
physical diagnostics stable.
```

### eta_E=90 certification

Do not continue to `eta_E=80, 70, 60` until:

```text
production_full <= 1e-5

poly_E <= 3e-5 exploratory, <=1e-5 certified
poly_R <= 3e-5 exploratory, <=1e-5 certified
FV_M   <= 3e-5 exploratory, <=1e-5 certified
FV_E   <= 3e-5 exploratory, <=1e-5 certified
FV_J   <= 3e-5 exploratory, <=1e-5 certified

old endpoint source_band_extra <= 3e-5 exploratory, <=1e-5 certified
```

Physical stability:

```text
Mdot_outer/Mdot_inner stable to <=1e-4 absolute
Lrad/LEdd stable to <=0.1-0.3%
Rson stable to <=1e-3 to 1e-2 rg
f_adv_global stable to <=1%
no new sonic-point defect
no new outer-buffer wall
```

Representation robustness:

```text
N/source-element refinement improves or plateaus monotonically
p-refinement and h-refinement agree
peak residual does not jump between source edges
no single source-edge cell dominates
legacy and polynomial audits agree at certification
```

---

## 14. When to Reject eta_E=90 for the Current Closure

If the mixed source-element formulation still stalls at \(O(10^{-2})\) after:

```text
true element-local states,
explicit interface rows,
flux variables,
polynomial radial/energy collocation,
FV mass,
FV energy,
FV angular momentum,
balanced penalty continuation,
h-refinement,
p-refinement,
and global polish with source-element rows active,
```

then reject the `eta_E=90` compact-source branch **for the current local-Mdot closure**.

That rejection would not mean the high-`Mdot` branch is physically gone. It would mean this specific compact-source + local-wind closure cannot be represented consistently at this wind loading without changing the source/closure formulation.

---

## 15. Relation to Shen & Matzner / Wind Power-Law Context

Shen & Matzner 2014 remain relevant as a **physics prior** for the eventual mass-loaded wind branch. Their windy advective disk framework uses a radial wind/accretion parameterization of the form:

```math
\dot M_{\rm acc}(R) \propto R^s,
\qquad 0 \le s \le 1,
```

and includes wind angular-momentum effects through the wind lever arm.

For this project, use the Shen-style power law as:

```text
a calibration and validation family for solved Mdot(R),
not as a hard imposed final profile.
```

However, the immediate blocker at `eta_E=90` is **not** the choice of \(s\), \(\eta_E\), wind lever arm, or heating. It is the compact source-annulus representation.

So:

```text
Do not tune Shen-style s or Ew to hide the source-annulus defect.
Do not lower eta_E yet.
First certify eta_E=90 source-annulus representation.
```

---

## 16. Codex-Ready Prompt

```text
Current status:
- Target:
    Mdot_inner/Edd = 5
    Rout = 335 rg
    Rinj = 240 rg
    f_s = 0.80
    eta_E = 90
    compact source annulus
    local-Mdot mass-loaded wind
- The eta_E=90 branch is not certified.
- New source-element LS mode is diagnostic but not production-successful.
- N201 strict:
    production = 3.929e-2
    old source_band_extra = 3.088e-2
    poly_R = 5.957e-2
    poly_E = 2.554e-1
    FV_M = 2.363e-2
    FV_E = 3.801e-2
- N251 strict:
    production = 3.752e-2
    old source_band_extra = 3.612e-2
    poly_R = 5.070e-2
    poly_E = 1.872e-1
    FV_M = 1.250e-2
    FV_E = 7.932e-2
- Relaxed LS reduces polynomial/FV groups but worsens old endpoint audit
  and leaves production full unchanged.
- Therefore the new polynomial audit exposes a bigger hidden source-annulus
  energy defect than the old endpoint-linear audit.

Do not lower eta_E.
Do not add wind complexity.
Do not just run more LS iterations.

Next implementation:
1. Build a true mixed source-element block.
2. Use actual element-local nodes, not only global-stencil polynomial evaluation:
       q = 0, 1/4, 1/2, 3/4, 1
       unknowns per node = logu, logT, logMdot.
3. Add flux variables per source element:
       DeltaM_e, DeltaE_e, DeltaJ_e.
4. Add explicit source-block interface rows:
       state continuity and flux continuity at source block edges.
5. Use polynomial collocation rows as local production rows:
       r_R = U' - F_R
       r_E = Theta' - F_E.
6. Keep finite-volume mass rows:
       DeltaM_e - int(Mwind_prime - Mstream_prime)dlnR = 0
       Mdot_R - Mdot_L - DeltaM_e = 0.
7. Add finite-volume energy rows:
       DeltaE_e - int 2*pi*R^2*(Qvisc+Qstream-Qrad-Qadv-Qwind)dlnR = 0.
   First run a differential/FV energy identity audit to ensure this row is
   consistent with the differential energy equation.
8. Add finite-volume angular-momentum rows:
       DeltaJ_e - int(Mwind_prime*l_w - Mstream_prime*l_s + tau_s)dlnR = 0
       [Mdot*l - G]_R - [Mdot*l - G]_L - DeltaJ_e = 0.
9. Use staged rectangular LS locally:
       gamma = 0.03, 0.10, 0.30, 1.0, 3.0, 10.0.
10. Use hierarchical filter:
       primary groups poly_E/poly_R/FV_M/FV_E/FV_J must improve;
       old endpoint audit and production full are guardrails during development;
       all audits must be small for final certification.
11. Run identity tests:
       no-source/no-wind,
       weak source-only,
       eta_E=100 then eta_E=90.
12. Certify eta_E=90 only if:
       production <= 1e-5,
       poly_R/poly_E/FV_M/FV_E/FV_J <= 3e-5 exploratory or 1e-5 certified,
       old source_band_extra <= 3e-5 exploratory or 1e-5 certified,
       physical diagnostics stable.
13. Only after eta_E=90 is certified should eta_E be lowered to 80, 70, 60.
```

---

## 17. Bottom Line

The new LS pilot is valuable because it finally reveals the real source-annulus defect more honestly:

```text
old endpoint audit: O(3e-2)
new polynomial energy audit: O(0.2)
```

But the current LS implementation is not yet a certified formulation because it trades residual groups and does not improve the production residual.

The correct next step is not more continuation. It is a mixed local source-element formulation with:

```text
true element states,
explicit interfaces,
flux variables,
polynomial radial/energy collocation,
finite-volume mass/energy/angular momentum,
balanced hierarchical penalty continuation,
and all old/new audits retained.
```

Only then should Codex attempt to lower `eta_E` below 90.
