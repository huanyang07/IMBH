# Codex Handoff: Mdot=5 Local-Mdot Eta_E=100 Residual-Floor Assessment and Next Plan

Date: 2026-07-05
Repository context: `huanyang07/IMBH`, latest reviewed GitHub state around the Mdot=5 local-`Mdot(R)` eta continuation sprint.

Primary files reviewed:

```text
Note/GPT_PROMPT_MDOT5_LOCAL_MDOT_ETA_RESIDUAL_FLOOR.md
Note/CODEX_MDOT5_LOCAL_MDOT_ETA_CONTINUATION_RESULTS.md
Note/CODEX_MDOT5_SHEN_DIAGNOSTIC_SPRINT_RESULTS.md
Note/CODEX_MDOT5_MASS_WIND_STATUS_AND_NEXT_PLAN_UPDATED_SHEN2014.md
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
scripts/run_mdot5_local_mdot_eta_continuation.py
```

Literature context:

```text
Shen & Matzner 2014, ApJ 784, 87, especially Section 4 and Appendix A.
They use a windy advective disk prior Mdot_acc(R) ∝ R^s, 0 <= s <= 1,
and include wind mass and angular-momentum loss in Appendix A.
```

---

## 1. Executive assessment

The latest result is real progress. The local-`Mdot(R)` mass-loaded wind BVP is no longer failing because of the local mass equation itself. The bad remap/refinement pathology was mostly repaired by targeted node-preserving nested insertion over `R = 100–320 rg`. The best eta_E=100 checkpoint now has:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
Rinj = 240 rg
f_s = 0.80
eta_E = 100
zeta anchor = 0.03 Shen-style weak mass-wind bridge

best repaired checkpoint:
    full differential residual = 2.075e-05
    interval_R = 2.075e-05 near R ~ 300.49 rg
    interval_E = 6.865e-06 near R ~ 7.83 rg
    mass_residual_max = 1.746e-06
    outer_omega = -1.335e-05
    Mdot_outer/Mdot_inner = 0.232809
    f_adv_global = -0.00389086
    Lrad/LEdd = 0.527513
    Rson = 5.29806 rg
```

Interpretation:

```text
This is not branch death.
This is not a gross local mass-loading failure.
This is not primarily an energy residual problem anymore.
This is a localized radial-momentum/collocation residual floor near the source/outer transition.
```

The immediate objective should be narrow and honest:

```text
Make eta_E=100 strict under the original physical differential audit,
without accepting an integrated residual that hides the localized radial defect.
```

Do **not** lower `eta_E` below 100, add wind angular momentum, or add stronger wind/heating physics until this weak local mass-loaded wind checkpoint is either made strict or fails for a clearly identified mesh-converged reason.

---

## 2. Current project status

The project now has four tiers:

```text
1. Standard no-wind high-Mdot slim disk:
   recovered to Mdot/Edd = 5.
   This is the clean high-Mdot advective backbone.

2. Finite stream-fed Mdot=5 no-wind compact-source branch:
   recovered and strongly advective at f_s = 0.80.
   This is a real steady high-Mdot upper-branch anchor.

3. Energy-only wind branch:
   numerically strict and mesh validated.
   But it removes energy without solving mass loss.

4. Local Mdot(R) mass-loaded wind BVP:
   now close at eta_E=100, but not yet strict.
   The remaining floor is interval_R near R ~ 300 rg.
```

Shen & Matzner-style `Mdot(R) ∝ R^s` profiles remain a good calibration/validation prior, not a final imposed solution. The current local-`Mdot(R)` BVP is the right direction because the final physical model should solve:

```math
\frac{d\dot M}{d\ln R} = \dot M'_w - \dot M'_s,
```

with

```math
\dot M'_w = \frac{2\pi R^2 Q_{\rm wind}}{E_w}.
```

For `m = ln Mdot` and `x = ln R`, the mass equation is:

```math
m' = \frac{\dot M'_w - \dot M'_s}{\dot M}.
```

At eta_E=100, that mass equation is already tight; the present failure is the radial momentum row.

---

## 3. Main diagnosis of the eta_E=100 floor

### 3.1 What the latest results say

The repaired local-Mdot ladder reached:

```text
N128 -> N136 -> N140 -> N152
refinement band = 100–320 rg
best direct N152 targeted residual = 2.515e-05
best mixed-prepolish + differential resume residual = 2.075e-05
```

The problem after the repaired remap is:

```text
interval_R ~ 2.1e-5 near R ~ 300 rg
interval_E < 7e-6
mass residual < 2e-6
outer_omega ~ 1.3e-5
```

So the mass-loaded wind closure is not the main issue at this weak-wind checkpoint. The radial row near the outer/source transition is.

### 3.2 What failed already

Do not repeat these as the main path:

```text
1. Generic fixed-N residual remeshing:
   creates source-annulus energy defects for the expanded local-Mdot state.

2. Ordinary N growth/prolongation:
   produces huge seed defects.

3. Integrated residual as final acceptance:
   can hide the physical differential radial defect.

4. Alternative outer closures:
   add boundary residuals and do not cure the R~300 interval_R floor.

5. Generic local least-squares relaxers:
   damage coupled mass/energy/source-budget rows.
```

Integrated or mixed residuals can still be useful as conditioning/pre-polish norms, but the accepted state must pass the original differential audit.

---

## 4. Recommended next numerical move

The best next move is a **radial-equation discretization and coupled-block correction sprint**, in this order:

```text
A. Radial residual representation audit.
B. Source-transition / buffer grid-alignment audit.
C. Row scaling and Jacobian conditioning audit for interval_R.
D. Block/Jacobian-aware local correction for the R~300 coupled block.
E. Only if A-D pass, promote a higher-order radial residual into the production solve.
```

Do not replace the final differential audit with an integrated defect criterion. Use integrated/higher-order forms to diagnose and correct, not to hide.

---

## 5. Equations for the residual audit

Use:

```math
x = \ln R,
\qquad
z = (U, \Theta, m) = (\ln u, \ln T, \ln \dot M).
```

The local ODE system can be written schematically as:

```math
U'      = F_R(x, U, \Theta, m; \lambda),
```

```math
\Theta' = F_E(x, U, \Theta, m; \lambda),
```

```math
m'      = F_M(x, U, \Theta, m; \lambda)
        = \frac{\dot M'_w - \dot M'_s}{\dot M}.
```

Here `F_R` is the radial momentum derivative implied by the slim/transonic equations, `F_E` is the energy derivative, and `F_M` is the local mass-continuity derivative.

### 5.1 Current differential-style interval residual

For interval `i`, with

```math
h_i = x_{i+1}-x_i,
\qquad
x_{i+1/2}=\frac{x_i+x_{i+1}}{2},
```

an endpoint-to-midpoint differential residual is effectively:

```math
R^R_{i,\rm diff}
=
\frac{(U_{i+1}-U_i)/h_i - F_R(x_{i+1/2}, z_{i+1/2})}{S^R_i}.
```

This is the residual currently setting the floor.

### 5.2 Trapezoid integral radial audit

Add a trapezoid radial audit:

```math
R^R_{i,\rm trap}
=
\frac{U_{i+1}-U_i
-\frac{h_i}{2}\left[F_R(x_i,z_i)+F_R(x_{i+1},z_{i+1})\right]}
{S^R_{i,\rm int}}.
```

### 5.3 Midpoint integral radial audit

Add a midpoint integral audit:

```math
R^R_{i,\rm mid}
=
\frac{U_{i+1}-U_i
-h_i F_R(x_{i+1/2}, z_{i+1/2})}
{S^R_{i,\rm int}}.
```

### 5.4 Simpson/Lobatto radial audit

If endpoint and midpoint values are available or can be reconstructed consistently:

```math
R^R_{i,\rm simp}
=
\frac{U_{i+1}-U_i
-\frac{h_i}{6}\left[F_R(x_i,z_i)+4F_R(x_{i+1/2},z_{i+1/2})+F_R(x_{i+1},z_{i+1})\right]}
{S^R_{i,\rm int}}.
```

The midpoint state should not be a naive average if the interval is steep. Prefer a Hermite-style reconstruction:

```math
z_{i+1/2}
\approx
\frac{z_i+z_{i+1}}{2}
-
\frac{h_i}{8}\left[F(x_{i+1},z_{i+1})-F(x_i,z_i)\right].
```

### 5.5 Local truncation / representation indicator

For each suspect interval, compute:

```math
\tau^R_i
=
\max\left(
|R^R_{i,\rm diff}-R^R_{i,\rm trap}|,
|R^R_{i,\rm diff}-R^R_{i,\rm simp}|,
|R^R_{i,\rm trap}-R^R_{i,\rm simp}|
\right).
```

Also compute a split-interval audit without moving accepted old nodes:

```text
Take interval [x_i, x_{i+1}], insert a virtual midpoint, reconstruct z_mid,
then evaluate the same radial residual on [x_i, x_mid] and [x_mid, x_{i+1}].
```

If splitting reduces the radial defect with the expected order, the floor is a representation/truncation issue. If it does not, the state itself is inconsistent in that block.

---

## 6. Source-transition / buffer grid-alignment audit

The radial residual peak sits near `R ~ 300 rg`, close to the source/outer transition. That suggests an interval may be straddling a rapidly changing source/wind/buffer region. A collocation interval should not straddle a piecewise-smooth transition and then be treated as a smooth ODE segment.

### 6.1 Identify transition radii

Dump these radii exactly:

```text
R_source_support_inner
R_source_peak = Rinj
R_source_support_outer
R_wind_active_inner/outer, if applicable
R_outer_buffer_inner, if used
Rout
R_peak_interval_R
```

For compact C2 source profiles, identify where the source and its first derivative are exactly zero.

### 6.2 Force nodes at transitions

Construct a node-preserving grid that includes:

```text
existing accepted nodes,
R_source_support_inner,
R_source_support_outer,
R_outer_buffer_inner = 300 rg if active/relevant,
R = 0.98 * R_transition,
R = 1.02 * R_transition,
R_peak_interval_R.
```

Do not let a long interval midpoint sit exactly on a source cutoff or buffer switch. If an interval straddles a source edge, split it.

### 6.3 Piecewise residual rule

For any interval crossing a known transition, either:

```text
1. split the interval at the transition; or
2. use separate left/right quadrature for source-active and source-free pieces.
```

For example, if `x_b` is a transition inside `[x_i, x_{i+1}]`:

```math
U_{i+1}-U_i
=
\int_{x_i}^{x_b} F_R^{\rm left}\,dx
+
\int_{x_b}^{x_{i+1}} F_R^{\rm right}\,dx.
```

This is not changing the physics. It is only respecting the piecewise-smooth structure of the source/buffer formulation.

---

## 7. Radial row scaling and Jacobian conditioning

The row scaling should help Newton find the correction, but final acceptance must still use the original physical differential residual.

### 7.1 Radial equation physical term audit

For the radial momentum equation, output the individual normalized terms at every interval, especially near `R~300 rg`:

```text
inertial term
pressure-gradient term
centrifugal term
gravity term
viscous/stress correction if present
source/wind/mass-loading terms if coupled into radial momentum
raw numerator residual
current residual scale S_R
Jacobian row norm
Jacobian column sensitivities wrt logu, logT, logMdot
condition estimate for local block
```

### 7.2 Suggested scaling for Newton merit

Use a scale such as:

```math
S^R_i
=
\max\left(
|A_i U'_i|,
|B_i \Theta'_i|,
|C_i m'_i|,
|G_i|,
|C_{\Omega,i}|,
S^R_{\rm floor}
\right),
```

where the terms represent the local contributions in the radial momentum equation. The exact decomposition should follow the code’s radial residual form.

Important rule:

```text
Use this scaling inside the Newton merit function only.
Continue to report the old unscaled/physical interval_R audit as the acceptance gate.
```

### 7.3 Expected result

If the problem is row-scaling/conditioning, the scaled Newton polish should lower the original interval_R below `1e-5` without changing physical diagnostics.

If the scaled polish lowers only the scaled residual but not the original physical interval_R, reject it.

---

## 8. Block/Jacobian-aware local correction

Generic local least-squares relaxers failed because they did not respect the coupled radial/energy/mass balance. The next local correction must use the real residual block and a real local Jacobian.

### 8.1 Define the local block

Let `k` be the interval where `interval_R` peaks. Use a block such as:

```text
nodes: k-3 ... k+4
interval rows: k-3 ... k+3
variables: logu, logT, logMdot on those nodes
optional: include lambda0 and logRson only in a second pass
fixed: outside block states
anchors: weak anchors on block-edge nodes to preserve global matching
```

Residual vector:

```math
r_B =
\left(
R^R_{k-q:k+q},
R^E_{k-q:k+q},
R^M_{k-q:k+q},
B_\Omega \; {\rm if\ relevant},
A_{\rm edge}(z_{\rm edge}-z_{\rm edge}^{old})
\right).
```

### 8.2 Solve the damped local Newton problem

Use:

```math
\left(J_B^T W J_B + \mu D^T D\right)\delta z_B
=
-J_B^T W r_B.
```

Then line search on the **global physical differential merit**:

```math
\Phi(z) =
\max\left(
|R^R|, |R^E|, |R^M|, |B_\Omega|, |B_E|
\right),
```

or use a Huber/softmax merit but always report the max norm separately.

### 8.3 Guardrails

Accept a block correction only if all are true:

```text
global full differential residual decreases;
interval_R peak near R~300 decreases;
interval_E remains < 1e-5;
mass_residual_max remains < 3e-6;
outer_omega does not increase above 2e-5;
Mdot_outer/Mdot_inner changes by < 1e-4;
Lrad changes by < 0.1%;
Rson changes by < 1e-3 rg;
no new interval_E/source-annulus defect appears near R~240-260 rg;
no new inner sonic radial defect appears near R~6-10 rg.
```

After local correction, run one global square/differential polish from the corrected state.

---

## 9. Higher-order radial residual: when to promote it

Promote a higher-order radial residual into the production solve only if the audit shows:

```text
1. The current differential interval_R peak is much larger than trapezoid/Simpson/split-interval estimates.
2. The high-order residual is stable under node-preserving nested refinement.
3. The high-order-residual-polished state also passes the original differential audit <= 1e-5.
```

The production option should be narrow at first:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_RADIAL_RESIDUAL_FORM=differential|trapezoid|simpson
```

Recommended first test:

```text
Use trapezoid radial residual for Newton merit,
but keep original differential radial audit as final acceptance.
```

Do not make `integrated_physical_energy` or a radial integral residual the final scientific gate until it has been cross-audited against the original differential residual and nested refinement.

---

## 10. Stricter integrated-defect audit without fooling ourselves

A stricter integrated audit is useful as a cross-check, not as a replacement.

For each equation and interval, compute both:

```math
D_i = \left|\frac{z_{i+1}-z_i}{h_i} - F_i\right|
```

and

```math
I_i = \left|z_{i+1}-z_i - \int_{x_i}^{x_{i+1}} F\,dx\right|.
```

Then require:

```text
max differential residual <= 1e-5;
max integral residual / h_i <= 1e-5 equivalent;
L1 integral defect small and decreases with nested refinement;
peak locations agree or are explained by truncation estimates;
no cancellation of positive/negative defects across the source transition.
```

Also output signed cumulative defects:

```math
C_R(x_j)=\sum_{i<j}\left[U_{i+1}-U_i-\int_i F_R dx\right],
```

```math
C_E(x_j)=\sum_{i<j}\left[\Theta_{i+1}-\Theta_i-\int_i F_E dx\right],
```

```math
C_M(x_j)=\sum_{i<j}\left[m_{i+1}-m_i-\int_i F_M dx\right].
```

If the max differential defect is localized but cumulative defects remain tiny and converge with nested refinement, it may be a representation issue. But still do not accept until the physical differential audit is below the chosen threshold or a clearly documented alternative criterion is validated against nested refinement.

---

## 11. Concrete next sprint plan

### Task 1 — Freeze the current best checkpoint

Freeze:

```text
outputs/checkpoints/m5_local_mdot_eta_polish_N152_integrated_physE_then_differential_resume/stage_00_etaE_100_N152.npz
```

Record:

```text
full differential residual = 2.075e-05
interval_R = 2.075e-05 near R~300.49 rg
interval_E = 6.865e-06
mass_residual_max = 1.746e-06
outer_omega = -1.335e-05
Mdot_outer/Mdot_inner = 0.232809
Lrad/LEdd = 0.527513
Rson = 5.29806 rg
```

### Task 2 — Radial residual form audit

Implement a seed-only audit mode:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_RADIAL_AUDIT_FORMS=differential,trapezoid,midpoint,simpson,split
```

Output for the top 20 radial intervals:

```text
R_mid
h = dlnR
R_diff
R_trap
R_midpoint
R_simpson
R_split_left
R_split_right
source_prime
wind_prime
outer_buffer_weight
radial term decomposition
```

Expected decision:

```text
If high-order/split residuals are also ~2e-5:
    true local state defect -> block/Jacobian correction.

If high-order/split residuals fall below 1e-5:
    representation/truncation issue -> high-order radial residual for Newton merit.
```

### Task 3 — Source-transition grid alignment

Add a node-preserving targeted grid generator that explicitly includes source and buffer transition radii:

```text
R_source_support_inner
R_source_support_outer
R_outer_buffer_inner = 300 rg, if relevant
R_peak_interval_R
```

Run seed-only and polish tests:

```text
N152 current grid
N152 + transition nodes, no old-node movement
N160 transition-node grid, old nodes preserved
N168 transition-node grid, old nodes preserved
```

Do not proceed if transition-node insertion reintroduces source-annulus energy defects.

### Task 4 — Row scaling / Jacobian audit

Output:

```text
radial row norm
energy row norm
mass row norm
radial column sensitivities wrt logu/logT/logMdot
smallest singular value of local block
condition number of local block
Newton step contribution by variable group
```

Then run a scaled Newton merit polish, but final-audit against the original differential residual.

### Task 5 — Block/Jacobian-aware local correction

Implement the local block solve over the `R~280–315 rg` region plus neighbor intervals. Include radial, energy, and mass rows together; do not solve radial-only.

Run:

```text
block half-width q = 2, 3, 4
anchor weight = 1e-2, 1e-1, 1
include outer_omega = false, true
include globals = false first
```

Accept only if original full differential residual drops below `1e-5` without moving physical diagnostics.

### Task 6 — Certification at eta_E=100

Once strict at N152, certify with nearby grids:

```text
N140, N152, N160 or N168 transition-node nested grids
same eta_E=100
same original differential audit
```

Acceptance:

```text
full differential residual <= 1e-5 preferred;
interval_R <= 1e-5;
interval_E <= 1e-5;
mass_residual_max <= 3e-6;
Mdot_outer/Mdot_inner stable within 0.1%;
Lrad stable within 0.2%;
Rson stable within 1e-3 to 1e-2 rg;
source-corrected s_eff_tilde stable;
no single unresolved cell dominates after transition-node alignment.
```

Only after this pass should Codex lower `eta_E` to 95, 90, 80, 70, 60.

---

## 12. Answers to the specific options

### Higher-order radial residual?

Yes, but first as an audit and Newton-merit/preconditioning tool. Do not immediately redefine acceptance around it.

### Block/Jacobian-aware local correction?

Yes. This is probably the best actual correction mechanism once the radial residual audit identifies whether the defect is a state defect or a representation defect. It must include radial, energy, and mass rows together.

### Radial row scaling?

Yes, for Newton conditioning. No, not as a way to lower the reported physical residual. The original physical differential residual remains the gate.

### Source-transition buffer/matching?

Yes. The R~300 location is too suspicious to ignore. Enforce transition nodes and split intervals that cross source/buffer edges. This is not adding physics; it is respecting piecewise smoothness.

### Stricter integrated-defect audit?

Yes, as a cross-audit. No, not as the final replacement for the physical differential gate unless it is proven equivalent under nested refinement.

---

## 13. Codex-ready implementation prompt

```text
Current state:
- Mdot_inner/Edd=5 local-Mdot mass-loaded wind BVP.
- f_s=0.80, Rout=335 rg, Rinj=240 rg, eta_E=100, weak zeta=0.03 Shen-style anchor.
- Bad remap/refinement pathology was fixed by targeted node-preserving nested insertion over 100-320 rg.
- Best repaired checkpoint:
    full differential residual = 2.075e-05
    interval_R = 2.075e-05 near R~300.49 rg
    interval_E = 6.865e-06
    mass_residual_max = 1.746e-06
    outer_omega = -1.335e-05
    Mdot_outer/Mdot_inner = 0.232809
    Lrad/LEdd = 0.527513
    Rson = 5.29806 rg
- This is not branch loss and not a gross mass-equation failure.
- The remaining problem is a localized radial differential/collocation floor near the source/outer transition.

Do not lower eta_E below 100 yet. Do not add wind angular momentum or new physical wind/heating terms yet.

Next tasks:

1. Freeze the best N152 repaired checkpoint as the eta_E=100 residual-floor anchor.

2. Add a radial residual representation audit:
   - differential, midpoint integral, trapezoid integral, Simpson/Lobatto, split-interval audit.
   - output top 20 interval_R rows with R_mid, h=dlnR, source_prime, wind_prime,
     buffer weights, and radial force-term decomposition.
   - decide whether the R~300 floor is truncation/representation or true local state defect.

3. Add source-transition/buffer grid alignment:
   - explicitly include compact source support edges, R_outer_buffer_inner=300 rg if relevant,
     and R_peak_interval_R as nodes.
   - node-preserving only; do not move accepted old nodes.
   - split intervals that cross source/buffer transitions.

4. Add radial row scaling and Jacobian diagnostics:
   - radial row norm, column sensitivities wrt logu/logT/logMdot,
     local block singular values, condition estimates.
   - use scaling only for Newton merit, not for final physical residual reporting.

5. Implement a block/Jacobian-aware local correction around R~300:
   - variables: logu/logT/logMdot on neighboring nodes.
   - residuals: radial + energy + mass rows together, not radial-only.
   - weak anchors on block edges.
   - damped least-squares/Newton step with line search on global physical differential residual.

6. Acceptance for eta_E=100 strict:
   - original full differential residual <= 1e-5;
   - interval_R <= 1e-5;
   - interval_E <= 1e-5;
   - mass_residual_max <= 3e-6;
   - Mdot_outer/Mdot_inner, Lrad, Rson, and s_eff_tilde stable;
   - no new source-annulus energy defect, no new inner sonic defect.

7. After eta_E=100 is strict and mesh-transfer stable, then lower eta_E gradually:
   - eta_E = 95, 90, 80, 70, 60;
   - same transition-node grid and original differential audit.
```

---

## 14. Bottom line

The best next move is **not** stronger wind, lower eta_E, or accepting an integrated residual. The best next move is:

```text
radial residual representation audit
+ source-transition node alignment
+ radial row/Jacobian scaling diagnostics
+ coupled block Newton correction near R~300
```

The target is simple:

```text
Make eta_E=100 strict under the honest physical differential audit.
```

Only then should the project continue toward lower launch energy, stronger mass loading, wind angular momentum, and eventually the reservoir-controlled QPE limit-cycle map.
