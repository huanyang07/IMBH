# Codex Handoff: Source-Band Two-Layer Replacement — Updated Assessment and Next Plan

Date: 2026-07-07  
Target repository: `huanyang07/IMBH`  
Latest reviewed commit on `main`: `f05e4d4` (`Add source-band replacement diagnostics`)  
Primary latest note: `Note/CODEX_SOURCE_BAND_REPLACEMENT_TWO_LAYER_RESULTS.md`

---

## 1. Current target

Physical/numerical target:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
Rinj = 240 rg
f_s = 0.80
compact-C2 stream source
torque_delta_l_fraction = +0.005
local-Mdot mass-loaded wind formulation
eta_E = 100 first
N = 164
```

Starting point for the latest two-layer tests:

```text
outputs/checkpoints/m5_source_band_replacement_chi050_eta100_N164/
    stage_00_etaE_100_N164.npz
```

This is the strict `chi_mass=0.50`, `chi_impl=0` source-band replacement checkpoint.

---

## 2. What the latest result says

The two-layer source-plus-buffer formulation is a useful diagnostic step, but it has **not** solved the source-band representation problem.

The latest results show:

```text
chi_mass = 0.50, no two-layer:
    active residual = 5.780e-6
    FV mass = 4.391e-9
    old source audit = 2.567e-2
    strict under active residual

chi_mass = 0.60, no two-layer:
    active residual = 7.881e-5
    FV mass = 7.881e-5
    not strict

chi_mass = 0.70, no two-layer:
    active residual = 5.205e-3
    FV mass = 5.205e-3
    line search alpha = 0.125
    not strict

chi_mass = 0.70, two-layer halo8, writable edges:
    active residual = 8.021e-3
    buffer old = 8.021e-3
    FV mass = 3.551e-3
    not strict

chi_mass = 0.60, two-layer halo8, frozen edges:
    active residual = 5.369e-5
    buffer old = 3.523e-5
    FV mass = 5.369e-5
    old source audit = 4.525e-2
    not strict

chi_mass = 0.70, two-layer halo8, frozen edges:
    active residual = 6.116e-5
    buffer old = 3.816e-5
    FV mass = 6.116e-5
    old source audit = 7.403e-2
    not strict

chi_mass = 0.50, chi_impl = 0.005, two-layer halo8, frozen edges:
    active residual = 5.067e-2
    FV mass = 8.827e-4
    implicit ODE = 1.921e-2
    Simpson = 5.390e-4
    interface = 5.067e-2
    not strict
```

The conclusion is now sharper:

```text
The problem is not simply source-band width, halo count, or scalar chi stepping.
The problem is the compatibility layer between:
    1. finite-volume mass increments,
    2. implicit radial/energy ODE rows,
    3. old midpoint outside rows.
```

Mass-only two-layer blending can move the floor into the `O(1e-5)`–`O(1e-4)` range, but cannot make `chi_mass >= 0.60` strict. The implicit radial/energy pilot became worse because the current slope-interface row over-constrains the replacement state.

---

## 3. Updated interpretation

### 3.1 This is not physical branch loss

The high-`Mdot` branch itself is already established in simpler settings:

```text
standard no-wind Mdot/Edd = 5:
    f_adv_global ~ 0.45
    f_adv_inner ~ 0.47
    H/R max ~ 0.32

stream-fed Mdot_inner/Edd = 5, f_s = 0.80:
    strongly advective / hot slim branch exists in the no-wind or energy-wind setting
```

The current failure is much more localized and much more numerical:

```text
source-band representation / interface compatibility
```

The bottleneck is confined to the compact source annulus and its buffer attachment.

### 3.2 The old midpoint source-band rows are not a reliable certification target

Previous identity audits showed:

```text
old midpoint source-band rows: strict
endpoint-compatible ODE-integral rows: O(10)
A g_old + c: O(10)
A g_direct + c: ~ machine precision
```

So the old midpoint interval slope is not the true local ODE-flow slope inside the compact source band. This means old midpoint rows should not remain the strict production target **inside** the source-plus-buffer band.

Outside the replacement band, the old midpoint production residual is still the correct production residual for now.

### 3.3 The current two-layer implementation likely still double-constrains the buffer

The latest two-layer mode keeps old midpoint rows active in buffer intervals while also ramping new finite-volume/implicit rows down through the same region. This is useful diagnostically, but as a production formulation it risks overdetermining the buffer.

The table makes this visible:

```text
two-layer halo8, chi_mass = 0.70, writable edges:
    active = 8.021e-3
    buffer old = 8.021e-3
```

The active residual is dominated by old buffer rows, not by the core finite-volume mass row. That is a sign that the buffer is not acting as a smooth transition; it is acting as a conflicting second discretization.

### 3.4 Freezing edge nodes is a useful diagnostic, not a final fix

Frozen edges let the solver accept full-alpha mass-only steps:

```text
chi_mass = 0.60:
    active = 5.369e-5

chi_mass = 0.70:
    active = 6.116e-5
```

This proves there is a near-active solution if the source-band boundaries are held fixed. But freezing edges also hides interface compatibility and makes the old source-row audit grow to `0.045`–`0.074`. It is not a certified source-band replacement.

### 3.5 The slope-interface row is currently the wrong attachment condition

The implicit pilot with a slope-interface row fails badly:

```text
chi_impl = 0.005:
    active = 5.067e-2
    interface = 5.067e-2
    implicit ODE = 1.921e-2
```

For a first-order ODE system, the natural interface condition is continuity of the state and physical fluxes, not equality of one-sided numerical slopes from two different discretizations. Slope matching forces the new source element to imitate the old midpoint manifold — exactly the manifold the identity audit showed is suspect.

So the slope-interface row should be removed, or demoted to a very weak diagnostic/regularizer with a small weight.

---

## 4. Best next numerical formulation

The next step should be a **true band-local replacement formulation**, not another scalar two-layer scan.

Use the old production residual outside the source-plus-buffer band.

Inside the source-plus-buffer band, **replace** old midpoint rows by a new source-band formulation:

```text
active residual =
    old production rows outside replacement band
  + source-band FV mass rows
  + source-band implicit radial/energy ODE rows
  + source-band Simpson/Hermite compatibility rows
  + C0 interface continuity rows
  + optional FV energy / angular-momentum penalty rows
```

Old midpoint rows inside the source/core/buffer band should be audits or guardrails, not active hard rows.

The main implementation change is:

```text
Do not append new rows to old rows in the same interval.
Replace the row set interval-by-interval.
```

---

## 5. Residual partitioning

Define source-band intervals as:

```text
core:
    true compact source support intervals

buffer:
    halo/transition intervals surrounding source support

outside:
    all remaining intervals
```

Recommended active rows by region:

```text
outside:
    old midpoint production rows

core:
    FV mass
    implicit radial/energy ODE
    Simpson/Hermite compatibility
    optional FV energy/J penalties

buffer:
    blended replacement rows, not old+new stacked rows
    C0 interface continuity
    weak physical regularization if needed

audits everywhere:
    old midpoint source-band rows
    source-interface FV energy
    source-element FV energy
    A g_old + c
    ODE-integral defect
```

For the buffer, use a **single blended row** rather than two active row families:

```text
r_buffer = (1 - w) r_old + w r_new
```

or, better, use separate continuation parameter `chi_buffer` with row replacement:

```text
w = 1 in core
w tapers from 1 to 0 across buffer
active row count stays fixed
old row is not separately active where r_buffer is active
```

This avoids the current situation where the active residual is dominated by old buffer rows that are fighting the new formulation.

---

## 6. Interface conditions

Use C0 state and flux compatibility. Do **not** impose C1/slope matching as a hard row.

Let:

```text
x = ln R
z = (U, Theta, m) = (ln u, ln T, ln Mdot)
```

At left and right source-band boundaries, enforce:

```math
z_{\rm source}(x_L) - z_{\rm outside}(x_L) = 0
```

```math
z_{\rm source}(x_R) - z_{\rm outside}(x_R) = 0
```

and, if duplicated states are used:

```math
\dot M_{\rm source}(x_L) - \dot M_{\rm outside}(x_L) = 0
```

```math
\dot M_{\rm source}(x_R) - \dot M_{\rm outside}(x_R) = 0
```

If the angular momentum flux is available, also audit or enforce:

```math
(\dot M l - G)_{\rm source} - (\dot M l - G)_{\rm outside} = 0 .
```

A slope-interface row should be removed or heavily relaxed:

```text
SOURCE_BAND_REPLACEMENT_SLOPE_INTERFACE_WEIGHT = 0 initially
```

If a slope guard is kept, it should be weak and diagnostic:

```text
slope_interface_weight <= 1e-3 to 1e-2 of main residual weight
```

Do not let it dominate the active residual.

---

## 7. Source-band implicit formulation

Inside the replacement band, use local slope unknowns:

```text
g_i     = dz/dx at source-band node i
g_mid_i = dz/dx at source-band midpoint i
```

with implicit ODE residual:

```math
A(x_i,z_i)g_i + c(x_i,z_i)=0
```

```math
A(x_{i+1/2},z_{i+1/2})g_{i+1/2}
+
c(x_{i+1/2},z_{i+1/2})=0 .
```

Use Simpson/Hermite compatibility:

```math
z_{i+1} - z_i -
\frac{h_i}{6}\left(g_i + 4g_{i+1/2}+g_{i+1}\right)=0 .
```

This avoids making `g = -A^{-1} c` the production map, while still forcing the source-band polynomial and local ODE to agree.

---

## 8. Finite-volume mass row

Use the conservative mass row in the replacement band:

```math
\dot M_{i+1}-\dot M_i
-
\int_{x_i}^{x_{i+1}}
(\dot M'_w-\dot M'_s)\,dx
=0 .
```

The sign convention is:

```math
\frac{d\dot M}{d\ln R} = \dot M'_w-\dot M'_s .
```

Mass replacement has already shown partial success, so it should be retained. But the current floor at `chi_mass >= 0.60` means the mass row must be integrated into the true replacement formulation, not layered on top of old buffer rows.

---

## 9. Finite-volume energy and angular momentum

Do not lower `eta_E` until finite-volume energy and angular-momentum compatibility are at least audited and preferably included as weak penalties.

Energy audit/penalty:

```math
R^E_i =
\frac{
\int_i^{i+1}
2\pi R^2
(Q_{\rm visc}+Q_{\rm stream}-Q_{\rm rad}-Q_{\rm adv}-Q_{\rm wind})
\,d\ln R
}{
\int_i^{i+1}
2\pi R^2
(|Q_{\rm visc}|+|Q_{\rm stream}|+|Q_{\rm rad}|+|Q_{\rm adv}|+|Q_{\rm wind}|)
\,d\ln R + \epsilon
}.
```

Angular-momentum audit/penalty:

```math
R^J_i =
\frac{
[\dot M l - G]_{i+1} - [\dot M l - G]_i
-
\int_i^{i+1}
(\dot M'_w l_w-\dot M'_s l_s+\tau_s)\,d\ln R
}{
|\dot M l - G|_{i+1}+|\dot M l - G|_i+\epsilon
}.
```

Start with:

```text
l_w = l
```

unless a wind lever arm is explicitly enabled.

---

## 10. Jacobian / preconditioning recommendations

The latest note correctly says the next step is not another scalar homotopy scan. It needs better local/global formulation and better derivative support.

Add analytic or semi-analytic local Jacobian support for:

```text
FV mass rows
implicit ODE rows
Simpson compatibility rows
C0 interface rows
FV energy/J penalty rows, if active
```

At minimum, implement local sparse finite-difference Jacobian with strong row/column scaling and a block structure.

Recommended row groups and scales:

```text
outside old rows:
    use existing production scaling

FV mass:
    scale by Mdot_inner or local max(|Mdot|)

implicit ODE:
    row-scale A g + c by row norms of A and c
    report raw and scaled values

Simpson compatibility:
    scale by max(|z_i|, |z_{i+1}|, |h g|, floor)

interface continuity:
    scale each variable separately

FV energy:
    scale by absolute heating/cooling integral

FV angular momentum:
    scale by local angular-momentum flux norm
```

Always report both scaled and raw residual maxima.

---

## 11. Solver strategy

### Stage 0 — freeze latest anchors

Freeze these as named regression references:

```text
m5_source_band_replacement_chi050_eta100_N164
m5_source_band_replacement_chi060_eta100_N164
m5_source_band_replacement_chi070_eta100_N164
m5_source_band_replacement_twolayer_halo8_chi070_eta100_N164
m5_source_band_replacement_twolayer_halo8_noedges_chi060_eta100_N164
m5_source_band_replacement_twolayer_halo8_noedges_chi070_eta100_N164
m5_source_band_replacement_twolayer_halo8_noedges_chi050_impl0005_eta100_N164
```

These are failure/diagnostic anchors, not certified physical branches.

### Stage 1 — implement row replacement, not row stacking

Add a mode such as:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_ROW_MODE=replace
```

with options:

```text
append     # current diagnostic behavior, old+new both active
replace    # one active row family per interval; preferred
blend      # convex homotopy row, fixed row count
audit      # compute new rows but do not activate
```

Use `replace` or `blend` for the production continuation.

### Stage 2 — remove slope-interface as a hard row

Set:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_SLOPE_INTERFACE_WEIGHT=0
```

for the next serious run.

Keep a diagnostic output:

```text
slope_interface_residual
```

but do not let it control line search.

### Stage 3 — finish mass replacement with true row replacement

Start from strict `chi_mass=0.50`.

Use:

```text
chi_mass = 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00
chi_impl = 0
row_mode = replace or blend
slope_interface_weight = 0
buffer old rows = audit only where replacement rows are active
```

Allow a full global polish under the replacement residual. Do not demand that old source-band midpoint rows remain strict. They are audits.

Acceptance for mass-only replacement:

```text
active residual <= 1e-5
outside old residual <= 1e-5
FV mass <= 3e-6 preferred, <=1e-5 allowed for exploratory
interface C0 continuity <=1e-6 to 1e-5
physical diagnostics stable
old source midpoint audit reported, not used as hard veto
```

### Stage 4 — if mass replacement stalls, release a wider buffer as variables

If `chi_mass` still stalls near `O(5e-5)`, then the problem is probably not the row definition alone; the old outside state needs to move.

Use an overlapping local/global solve:

```text
unknowns:
    source core states
    buffer states
    1-2 outside-halo layers on each side
    logRson/lambda0 only after local band succeeds

active rows:
    old outside rows outside overlap
    replacement/blended rows inside source+buffer
    C0 interface rows at overlap boundaries
```

Then run a global polish under the active replacement residual.

### Stage 5 — turn on implicit radial/energy rows only after mass replacement is strict

When `chi_mass = 1` is strict:

```text
chi_impl = 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10
slope_interface_weight = 0
row_mode = replace/blend
```

Do not repeat the current `chi_impl=0.005` test with the hard slope-interface row.

Acceptance for implicit rows:

```text
active residual <= 1e-5
outside old residual <= 1e-5
implicit ODE <= 3e-5 exploratory, <=1e-5 certified
Simpson compatibility <= 3e-5 exploratory, <=1e-5 certified
FV mass <= 3e-6 preferred
FV energy/J audits not worsening
physical diagnostics stable
```

### Stage 6 — certify eta_E=100 before eta_E=90

Do not lower `eta_E` until the replacement formulation is strict at `eta_E=100`.

Certification target:

```text
eta_E = 100
chi_mass = 1
chi_impl > 0, ideally at least 0.05 or 0.10
active residual <= 1e-5
outside old residual <= 1e-5
no O(1e-2) hidden source-band residual
FV mass/energy/J audits controlled
identity audit no longer shows O(10) contradiction
```

Then continue:

```text
eta_E = 95, 90
```

Do not continue to:

```text
eta_E = 80, 70, 60
```

until `eta_E = 90` is certified under the same replacement formulation.

---

## 12. Diagnostics to add to every run

For each source/core/buffer interval:

```text
R_left_rg
R_mid_rg
R_right_rg
interval class: core / buffer / outside
row mode: old / new / blend / audit
new_weight
old_weight
active row group
old midpoint radial/energy/mass
FV mass
implicit ODE radial/energy
Simpson compatibility
C0 interface residual
slope-interface audit
FV energy numerator and denominator
FV angular momentum numerator and denominator
A g_old + c
A g_direct + c
condition(A), raw and scaled
Mdot_stream_prime/Mdot
Mdot_wind_prime/Mdot
Qstream/Qvisc
Qwind/Qvisc
```

Summary table should include:

```text
active total residual
outside old residual
core new residual
buffer blended residual
old source midpoint audit
FV mass
implicit ODE
Simpson
C0 interface
FV energy
FV angular momentum
identity ODE-integral max
A g_old + c max
Mdot_outer/Mdot_inner
Lrad/LEdd
f_adv_global
Rson
nfev
alpha
accepted flag
```

---

## 13. Decision tree

### If mass replacement reaches `chi_mass = 1` strict

Proceed to implicit radial/energy replacement with slope-interface disabled.

### If mass replacement remains stuck around `5e-5`

Do not scan chi forever. Implement overlapping global polish with wider variable release and true row replacement.

### If implicit rows fail only with slope-interface active

Remove slope-interface permanently as production row. Keep it as an audit.

### If implicit rows fail even without slope-interface and with active row replacement

Then the source-band ODE formulation and old outside solution are not smoothly compatible. Move to a full micro-domain/mortar formulation with duplicated interface states and a source-block polynomial defined from implicit slopes from the start.

### If full micro-domain/mortar formulation still stalls at `O(1e-2)`

Then reject the current compact source + local-Mdot closure at this wind loading and revisit the physical source/wind coupling. This would still not mean the high-`Mdot` branch is physically absent.

---

## 14. Codex-ready short prompt

```text
Latest source-band two-layer results show that eta_E=100 is still not certified
under source-band replacement beyond chi_mass=0.50.

Current facts:
- chi_mass=0.50, chi_impl=0 is strict:
    active = 5.780e-6, FV mass = 4.391e-9.
- chi_mass=0.60 without two-layer gives active = 7.881e-5.
- chi_mass=0.70 without two-layer gives active = 5.205e-3.
- two-layer halo8 with writable edges at chi_mass=0.70 improves only to
    active = 8.021e-3, dominated by active old buffer rows.
- two-layer halo8 with frozen edges reaches:
    chi_mass=0.60: active = 5.369e-5
    chi_mass=0.70: active = 6.116e-5
  but old source-row audit grows to 0.045--0.074, so this is not certified.
- chi_impl=0.005 with current slope-interface row fails badly:
    active = 5.067e-2,
    interface = 5.067e-2,
    implicit ODE = 1.921e-2.
- Therefore the simple slope-interface row is over-constraining the source-band
  replacement state.

Interpretation:
- Do not lower eta_E.
- Do not keep scanning scalar chi with the current row stacking.
- The bottleneck is compatibility between FV mass increments, implicit
  radial/energy ODE rows, and old midpoint outside rows.
- Old midpoint source-band rows are not reliable production rows inside the
  source band; keep them as audits.
- The next step is true row replacement / blended row homotopy, not old+new
  row stacking.

Implement:
1. Add source-band replacement row modes:
       append, replace, blend, audit
   Use replace/blend for production.

2. In core/source intervals:
       active rows = FV mass + implicit ODE + Simpson compatibility.
   Do not also keep old midpoint rows active there.

3. In buffer intervals:
       use one blended row:
           r_buffer = (1-w) r_old + w r_new
       or true replacement with tapered w.
   Do not stack old and new rows in the same interval.

4. Disable hard slope-interface production row:
       SOURCE_BAND_REPLACEMENT_SLOPE_INTERFACE_WEIGHT = 0
   Keep it only as an audit.

5. Use C0 state/interface continuity instead:
       z_source - z_outside = 0
       Mdot_source - Mdot_outside = 0
       optionally angular-momentum flux continuity.

6. Finish mass replacement first:
       start from chi_mass=0.50
       chi_mass = 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00
       chi_impl = 0
       row_mode = replace/blend
       old source rows audit only.

7. If mass replacement stalls around 5e-5, release a wider overlapping
   source+buffer+outside halo variable block and globally polish under the
   active replacement residual.

8. After chi_mass=1 is strict, turn on implicit radial/energy rows:
       chi_impl = 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10
       slope-interface disabled.

9. Add analytic/local Jacobian support and row/column scaling for:
       FV mass,
       implicit ODE,
       Simpson compatibility,
       C0 interface,
       FV energy/J penalty rows.

10. Certification before eta_E=90:
       active residual <= 1e-5
       outside old residual <= 1e-5
       FV mass <= 3e-6 preferred
       implicit ODE <= 3e-5 exploratory, <=1e-5 certified
       Simpson <= 3e-5 exploratory, <=1e-5 certified
       FV energy/J audits controlled
       no O(1e-2) hidden source-band residual
       identity audit no longer shows O(10) contradiction
       Mdot_outer/Mdot_inner, Lrad/LEdd, Rson stable.

Only after eta_E=100 passes this replacement formulation should eta_E=95/90 be
attempted. Do not lower to eta_E=80/70/60 until eta_E=90 is source-band
certified under the same formulation.
```

---

## 15. Bottom line

The two-layer sprint was useful because it identified the next problem cleanly:

```text
The source-band replacement is currently fighting the old buffer/midpoint
discretization rather than replacing it.
```

The next implementation should therefore:

```text
1. stop stacking old and new rows in the same source/buffer intervals;
2. disable the hard slope-interface row;
3. use C0/mortar-style interface continuity;
4. finish mass replacement with true row replacement;
5. then reintroduce implicit radial/energy rows with analytic/local Jacobian
   support and a global active-residual polish.
```

This should be done at `eta_E=100` before any attempt to lower `eta_E`.
