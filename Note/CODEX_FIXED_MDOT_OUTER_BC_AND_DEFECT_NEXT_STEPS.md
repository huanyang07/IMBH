# Codex Next-Step Brief: Resolve the Fixed-\(\dot M\) Residual Floor by Replacing the Finite-Radius Keplerian Outer Boundary

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
commit: c25af208bb2a385c8893fe7696cbb4fc26a2dbec
```

Primary new outputs reviewed:

```text
outputs/tables/transonic_fixed_mdot_root_audit.md
outputs/tables/transonic_fixed_mdot_block_audit.md
outputs/tables/transonic_outer_closure_audit.md
scripts/run_transonic_fixed_mdot_root_audit.py
scripts/run_transonic_fixed_mdot_block_audit.py
scripts/run_transonic_outer_closure_audit.py
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
src/imri_qpe/layer3_minidisk_1d/transonic_continuation.py
```

## Executive conclusion

The latest audits are highly informative. They show that the present
\(N=64\), \(\dot M/\dot M_{\rm Edd}\simeq0.903\) residual floor is **not mainly
a sonic-condition failure** and is **not primarily a finite-difference
Jacobian error**.

The dominant problem is the finite-radius outer boundary condition

```math
\Omega(R_{\rm out})=\Omega_K(R_{\rm out}),
```

which is imposed exactly in `_outer_boundary_residual()`.

At a finite radius, a disk with nonzero radial pressure support is not exactly
Keplerian. The expected fractional correction is of order

```math
\left|\ln\frac{\Omega}{\Omega_K}\right|
\sim
\left(\frac{H}{R}\right)^2.
```

The current outer disk has

```text
H/R ~= 0.0148
(H/R)^2 ~= 2.2e-4
```

while the irreducible outer-\(\Omega\) residual is

```text
B_Omega ~= 1.6e-4 to 2.8e-4.
```

Those scales agree. The residual floor is therefore very likely the numerical
signature of an **asymptotically inconsistent exact-Keplerian boundary imposed
at finite radius**, not evidence that the transonic branch ends near one
Eddington.

The immediate next step is:

```text
Replace the exact Omega=Omega_K endpoint condition with a pressure-supported,
matched-asymptotic outer closure; then reformulate interval collocation as an
integrated defect to improve Jacobian conditioning.
```

Do not add wind, stream feeding, tides, or time dependence yet.

---

# 1. What the new audits establish

## 1.1 No polished fixed-\(\dot M\) root exists under the current boundary closure

The algebraic-pivot root audit finds no root at:

```text
Mdot/Edd ~= 0.903
Mdot/Edd ~= 0.965
Mdot/Edd ~= 0.996
```

for:

```text
N = 32, 48, 64
pivot = C1, C2.
```

At the native \(N=64\) points, dense SVD Newton does not reduce the residual:

```text
0.903: 1.584e-4 -> 1.584e-4
0.965: 1.687e-4 -> 1.687e-4
0.996: 1.735e-4 -> 1.735e-4
```

The dominant block is consistently:

```text
outer_Omega.
```

The sonic residuals are already much smaller, generally of order:

```text
1e-5 to 1e-4.
```

## 1.2 The interval equations themselves are solvable

The block audit shows:

```text
interval_only selected residual:
    3.5e-10
```

so the midpoint differential equations can be solved essentially exactly.

When the outer rows are removed, the interval plus sonic equations reach:

```text
~1e-5
```

but the omitted endpoint conditions become:

```text
outer_Omega ~ 2.5e-4 to 2.8e-4
outer_energy ~ 3.8e-4
```

When only the outer-\(\Omega\) condition is removed, the remaining selected
system reaches:

```text
~1e-5
```

while the omitted outer-\(\Omega\) residual remains:

```text
~2.5e-4.
```

This identifies the outer rotation condition as the main incompatibility.

## 1.3 Removing sonic conditions does not remove the floor

The `no_sonic` solve remains at approximately:

```text
1.58e-4
```

and is still dominated by `outer_Omega`.

Therefore the current floor is not primarily caused by choosing \(C_1\),
\(C_2\), or \(K\) at the sonic point.

## 1.4 Downweighting outer rows only trades one residual for another

Outer-row weights of:

```text
1e-1
1e-2
1e-3
```

do not produce a full root. They allow the optimizer to satisfy the interior
more closely while moving the mismatch between:

```text
outer_Omega
outer_energy.
```

This is expected when the boundary target itself is inconsistent.

## 1.5 Alternative outer closures tested so far do not solve the actual problem

The audit tried:

```text
thin_value
value_full_energy
value_zero_adv
shear_thin_energy
shear_full_energy
```

None gives a polished full solution.

The tested shear closure constrains:

```math
\frac{d\ln\Omega}{d\ln R}
=
\frac{d\ln\Omega_K}{d\ln R},
```

but does not supply the physically required **pressure-supported amplitude**
of \(\Omega\). It is therefore not the same as a pressure-corrected outer
match.

---

# 2. Physical reason exact Keplerian rotation is inconsistent at finite radius

The steady radial momentum equation is:

```math
u\frac{du}{dR}
=
R(\Omega^2-\Omega_K^2)
-
\frac{1}{\Sigma}\frac{d\Pi}{dR},
```

where \(u=-v_R>0\) and \(\Pi\) is the vertically integrated pressure.

Using logarithmic derivatives:

```math
u^2\frac{d\ln u}{d\ln R}
=
R^2(\Omega^2-\Omega_K^2)
-
\frac{\Pi}{\Sigma}\frac{d\ln\Pi}{d\ln R}.
```

Therefore:

```math
\frac{\Omega^2-\Omega_K^2}{\Omega_K^2}
=
\frac{u^2}{R^2\Omega_K^2}
\frac{d\ln u}{d\ln R}
+
\frac{\Pi}{\Sigma R^2\Omega_K^2}
\frac{d\ln\Pi}{d\ln R}.
```

For a thin outer disk:

```math
\frac{\Pi}{\Sigma}
\sim
H^2\Omega_K^2,
```

so:

```math
\frac{\Omega^2-\Omega_K^2}{\Omega_K^2}
\sim
\left(\frac{H}{R}\right)^2
\frac{d\ln\Pi}{d\ln R},
```

up to a small radial-inertia term.

Thus:

```math
\ln\frac{\Omega}{\Omega_K}
\approx
\frac{1}{2}
\left(\frac{H}{R}\right)^2
\frac{d\ln\Pi}{d\ln R}.
```

At the current outer boundary:

```text
H/R ~= 0.0148
```

so a natural pressure-supported correction is:

```text
|delta ln Omega| ~ 1e-4 to several 1e-4,
```

precisely the observed residual floor.

The correct asymptotic condition is not:

```math
\Omega=\Omega_K
```

at finite \(R\), but:

```math
\Omega=\Omega_K+\text{pressure-support correction}.
```

---

# 3. Implement a pressure-supported outer closure

Implement two levels. Start with Level A because it directly tests the
diagnosis. Move to Level B for the production solver.

---

## Level A: pressure-corrected \(\Omega\) plus outer thermal equilibrium

Use a slope estimate from the repaired reduced thin/slim solution at the same:

```text
Mdot
alpha
stress normalization
R_out
lambda0
```

Let:

```math
g_u^{\rm match}
=
\left.\frac{d\ln u}{d\ln R}\right|_{\rm reduced},
```

```math
g_T^{\rm match}
=
\left.\frac{d\ln T}{d\ln R}\right|_{\rm reduced}.
```

From these and the full transonic thermodynamic closure, calculate:

```math
g_\Pi^{\rm match}
=
\left.\frac{d\ln\Pi}{d\ln R}\right|_{\rm match}.
```

Define the pressure-supported rotation target:

```math
\delta_{\Omega,\rm ps}
=
\frac{1}{2}
\left[
\frac{u^2}{R^2\Omega_K^2}g_u^{\rm match}
+
\frac{\Pi}{\Sigma R^2\Omega_K^2}g_\Pi^{\rm match}
\right].
```

Use the outer residual:

```math
B_{\Omega,\rm ps}
=
\ln\left(\frac{\Omega}{\Omega_K}\right)
-
\delta_{\Omega,\rm ps}.
```

For the second outer equation, first use the thin radiative balance already in
the code:

```math
B_E
=
\frac{Q_{\rm visc,thin}-Q_{\rm rad}}
{|Q_{\rm visc,thin}|+|Q_{\rm rad}|+\epsilon}.
```

New closure name:

```text
pressure_supported_thin_energy
```

### Immediate test

At the native:

```text
Mdot/Edd = 0.90277664
N = 64
R_out = 3000 r_g
```

run both:

```text
pivot C1
pivot C2
```

Target:

```text
full square residual < 1e-6
unused compatibility < 1e-6
```

If the residual floor collapses by two orders of magnitude, the diagnosis is
confirmed.

---

## Level B: match directly to a local full outer asymptotic state

This is the preferred production boundary condition.

### Step 1: obtain outer slopes

Use the repaired reduced solver or an independently continued outer thin-disk
solution to estimate:

```text
g_u_match
g_T_match
```

over a small outer annulus.

Use a smooth polynomial/PCHIP fit in \(\ln R\), not a two-point noisy slope.

### Step 2: solve a local two-variable match

At fixed:

```text
R_out
Mdot
lambda0
g_u_match
g_T_match
```

solve for:

```text
log u_match
log T_match
```

from the full local equations:

```math
F_R(R_{\rm out}, y_{\rm match}, g_{\rm match}, \lambda_0)=0,
```

```math
F_E(R_{\rm out}, y_{\rm match}, g_{\rm match}, \lambda_0)=0.
```

Start from the reduced solution's outer state and choose the nearest physical
root.

This local solve includes:

```text
radial pressure support
radial inertia
full stress-shear heating
full entropy advection
```

at the outer match.

### Step 3: use value matching in the global BVP

Define:

```math
B_u
=
\ln u_{\rm out}
-
\ln u_{\rm match},
```

```math
B_T
=
\ln T_{\rm out}
-
\ln T_{\rm match}.
```

New closure name:

```text
matched_outer_state
```

The matched state should be recomputed smoothly as \(\lambda_0\) and
\(\dot M\) vary. Cache the local solution and use continuation so this does not
become expensive.

### Why value matching is preferable

It supplies two independent asymptotic conditions without forcing exact
Keplerian rotation at finite radius. It also avoids using endpoint derivatives
as boundary conditions that are redundant with the ODEs.

---

# 4. Do not use endpoint radial momentum itself as an independent boundary condition

A tempting closure is:

```text
outer radial momentum residual = 0
outer energy residual = 0
```

This should not be the production choice.

Those are the same differential equations already enforced throughout the
domain. In the continuum limit, imposing them again at the endpoint does not
supply two independent asymptotic conditions and can leave the BVP
underdetermined.

Use either:

```text
pressure-corrected Omega + one thermal condition
```

or, preferably:

```text
matched outer values [u_match, T_match].
```

---

# 5. Reformulate midpoint collocation as an integrated defect

The fixed-\(\dot M\) Jacobian becomes dramatically more ill-conditioned with
increasing \(N\):

```text
N=32: equilibrated condition ~1e9
N=48: equilibrated condition ~1e12
N=64: equilibrated condition ~1e14
```

Part of this is caused by writing interval residuals in differential form:

```math
F_i
=
A_i\frac{y_{i+1}-y_i}{\Delta x_i}
+
c_i.
```

The Jacobian then contains entries of order \(1/\Delta x_i\), so conditioning
worsens as the grid is refined.

## Required change

Use an integrated collocation defect:

```math
\widetilde F_i
=
A_i(y_{i+1}-y_i)
+
\Delta x_i c_i.
```

Equivalently:

```python
integrated_interval_residual = dx * current_scaled_interval_residual
```

provided the same state-dependent scales are used consistently.

This does not change the discrete root because \(\Delta x_i>0\), but it removes
the explicit \(1/\Delta x_i\) amplification.

### Better formulation

In `transonic_local.py`, expose:

```python
A_bar, c_bar = scaled_differential_matrix(...)
```

Then in the collocation code use:

```python
dy = y_right - y_left
residual = A_bar @ dy + dx * c_bar
```

rather than reconstructing:

```python
g = dy / dx
residual = A_bar @ g + c_bar
```

### Tests

Run:

```text
N = 32, 48, 64, 96, 128
```

and report raw/equilibrated condition numbers before and after defect scaling.

Expected result:

```text
condition growth with N becomes much weaker.
```

Update residual tolerances because the integrated defect has a different
magnitude. Continue to report a reconstructed differential residual for
physical auditing.

---

# 6. Row and column equilibration remains necessary

After converting to integrated defects:

1. apply 3--5 Ruiz row/column equilibration iterations;
2. compute dense SVD for N64/N96 diagnostics;
3. solve the equilibrated Newton system;
4. transform the step back.

Report:

```text
raw condition
equilibrated condition
smallest singular value
near-null vector block fractions
```

A valid improvement should reduce both:

```text
condition number
line-search cuts.
```

Do not rely on a fixed SVD `rcond` chosen independently for each case. Select
regularization from predicted-versus-actual residual reduction.

---

# 7. The N32/N48 and R_out sweeps are not yet convergence tests

## 7.1 N32 and N48

The root audit starts from remapped N64 checkpoints. Their initial interval
energy residuals are:

```text
N32 ~ 0.48
N48 ~ 0.24
```

These are not nearby roots. Failure of ten dense-Newton iterations from these
states does not prove that N32/N48 roots do not exist.

Run each resolution independently:

```text
start at low Mdot
continue in Mdot on that same grid
use the corrected outer closure
polish each anchor.
```

## 7.2 Outer-radius sweep

The one-shot remaps to:

```text
R_out = 300, 1000, 10000 r_g
```

produce very large initial residuals and are not valid outer-boundary
convergence tests.

After the new outer closure is implemented:

```text
continue gradually in ln R_out
```

using steps of approximately:

```text
Delta ln R_out = 0.05--0.15
```

and polish each point.

Do not infer physical R_out dependence from one-step remapping.

---

# 8. Use nested grids for true mesh convergence

Migrate production grids to a nested hierarchy, for example:

```text
N = 33, 65, 129
```

or:

```text
N = 65, 129, 257.
```

This allows coarse nodes to be retained on the fine grid.

Use derivative-aware prolongation:

```text
cubic Hermite using collocation slopes
```

or at minimum:

```text
PCHIP in logR for logu and logT.
```

Then:

1. prolong the coarse root;
2. solve the fine-grid fixed-\(\dot M\) BVP;
3. compare independently polished roots.

Do not use raw interpolation residuals as convergence metrics.

---

# 9. Re-evaluate pseudo-arclength only after a true fixed-\(\dot M\) root is recovered

The continuation stall near \(1.05\) was built from checkpoints with a
fixed-\(\dot M\) residual floor. Pseudo-arclength cannot repair a boundary
condition that admits no exact root.

After the outer closure and integrated defect are implemented:

1. recover a polished root at \(0.90\);
2. recover roots at \(0.965\) and \(0.996\);
3. require residual \(<10^{-6}\);
4. recompute Jacobian tangents from those roots;
5. restart pseudo-arclength.

Only then perform the fold audit:

```text
sigma_min(J_z)
u_min^T F_mu
t_mu
condition of bordered Jacobian
```

The previous apparent stall should not be interpreted as a fold unless it
persists after these corrections.

---

# 10. Suggested code changes

## `transonic_collocation.py`

Add outer closure selection to `TransonicSlimParams`:

```python
outer_closure: str = "pressure_supported_thin_energy"
```

Add:

```python
def reduced_outer_slopes(params, lambda0):
    ...

def pressure_supported_omega_target(logR, y, g_match, lambda0, params):
    ...

def matched_outer_state(lambda0, params, initial_guess=None):
    ...

def outer_boundary_residual_matched(logR, y, lambda0, params):
    ...
```

Modify `_outer_residual_block()` to dispatch by `params.outer_closure`.

Add an integrated interval residual:

```python
def _integrated_interval_residual_from_unpacked(...):
    dx = logR[idx + 1] - logR[idx]
    y_left = ...
    y_right = ...
    y_mid = 0.5 * (y_left + y_right)
    x_mid = 0.5 * (...)
    A_bar, c_bar, *_ = scaled_differential_matrix(...)
    return A_bar @ (y_right - y_left) + dx * c_bar
```

Keep the old differential residual as a diagnostic:

```python
def differential_interval_audit(...):
    ...
```

## `transonic_continuation.py`

Do not change high-rate continuation yet except to support the new outer
closure in checkpoints.

After fixed-\(\dot M\) roots are recovered:

```text
discard or mark old tolerance-limited anchors as legacy
rebuild the branch from polished roots
```

## New scripts

Create:

```text
scripts/run_transonic_pressure_supported_outer_audit.py
scripts/run_transonic_integrated_defect_audit.py
scripts/run_transonic_nested_grid_branch.py
```

Outputs:

```text
outputs/tables/transonic_pressure_supported_outer_audit.md
outputs/tables/transonic_integrated_defect_conditioning.md
outputs/tables/transonic_nested_grid_convergence.md
outputs/figures/transonic_outer_pressure_match.png
outputs/figures/transonic_conditioning_vs_N.png
```

---

# 11. Immediate experiment matrix

## Experiment A: diagnose the expected pressure correction

At the current N64 \(0.9028\) checkpoint calculate:

```text
H/R
g_Pi = d ln Pi / d ln R
g_u  = d ln u / d ln R
predicted delta ln Omega
actual outer_Omega residual
```

Verify:

```text
same sign
same order of magnitude
```

This is the fastest confirmation of the central diagnosis.

## Experiment B: pressure-supported outer closure

Run:

```text
N=64
R_out=3000 r_g
Mdot/Edd=0.9028
pivot=C1,C2
```

with:

```text
pressure_supported_thin_energy
```

Then run:

```text
matched_outer_state
```

Target:

```text
square residual < 1e-6
all D,C1,C2,K < 1e-6
```

## Experiment C: integrated-defect conditioning

At the same state compare:

```text
differential residual formulation
integrated defect formulation
```

for:

```text
N=32,48,64,96
```

Report:

```text
raw/equilibrated condition
Newton iterations
line-search cuts
final residual.
```

## Experiment D: independent grid branches

With the corrected closure, independently continue:

```text
N=33
N=65
N=129
```

from low \(\dot M\) to \(1.1\).

## Experiment E: restart pseudo-arclength

Only after Experiments B--D pass, restart continuation above one Eddington.

---

# 12. Go/no-go criteria

## Proceed with the transonic branch if

```text
- the pressure-supported or matched outer closure gives an actual fixed-Mdot root;
- integrated-defect conditioning improves with N;
- C1 and C2 pivots converge to the same physical profile;
- independently solved nested grids agree;
- fixed-Mdot anchors polish below 1e-6.
```

## Revisit the stress/outer asymptotic model if

```text
- no root exists even after pressure-supported matching;
- the mismatch remains localized at the outer endpoint;
- the result is strongly sensitive to R_out after gradual continuation.
```

## Revisit the differential equations if

```text
- interval-only equations cease to converge after integrated-defect scaling;
- the near-null singular vector is localized in interior variables rather than
  boundary/eigenparameter blocks;
- C1 and C2 roots remain genuinely different after exact polishing.
```

## Do not claim a physical fold or branch termination unless

```text
- exact fixed-Mdot roots exist on both sides of the suspected region;
- the corrected BVP is mesh-converged;
- the fold transversality conditions are satisfied;
- the bordered Jacobian remains regular.
```

---

# 13. Compact Codex prompt

```text
The latest fixed-Mdot audits show that the residual floor near Mdot/Edd~0.9--1
is dominated by the exact outer condition Omega=Omega_K, not by the sonic
equations. At R_out=3000 rg, H/R~0.0148 so (H/R)^2~2.2e-4, matching the
irreducible outer-Omega residual ~1.6e-4--2.8e-4. This is the expected finite-
radius radial-pressure correction.

Implement next:

1. Add a pressure-supported outer rotation condition:
       delta_Omega =
         0.5 * [
           u^2/(R^2 OmegaK^2) * g_u_match
           + Pi/(Sigma R^2 OmegaK^2) * g_Pi_match
         ]
       B_Omega = log(Omega/OmegaK) - delta_Omega
   with slopes from the repaired reduced outer solution.

2. Preferably add a matched_outer_state closure:
   obtain g_u,g_T from the reduced outer solution, solve local full
   F_R=F_E=0 for logu_match,logT_match, and impose
       logu_out=logu_match
       logT_out=logT_match.

3. Reformulate midpoint interval residuals as integrated defects:
       R_i = Abar_i @ (y_{i+1}-y_i) + dx_i * cbar_i
   instead of Abar_i @ ((y_{i+1}-y_i)/dx_i) + cbar_i.
   This should remove 1/dx Jacobian amplification and improve N-scaling.

4. Keep the old differential residual only as a physical audit.

5. Test N64, Mdot/Edd=0.9028, R_out=3000, pivots C1/C2.
   Require square residual and unused compatibility <1e-6.

6. Do not interpret one-shot N32/N48 or R_out remaps as convergence tests.
   Continue each grid/R_out gradually and independently.

7. Use nested grids N=33,65,129 with Hermite/PCHIP prolongation.

8. Rebuild fixed-Mdot roots at 0.90,0.965,0.996 before restarting
   pseudo-arclength. Old tolerance-limited anchors should be marked legacy.

9. Do not add wind, stream, tide, or time dependence yet.
```

---

# 14. Bottom line

The new audits have narrowed the difficulty considerably:

```text
The transonic differential equations are solvable, and the sonic block is not
the main source of the residual floor. The exact finite-radius Keplerian outer
condition is inconsistent at precisely the expected O[(H/R)^2] level, while
the differential-form collocation makes the global Jacobian increasingly
ill-conditioned as N grows.
```

The next sprint should fix those two issues before further high-rate
continuation.
