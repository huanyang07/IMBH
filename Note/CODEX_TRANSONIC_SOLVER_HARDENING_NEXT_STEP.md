# Codex Next-Step Brief: Harden the Transonic Slim-Disk Solver Before Interpreting the High-\(\dot M\) Failure

Repository:

```text
https://github.com/huanyang07/IMBH
```

This brief is based on the current repository state, especially:

```text
GPT_REPO_HANDOFF.md
README.md
outputs/tables/transonic_solver_audit.md
outputs/figures/transonic_branch_summary.png
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
src/imri_qpe/layer3_minidisk_1d/transonic_continuation.py
src/imri_qpe/layer3_minidisk_1d/transonic_potential.py
src/imri_qpe/layer3_minidisk_1d/transonic_thermo.py
scripts/plot_transonic_slim_branch.py
tests/test_transonic_*.py
```

The current run is useful, but the present table should still be treated as a **solver-hardening result**, not as a physical transonic branch.

---

# 1. Current Result and Main Diagnosis

The current audit reports:

| \(\dot M/\dot M_{\rm Edd}\) | reported converged | max residual | \(R_{\rm son}/r_g\) | \(\lambda_0=l_0/(r_gc)\) | max \(H/R\) | sonic crossings |
|---:|:---:|---:|---:|---:|---:|---:|
| 0.001 | yes | \(7.8\times10^{-5}\) | 3.96 | 1.51 | 0.0085 | 1 |
| 0.003 | yes | \(1.6\times10^{-4}\) | 4.32 | 1.60 | 0.0195 | 1 |
| 0.01 | yes | \(3.9\times10^{-4}\) | 4.90 | 1.58 | 0.0828 | 1 |
| 0.02 | yes | \(6.7\times10^{-4}\) | 5.44 | 1.64 | 0.220 | 1 |
| 0.03 | yes | \(8.8\times10^{-4}\) | 6.22 | 1.65 | 0.343 | 0 |
| 0.05 | no | \(1.2\times10^{-3}\) | 7.73 | 1.79 | 0.559 | 0 |
| 0.1 | no | \(1.4\times10^{-3}\) | 13.8 | 2.46 | 0.472 | 0 |
| 1 | no | \(1\) | 60 | 5.56 | 8.68 | 0 |

The right interpretation is:

```text
The code has built a promising transonic collocation prototype, but it has not
yet established a physically validated sonic eigen-solution.
```

There are four immediate warning signs.

## 1.1 “Converged” does not currently mean physically converged

`solve_transonic_outer_branch()` sets

```python
converged = max_residual <= residual_tol
```

without requiring:

```text
optimizer_success
sonic regularity
one sonic crossing
inactive parameter bounds
physical low-Mdot eigenvalue
```

Several rows marked “yes” actually terminate because `max_nfev` was exceeded. The \(\dot M=0.03\) row is marked converged even though `sonic_crossings = 0`.

This must be fixed before interpreting the branch.

## 1.2 The low-\(\dot M\) eigenvalue is suspicious

For the Paczyński-Wiita potential used in the code,

```math
\frac{l_K(R)}{r_gc}
=
\frac{r^{3/2}}{r-2},
\qquad r=\frac{R}{r_g},
```

so at the ISCO \(r=6\),

```math
\lambda_{K,\rm ISCO}
=
\frac{6^{3/2}}{4}
\simeq 3.674.
```

At very low \(\dot M\) and \(\alpha=0.01\), the swallowed angular momentum should approach the thin-disk value, modulo a modest transonic correction. The current solutions instead give

```text
lambda0 ~= 1.5--1.6
```

which is lower by a factor of about 2.3.

That is a major physical validation failure. It strongly suggests that the free-boundary conditions are underconstrained, poorly scaled, or converging to a spurious minimum.

Do not continue toward the QPE target until this is resolved.

## 1.3 The sonic condition becomes absent by \(\dot M=0.03\)

The audit's `sonic_crossings` drops from one to zero while the global residual still lies below the smoke-test tolerance.

This shows that:

```text
small least-squares residual != valid critical-point solution
```

The sonic residual block needs stronger, smoother, and separately audited conditions.

## 1.4 The high-rate run hits a parameter bound

At \(\dot M/\dot M_{\rm Edd}=1\),

```text
R_son/r_g = 60
```

which is exactly the current upper bound.

That row is not a physical solution. It is a bound-saturated optimizer failure.

Do not simply increase the upper bound and continue. First make the low-rate branch physically correct and add active-bound diagnostics.

---

# 2. Priority Order

Implement the next work in this order:

```text
P0. Redefine convergence and expand diagnostics.
P1. Repair and rescale the sonic regularity residual.
P2. Validate the low-Mdot eigenvalue against the thin-disk limit.
P3. Increase resolution and improve radial node placement.
P4. Replace fixed large Mdot jumps with adaptive/secant continuation.
P5. Add a staged homotopy/Newton solve.
P6. Only if needed, upgrade to an analytic/autodiff global Jacobian.
P7. Add the inner supersonic branch only after the outer sonic eigenproblem passes.
```

Do not add stream feeding, tidal torque, wind, or time dependence yet.

---

# 3. Redefine Solver Status

Replace the single Boolean `converged` with four levels.

```python
@dataclass(frozen=True)
class TransonicSolveStatus:
    optimizer_converged: bool
    equations_converged: bool
    sonic_regular: bool
    physically_valid: bool
```

Suggested definitions:

## 3.1 Optimizer convergence

```python
optimizer_converged = result.success
```

Also store:

```text
result.status
result.message
result.optimality
result.active_mask
result.nfev
result.njev
```

## 3.2 Equation convergence

Require each residual block separately:

```text
max interval radial residual < tol_interval
max interval energy residual < tol_interval
max outer BC residual < tol_outer
```

Production targets:

```text
tol_interval = 1e-7 to 1e-6
tol_outer    = 1e-8 to 1e-7
```

The present `1e-3` is acceptable only for smoke testing.

## 3.3 Sonic regularity

Require:

```text
abs(D_sonic) < tol_D
abs(N_sonic) < tol_N
smin/smax < tol_svd
one physical critical point
```

Initial production targets:

```text
tol_D   = 1e-8
tol_N   = 1e-8
tol_svd = 1e-8
```

These values may need rescaling after the sonic equations are nondimensionalized.

## 3.4 Physical validity

Require all of:

```text
optimizer_converged
equations_converged
sonic_regular
no active bound on R_son or lambda0
Sigma > 0, T > 0, u > 0
Q_visc > 0 over the physical branch
finite optical depth
one sonic crossing
low-Mdot lambda0 sanity check
outer boundary genuinely thin
```

Only `physically_valid=True` should appear as “converged” in scientific tables.

---

# 4. Report Residual Blocks Separately

The present `max_abs_residual` mixes:

```text
interval radial momentum
interval energy
outer Omega condition
outer thermal condition
sonic D
sonic N
```

This hides the actual failure.

Add a diagnostic dataclass:

```python
@dataclass(frozen=True)
class ResidualAudit:
    interval_radial_max: float
    interval_radial_l2: float
    interval_energy_max: float
    interval_energy_l2: float
    outer_omega: float
    outer_energy: float
    sonic_D: float
    sonic_N: float
    sonic_smin_over_smax: float
    active_bounds: tuple[str, ...]
```

Write all fields to:

```text
outputs/tables/transonic_solver_audit.md
```

Also report:

```text
global energy L1
signed global energy residual
Omega_out/OmegaK_out - 1
H/R at outer boundary
Qadv/Qvisc at outer boundary
lambda0/lK_ISCO
```

This will immediately show whether the optimizer is failing in:

```text
the differential equations
the outer boundary
the sonic conditions
or the parameter bounds
```

---

# 5. Rescale the Sonic Differential Matrix Before SVD

The current `sonic_diagnostics()` computes the SVD of the **unscaled** matrix \(A\).

The two rows of \(A\) correspond to:

```text
radial momentum
energy
```

and have different dimensions and dynamic ranges. The singular vectors and compatibility residual therefore depend on unit scaling.

This is likely one of the main causes of the weak/spurious sonic condition.

## 5.1 Build the sonic matrix from scaled equations

Use the same physical scales as the interval residual, but make them smooth.

Current scaling uses `max(...)`, which is non-differentiable. Replace it with root-sum-square scales:

```math
S_R
=
\left[
u^4
+
(R^2\Omega_K^2)^2
+
(\Pi/\Sigma)^2
+
S_{R,\rm floor}^2
\right]^{1/2},
```

```math
S_E
=
\left[
(W\Omega)^2
+
Q_{\rm rad}^2
+
(\Sigma u e/R)^2
+
S_{E,\rm floor}^2
\right]^{1/2}.
```

Define:

```math
\bar F
=
\begin{pmatrix}
F_R/S_R\\
F_E/S_E
\end{pmatrix}
=
\bar A g+\bar c.
```

Perform all sonic diagnostics on \(\bar A,\bar c\), not on the dimensional \(A,c\).

Add:

```python
def scaled_differential_matrix(...):
    # return A_bar, c_bar, radial_scale, energy_scale
```

Use this same function for:

```text
interval equations
sonic diagnostics
unit tests
```

---

# 6. Remove the Nonsmooth SVD-Sign Sonic Compatibility Residual

The current compatibility residual is

```math
N
=
\frac{u_{\rm left}^{T}c}{\|c\|},
```

where `u_left` comes from an SVD.

An SVD singular vector can flip sign under an arbitrarily small state change. This makes \(N\) nonsmooth and can poison the finite-difference global Jacobian.

For a scaled \(2\times2\) system

```math
\bar A=
\begin{pmatrix}
a & b\\
c & d
\end{pmatrix},
\qquad
\bar c=
\begin{pmatrix}
e\\
f
\end{pmatrix},
```

use algebraic compatibility conditions from the adjugate:

```math
C_1=de-bf,
```

```math
C_2=af-ce.
```

At a rank-one critical point, compatibility requires:

```math
\operatorname{adj}(\bar A)\bar c=0,
```

so both \(C_1\) and \(C_2\) vanish.

Use normalized forms:

```math
\hat C_1
=
\frac{de-bf}
{\sqrt{d^2+b^2}\sqrt{e^2+f^2}+\epsilon},
```

```math
\hat C_2
=
\frac{af-ce}
{\sqrt{a^2+c^2}\sqrt{e^2+f^2}+\epsilon}.
```

Keep a signed determinant:

```math
\hat D
=
\frac{ad-bc}
{\sqrt{a^2+c^2}\sqrt{b^2+d^2}+\epsilon}.
```

Recommended sonic residual block:

```text
[D_hat, C1_hat, C2_hat]
```

This makes the least-squares problem overdetermined by one residual, which is fine. One compatibility equation is redundant at the exact solution, but both improve robustness away from it.

Continue to report `smin/smax` as a diagnostic, but do not use an SVD vector in the residual function.

Update the expected residual length and Jacobian sparsity pattern accordingly.

---

# 7. Verify That the Critical Mode Is Radial

A matrix can become singular for a thermal degeneracy rather than a radial sonic transition.

At the candidate sonic point, inspect the right null vector:

```math
v_{\rm null}
=
(v_u,v_T).
```

Require a substantial radial-velocity component:

```text
abs(v_u) / sqrt(v_u^2 + v_T^2) > 0.3
```

as a first diagnostic threshold.

Also compute a physically interpretable effective Mach number. A useful development diagnostic is:

```math
{\cal M}_{\rm eff}
=
\frac{u}{c_{\rm eff}},
```

where \(c_{\rm eff}\) can initially be estimated from the local effective compressibility of the vertically integrated closure:

```math
c_{\rm eff}^2
\sim
\left(\frac{\partial\Pi}{\partial\Sigma}\right)_s.
```

The exact critical condition is still the matrix singularity, but a valid sonic point should have:

```text
M_eff of order unity
```

and should not be a purely thermal singularity.

Add these to the audit:

```text
null_radial_fraction
M_eff_at_sonic
```

---

# 8. Low-\(\dot M\) Validation Must Come Before Continuation

The current low-rate solution is not yet acceptable because \(\lambda_0\) is far below the expected thin-disk value.

For the Paczyński-Wiita potential:

```math
\lambda_{K,\rm ISCO}=3.674.
```

At:

```text
Mdot/Mdot_Edd = 1e-3
alpha = 0.01
```

require the transonic result to approach this value.

Suggested sanity target:

```text
0.8 < lambda0/lambda_K_ISCO < 1.1
```

This is not a mathematical boundary condition; it is a validation check. If the physical slim-disk equations predict a slightly different value, it should converge toward the thin-disk value as \(\dot M\to0\).

## 8.1 Direct profile comparison

At:

```text
1e-3
3e-3
1e-2
```

compare the transonic and repaired reduced profiles outside the inner region, for example:

```text
R > 15 r_g
```

Plot fractional differences in:

```text
Sigma
T
H/R
u
Omega/Omega_K
Qrad
```

Targets away from boundaries:

```text
Sigma agreement < 3 percent
T agreement < 3 percent
H/R agreement < 5 percent
Omega/OmegaK - 1 < 1 percent
Qadv/Qvisc << 1 at 1e-3
```

If these tests fail, do not continue in \(\dot M\).

## 8.2 Inspect the unexpectedly large thickness

The current transonic sequence reaches:

```text
H/R = 0.22 at Mdot/Mdot_Edd = 0.02
H/R = 0.343 at Mdot/Mdot_Edd = 0.03
```

Compare these values directly to the repaired reduced branch. If the reduced branch is much thinner at the same rates, the transonic solution is on the wrong eigen-branch.

---

# 9. Increase Radial Resolution and Cluster Nodes Near the Sonic Point

The production script currently uses approximately:

```text
n_nodes = 18
R_out = 300 r_g
```

for the smoke scan.

This is too coarse for a free-boundary sonic problem. The logarithmic interval from roughly \(4r_g\) to \(300r_g\) spans more than four e-folds, so only a few intervals resolve the inner critical region.

## 9.1 Production grid

For low-rate validation use:

```text
N = 32, 48, 64, 96
R_out = 1000 r_g
max_nfev >= 1000
```

## 9.2 Sonic-point clustering

Replace the uniform computational mapping

```math
x=x_{\rm son}+\xi(x_{\rm out}-x_{\rm son})
```

with:

```math
x=x_{\rm son}+\xi^p(x_{\rm out}-x_{\rm son}),
```

with:

```text
p = 1.5--2
```

to cluster nodes near the sonic point.

Audit:

```text
R_son
lambda0
D, C1, C2
energy L1
integrated advective fraction
```

against both \(N\) and \(p\).

A valid branch should converge as resolution increases.

---

# 10. Use a Genuinely Thin Outer Boundary

For the isolated benchmark, use:

```text
R_out = 1000--3000 r_g
```

and verify at the outer node:

```text
H/R < 0.05
abs(Qadv/Qvisc) < 0.01
abs(Omega/OmegaK - 1) < 0.01
```

The current boundary conditions are:

```text
Omega = Omega_K
Qvisc_thin = Qrad
```

These are reasonable, but test sensitivity against an alternative that matches the repaired reduced solution:

```math
B_u=\ln(u/u_{\rm reduced})=0,
```

```math
B_T=\ln(T/T_{\rm reduced})=0.
```

Run both outer closures at low \(\dot M\). The eigenvalue and sonic radius should agree as \(R_{\rm out}\) is increased.

If they do not, the solution remains boundary dominated.

---

# 11. Add Active-Bound Diagnostics and Do Not Silently Clip

Report whether any variable lies within a small fraction of a bound:

```python
distance = min((z-lower)/(upper-lower), (upper-z)/(upper-lower))
```

Flag:

```text
distance < 1e-4
```

At minimum report active bounds for:

```text
R_son
lambda0
log u at every node
log T at every node
```

The \(\dot M=1\) run has `R_son=60 r_g`, so it must be flagged as invalid.

Do not expand the sonic-radius bound until the low-rate solution is validated. After validation, use a larger high-rate bound such as:

```text
R_son < 200--300 r_g
```

for exploratory high-rate continuation.

A cleaner later implementation can transform bounded global parameters with smooth logistic maps, avoiding active-set kinks.

---

# 12. Replace Fixed-Step Continuation With Adaptive Secant Prediction

The current continuation remaps one previous profile and applies:

```python
T_new = T_old * (Mdot_new/Mdot_old)**0.25
```

while keeping the previous \(R_{\rm son}\) and \(\lambda_0\).

This is too weak near a rapidly moving sonic point.

## 12.1 Adaptive log-\(\dot M\) continuation

Implement:

```text
initial Delta ln Mdot = 0.03--0.05
successful fast solve -> increase step by 1.2
slow solve -> keep step
failure -> halve step and retry from last valid solution
minimum Delta ln Mdot = 1e-3
```

Never jump from a failed \(0.05\) solution to \(0.1\) or \(1\).

## 12.2 Secant predictor

With the last two physically valid solutions, predict the full unknown vector:

```math
z_{\rm pred}
=
z_n
+
\frac{p_{n+1}-p_n}{p_n-p_{n-1}}
(z_n-z_{n-1}),
```

where:

```math
p=\ln\dot M.
```

This predictor must include:

```text
all log u nodes
all log T nodes
log R_son
lambda0
```

Remap both old solutions to the new computational grid before forming the secant.

## 12.3 Pseudo-arclength later

If a fold is encountered after the branch is physically validated, add pseudo-arclength continuation. Do not use it to bypass a failed low-rate eigenvalue check.

---

# 13. Use a Staged Homotopy Solve Before Building a Full Analytic Jacobian

The current audit recommends either a true analytic global Jacobian or a staged Newton solve.

The block-local finite-difference Jacobian is not obviously the main failure yet. It already reproduces the sparsity structure. First remove nonsmooth sonic residuals and use a staged solve.

## Stage A: Profile solve with fixed eigenparameters

Fix:

```text
R_son near 4--6 r_g
lambda0 near 3.674
```

Solve only:

```text
interval equations
outer boundary equations
```

for the nodal profiles.

## Stage B: Free \(R_{\rm son}\), hold \(\lambda_0\)

Add the signed determinant/compatibility residuals and free \(R_{\rm son}\), while keeping \(\lambda_0\) close to the thin value.

## Stage C: Free both eigenparameters

Free \(R_{\rm son}\) and \(\lambda_0\), and solve the complete overdetermined least-squares system.

## Stage D: Tighten tolerances

Progressively reduce residual targets:

```text
1e-3 -> 1e-5 -> 1e-7
```

and increase resolution.

This homotopy is preferable to asking the full free-boundary solve to jump directly from a rough initial profile to a critical eigen-solution.

---

# 14. Jacobian Strategy

After the sonic residual is made smooth:

## 14.1 First choice

Keep the block-local sparse finite-difference Jacobian, but add tests against a full finite-difference Jacobian for small grids.

Require:

```text
relative column error < 1e-5--1e-4
```

for every block type:

```text
interval
outer boundary
sonic
```

## 14.2 Second choice

If convergence still stalls, use automatic differentiation or an exact sparse Jacobian.

Possible options:

```text
JAX
complex-step differentiation for smooth algebraic blocks
hand-derived sparse derivatives
```

Complex-step will not work through SVD branch/sign logic, another reason to replace the SVD-vector residual.

Do not invest in a full analytic Jacobian before fixing:

```text
residual scaling
sonic smoothness
status definition
continuation
grid resolution
```

---

# 15. Improve the Collocation Scheme After the Solver Is Stable

The current midpoint collocation is acceptable for development, but it is low order.

After a valid low-rate branch is obtained:

```text
- compare midpoint collocation against trapezoidal/Hermite-Simpson;
- verify second-order or better convergence;
- use mesh refinement based on interval residuals;
- add more nodes where D changes rapidly.
```

Do not change the discretization and the sonic equations simultaneously; isolate one source of error at a time.

---

# 16. Recommended Immediate Codex Sprint

Implement the following concrete tasks.

## Task 1: Scientific convergence status

Modify:

```text
transonic_collocation.py
transonic_solver_audit.md generator
```

to distinguish optimizer, equation, sonic, and physical validity.

## Task 2: Residual block audit

Add per-block residuals, active-bound reporting, \(\lambda_0/\lambda_{K,\rm ISCO}\), and outer thinness diagnostics.

## Task 3: Scaled smooth sonic residual

Build \(\bar A,\bar c\), replace SVD-vector \(N\) by:

```text
D_hat
C1_hat
C2_hat
```

and keep `smin/smax` only as a diagnostic.

## Task 4: Low-rate production solve

Run:

```text
Mdot/Mdot_Edd = 1e-3
N = 32, 48, 64, 96
R_out = 1000, 3000 r_g
p_grid = 1, 1.5, 2
max_nfev >= 1000
```

Require a physically valid solution with \(\lambda_0\) close to the PW thin-disk value.

## Task 5: Reduced/transonic comparison

Generate a profile-comparison figure at:

```text
1e-3
3e-3
1e-2
```

## Task 6: Staged homotopy

Implement fixed-eigenparameter, one-free-parameter, and full-free-boundary stages.

## Task 7: Adaptive continuation

Continue only after the \(10^{-3}\) solution passes every validation criterion.

---

# 17. New Tables and Figures

Create:

```text
outputs/tables/transonic_residual_blocks.md
outputs/tables/transonic_low_mdot_convergence.md
outputs/tables/transonic_active_bounds.md
outputs/figures/transonic_reduced_comparison.png
outputs/figures/transonic_sonic_block_audit.png
outputs/figures/transonic_resolution_convergence.png
```

## 17.1 Sonic audit figure

Plot near the sonic point:

```text
D_hat
C1_hat
C2_hat
smin/smax
M_eff
radial null-vector fraction
```

## 17.2 Low-rate comparison

Plot:

```text
Sigma_trans/Sigma_reduced - 1
T_trans/T_reduced - 1
H_trans/H_reduced - 1
Omega/OmegaK - 1
Qadv/Qvisc
```

## 17.3 Continuation summary

Use distinct markers:

```text
optimizer only
equations converged
sonic regular
physically valid
bound saturated
```

Do not label all residual-tolerance solutions as converged.

---

# 18. Go/No-Go Criteria

## Proceed to higher \(\dot M\) only if

At \(\dot M/\dot M_{\rm Edd}=10^{-3}\):

```text
optimizer success
max interval residual < 1e-6
outer residuals < 1e-7
abs(D_hat), abs(C1_hat), abs(C2_hat) < 1e-8
smin/smax < 1e-8
exactly one physical sonic point
no active bounds
lambda0/lambda_K_ISCO between about 0.8 and 1.1
outer transonic/reduced profiles agree to a few percent
R_son and lambda0 converge with N and R_out
```

## Proceed to the inner supersonic extension only if

```text
the outer sonic eigen-solution is physically valid through at least
Mdot/Mdot_Edd ~ 1--10
```

## Add wind only if

```text
the no-wind transonic solver is validated and either:
- H/R approaches or exceeds unity at high Mdot, or
- no regular no-wind solution exists only at high Mdot.
```

## Treat the current astrophysical model as untested, not rejected, if

```text
the numerical eigenproblem has not yet passed the low-Mdot thin-disk limit.
```

---

# 19. Compact Codex Prompt

```text
Harden the current transonic collocation solver before interpreting the
high-Mdot failure.

1. Replace the single `converged` flag with optimizer_converged,
   equations_converged, sonic_regular, and physically_valid.

2. Report residual blocks separately:
   interval radial, interval energy, outer Omega, outer energy,
   sonic determinant/compatibility, active bounds, and optimality.

3. Construct sonic diagnostics from the same smoothly scaled radial and energy
   equations used by collocation. Replace max-based scales with RSS scales.

4. Remove the SVD-left-vector compatibility residual. For scaled
   A=[[a,b],[c,d]], cvec=[e,f], use:
       D  = normalized(ad-bc)
       C1 = normalized(de-bf)
       C2 = normalized(af-ce)
   Keep smin/smax only as a diagnostic.

5. Validate the low-Mdot eigenvalue. In the PW potential,
   lK_ISCO/(rg c)=6^(3/2)/4=3.674. At Mdot/Mdot_Edd=1e-3 and alpha=0.01,
   lambda0 should approach this thin-disk value. The current lambda0~1.5 is
   a red flag.

6. Compare the transonic and repaired reduced profiles outside 15 rg at
   Mdot/Mdot_Edd=1e-3, 3e-3, and 1e-2.

7. Use N=32,48,64,96; Rout=1000 and 3000 rg; max_nfev>=1000; and a grid
   mapping x=xson+xi^p(xout-xson), p=1.5--2.

8. Add staged homotopy:
   A. fixed Rson and lambda0;
   B. free Rson with lambda0 fixed;
   C. free both eigenparameters;
   D. tighten residual tolerances.

9. Add adaptive continuation in log Mdot with a full-state secant predictor.
   On failure, halve the step and retry from the last physically valid state.

10. Do not add stream, tide, wind, time dependence, or inner supersonic
    continuation until the low-rate sonic eigen-solution passes all tests.
```

---

# 20. Primary References

Follow the critical-point/eigenvalue logic in:

1. Abramowicz et al. 1988, *Slim Accretion Disks*  
   https://ui.adsabs.harvard.edu/abs/1988ApJ...332..646A/abstract

2. Sądowski 2009, *Slim Disks Around Kerr Black Holes Revisited*  
   https://arxiv.org/abs/0906.0355

3. Sądowski 2011, *Slim Accretion Disks Around Black Holes*  
   https://arxiv.org/abs/1108.0396

The key principle is:

```text
A transonic slim-disk solution is an eigenvalue problem. A small global
least-squares residual is not sufficient unless the critical point is regular,
the inner angular momentum is physically sensible, and the solution converges
with grid, outer boundary, and continuation step.
```
