# Codex Next-Step Plan: Transonic Slim-Disk Solver for the IMRI QPE Minidisk Project

This document is an implementation brief for the next Codex development sprint in:

```text
https://github.com/huanyang07/IMBH
```

It was written after reviewing, in order:

```text
GPT_REPO_HANDOFF.md
README.md
outputs/tables/isolated_slim_branch_summary.md
outputs/tables/slim_wind_upgrade_audit.md
Note/CODEX_REPAIR_ISOLATED_SLIM_SOLVER.md
Note/CODEX_GLOBAL_SLIM_NEXT_STEPS.md
src/imri_qpe/layer3_minidisk_1d/isolated_slim_solver.py
src/imri_qpe/layer3_minidisk_1d/global_slim.py
src/imri_qpe/layer3_minidisk_1d/entropy_advection.py
tests/test_isolated_slim_solver.py
```

The current repaired isolated solver is a real success:

```text
- the thin-disk limit is recovered;
- the old failure near Mdot/Mdot_Edd = 0.03 was numerical/conventional;
- a smooth reduced advective sequence is obtained through Mdot/Mdot_Edd ~= 10;
- Q_adv/Q_visc rises to ~= 0.77 at Mdot/Mdot_Edd = 10;
- energy residuals remain small through that sequence.
```

The remaining failure is now physical/model-form rather than an immediate coding failure:

```text
H/R exceeds 0.3 near Mdot/Mdot_Edd ~= 1.5
H/R exceeds 0.4 near Mdot/Mdot_Edd ~= 3
H/R ~= 0.60 at Mdot/Mdot_Edd = 10
QPE target Mdot/Mdot_Edd ~= 94
```

Therefore the nearly Keplerian reduction becomes unreliable well below the QPE target. The correct next calculation is an **isolated, no-wind, transonic slim-disk solver with radial momentum and sonic regularity**.

Do not reintroduce stream feeding, tidal truncation, or wind during this sprint.

Implementation status after first Codex pass:

```text
Milestone T1 has been scaffolded and smoke-tested.

Implemented:
- Paczynski-Wiita potential helpers.
- Transonic vertical thermodynamics and algebraic alpha stress.
- Local differential residual F(g)=A g+c with numerical partials.
- Sonic matrix diagnostics D and N.
- Free-boundary midpoint collocation residual.
- Block-local sparse finite-difference Jacobian for the collocation system.
- scipy.optimize.least_squares solver wrapper.
- Mdot continuation/remapping helpers.
- First diagnostic figure/table:
  outputs/figures/transonic_branch_summary.png
  outputs/tables/transonic_solver_audit.md

Current result:
- Analytic local partials now replace the original finite-difference local
  partials.
- Mdot/Mdot_Edd = 1e-3 converges with max residual ~= 7.8e-5.
- Mdot/Mdot_Edd = 0.003, 0.01, 0.02, and 0.03 also satisfy the current smoke-test
  residual tolerance.
- Sonic regularity is satisfied at a free sonic radius on the converged
  low-rate branch.
- Continuation is not robust yet at Mdot/Mdot_Edd = 0.05, 0.1, or 1.

Current numerical caveat:
- The block-local sparse Jacobian improves the continuation frontier, but the
  solver still uses finite differences for global block derivatives. The next
  hardening step is a true analytic global Jacobian or staged Newton solve
  plus better continuation/scaling.
```

---

# 1. Executive Decision

Implement the transonic solver in two milestones.

## Milestone T1: Outer transonic eigenvalue problem

Build a Schwarzschild, pseudo-Newtonian, vertically integrated slim-disk solver that:

```text
- includes radial momentum;
- permits non-Keplerian rotation;
- computes entropy advection from the radial solution;
- treats the sonic radius as a free boundary;
- treats the swallowed angular momentum l0 as an eigenvalue;
- solves the subsonic branch from the sonic point to a large outer radius;
- imposes two sonic regularity conditions.
```

The recommended numerical method is an **undivided free-boundary collocation solve**, using `scipy.optimize.least_squares`, rather than direct shooting or `solve_bvp` through a singular interior point.

## Milestone T2: Inner supersonic continuation

After T1 is validated:

```text
- calculate a regular derivative at the sonic point;
- continue the solution inward toward the pseudo-horizon;
- verify a single smooth transonic crossing;
- close global mass, angular-momentum, and energy budgets.
```

Only after T1 and T2 work should the project add:

```text
stream source -> tidal edge -> wind -> time dependence
```

---

# 2. Why a Pseudo-Newtonian Baseline Is the Best Next Step

Do not start with Kerr GR.

Use the Paczynski-Wiita potential for the first transonic solver:

```math
\Phi(R)=-\frac{GM_2}{R-R_{\rm PW}},
\qquad
R_{\rm PW}=\frac{2GM_2}{c^2}.
```

Then

```math
\Omega_K^2(R)
=
\frac{GM_2}{R(R-R_{\rm PW})^2},
```

and

```math
l_K(R)=R^2\Omega_K(R).
```

The marginally stable orbit is:

```math
R_{\rm ISCO}=3R_{\rm PW}=\frac{6GM_2}{c^2}.
```

This model captures:

```text
- a pseudo-horizon;
- an ISCO;
- rapid inner acceleration;
- a transonic eigenvalue problem;
- the main mathematical structure needed to test the hot branch.
```

Once the solver is validated, the potential module can later be replaced by relativistic Kerr functions.

Use distinct notation:

```text
R_PW   = pseudo-horizon radius
R_son  = sonic radius
```

Do not call both of them `R_s`.

---

# 3. Choice of Stress Closure

The repaired reduced solver currently uses:

```math
\nu=\alpha H^2\Omega_K,
```

```math
W=\frac{3}{2}\nu\Sigma\Omega_K.
```

For the first transonic solver, use the standard algebraic alpha-stress closure employed in classical slim-disk models:

```math
W_{R\phi}
=
C_\alpha\,\alpha\,(2H)\,
P_{\rm gas}^{\mu}
P_{\rm tot}^{1-\mu}.
```

Baseline:

```text
mu_stress = 0
C_alpha   = 1
```

so:

```math
W_{R\phi}=\alpha\Pi,
\qquad
\Pi=2HP_{\rm tot}.
```

This makes angular momentum algebraic and reduces the transonic problem to two first-order differential equations.

## 3.1 Compatibility with the repaired solver

In the thin Keplerian limit, the current closure is approximately:

```math
W_{\rm old}
\simeq
\frac{3}{2}\alpha\Pi.
```

Therefore add a parameter:

```python
stress_factor = C_alpha
```

and support:

```text
C_alpha = 1.5   # migration/comparison mode
C_alpha = 1.0   # standard alpha-stress literature mode
```

First verify that `C_alpha = 1.5` approximately reproduces the repaired reduced sequence at low `Mdot`. Then use `C_alpha = 1` as the standard baseline and interpret the difference as an alpha normalization.

Do not mix:

```text
W = alpha Pi
```

with:

```text
W = -nu Sigma R dOmega/dR
```

inside one run. A diffusive-stress transonic solver can be added later as a separate mode with `Omega(R)` as a third ODE.

---

# 4. Variables and Sign Conventions

Use inward speed:

```math
u\equiv-v_R>0.
```

Use inward-positive accretion rate:

```math
\dot M=2\pi R\Sigma u>0.
```

Use logarithmic radius:

```math
x=\ln R.
```

Use primary differential variables:

```math
y_1=\ln u,
\qquad
y_2=\ln T.
```

The surface density is algebraic:

```math
\Sigma
=
\frac{\dot M}{2\pi Ru}.
```

Therefore:

```math
\frac{d\ln\Sigma}{dx}
=
-1-\frac{d\ln u}{dx}.
```

Let:

```math
g_u=\frac{d\ln u}{dx},
\qquad
g_T=\frac{d\ln T}{dx}.
```

---

# 5. Vertical and Thermodynamic Closure

Reuse the existing gas+radiation thermodynamics, but replace the Newtonian `Omega_K` with the pseudo-Newtonian value.

Use:

```math
\rho=\frac{\Sigma}{2H},
```

```math
P_{\rm gas}=\rho\mathcal R T,
\qquad
\mathcal R=\frac{k_B}{\mu m_p},
```

```math
P_{\rm rad}=\frac{a_{\rm r}T^4}{3},
```

```math
P_{\rm tot}=P_{\rm gas}+P_{\rm rad},
```

```math
\Pi=2HP_{\rm tot}.
```

Vertical hydrostatic equilibrium gives:

```math
\Omega_K^2 H^2
=
\mathcal R T
+
\frac{2a_{\rm r}T^4}{3\Sigma}H.
```

Use the positive analytic root already implemented in `vertical_structure_arrays`.

Specific internal energy:

```math
e
=
\frac{\mathcal R T}{\gamma_{\rm gas}-1}
+
\frac{a_{\rm r}T^4}{\rho}.
```

Optical depth:

```math
\tau_{\rm es}=\frac{\kappa_{\rm es}\Sigma}{2}.
```

For this first benchmark, keep electron scattering opacity and diffusion cooling:

```math
Q_{\rm rad}^{-}
=
\frac{16\sigma_{\rm SB}T^4}
{3\kappa_{\rm es}\Sigma}.
```

This is a two-face cooling convention.

---

# 6. Angular Momentum as an Algebraic Closure

The steady angular-momentum integral is:

```math
\dot M(l-l_0)
=
2\pi R^2W_{R\phi}.
```

Therefore:

```math
l(R)
=
l_0
+
\frac{2\pi R^2W_{R\phi}}{\dot M},
```

and:

```math
\Omega(R)=\frac{l(R)}{R^2}.
```

Here:

```text
l0 is not fixed to l_K(R_ISCO)
l0 is an eigenvalue selected by sonic regularity
```

At low accretion rates, the converged value should approach the standard thin-disk inner angular momentum.

Diagnostics:

```text
Omega/Omega_K
l/l_K
W
l0
```

The solution may be mildly sub-Keplerian outside and can become super-Keplerian in part of the inner region. Do not clip `Omega/Omega_K`.

---

# 7. Differential Equations

## 7.1 Radial momentum

Use:

```math
u\frac{du}{dR}
=
R(\Omega^2-\Omega_K^2)
-
\frac{1}{\Sigma}\frac{d\Pi}{dR}.
```

In logarithmic radius:

```math
u^2g_u
=
R^2(\Omega^2-\Omega_K^2)
-
\frac{1}{\Sigma}\frac{d\Pi}{dx}.
```

Define the radial-momentum residual:

```math
F_R
=
u^2g_u
-
R^2(\Omega^2-\Omega_K^2)
+
\frac{1}{\Sigma}\frac{d\Pi}{dx}.
```

A solution requires:

```math
F_R=0.
```

## 7.2 Viscous heating

Use the stress-shear expression:

```math
Q_{\rm visc}^{+}
=
-W_{R\phi}\frac{d\Omega}{dx}.
```

Because normal disk rotation has `dOmega/dx < 0`, this is positive.

Do not use a hard-coded Keplerian factor such as `9/4` in the transonic solver.

## 7.3 Entropy advection

Use the existing first-law expression:

```math
T\frac{ds}{dx}
=
\frac{de}{dx}
-
\frac{P_{\rm tot}}{\rho^2}\frac{d\rho}{dx}.
```

Then:

```math
Q_{\rm adv}
=
\Sigma v_R T\frac{ds}{dR}
=
-\frac{\Sigma u}{R}
T\frac{ds}{dx}.
```

Keep the sign. Do not force `Q_adv >= 0`.

## 7.4 Energy equation

Use:

```math
Q_{\rm visc}^{+}
=
Q_{\rm rad}^{-}
+
Q_{\rm adv}.
```

Define:

```math
F_E
=
Q_{\rm visc}^{+}
-
Q_{\rm rad}^{-}
-
Q_{\rm adv}.
```

A solution requires:

```math
F_E=0.
```

---

# 8. Construct the Local Differential Matrix Numerically

The radial momentum and energy residuals are linear in:

```math
g=
\begin{pmatrix}
g_u\\
g_T
\end{pmatrix}.
```

Write:

```math
F(g)=Ag+c,
```

where:

```math
F=
\begin{pmatrix}
F_R\\
F_E
\end{pmatrix}.
```

Do not initially derive all matrix coefficients by hand. Build them from thermodynamic partial derivatives.

## 8.1 Recommended local partial-derivative method

At a local state:

```text
x = ln R
y = [ln u, ln T]
```

compute each closure quantity `q(x,y)` and its partial derivatives:

```math
q_x
=
\left.\frac{\partial q}{\partial x}\right|_y,
```

```math
q_{y_j}
=
\left.\frac{\partial q}{\partial y_j}\right|_x.
```

Then along a trial gradient `g`:

```math
\frac{dq}{dx}
=
q_x
+
q_y\cdot g.
```

Use centered finite differences in log variables for the first implementation:

```python
eps_x = 1e-5
eps_y = 1e-5
```

with convergence tests over:

```text
1e-4, 3e-5, 1e-5, 3e-6
```

Calculate partials for at least:

```text
Pi
rho
e
Omega
```

Then form `F_R(g)` and `F_E(g)`.

## 8.2 Matrix construction

Evaluate:

```python
c = F(g=[0, 0])
A[:, 0] = F(g=[1, 0]) - c
A[:, 1] = F(g=[0, 1]) - c
```

Away from the sonic point:

```math
Ag=-c.
```

The local gradient is:

```python
g = np.linalg.solve(A, -c)
```

This approach automatically includes:

```text
- gas+radiation thermodynamics;
- vertical-thickness response;
- pressure-gradient response;
- the state dependence of Omega through alpha stress;
- entropy advection.
```

## 8.3 Required local tests

Add manufactured tests verifying:

```text
F(g) - (A @ g + c) is small for random g
```

Target:

```text
relative linearity error < 1e-7--1e-5
```

depending on finite-difference step.

---

# 9. Sonic Regularity Conditions

At the sonic point, the local differential matrix becomes singular.

Do not define the sonic point only by:

```text
u = c_s
```

because the true critical speed depends on vertical closure, radiation pressure, and the coupled energy equation.

Use the differential matrix itself.

Let:

```math
A=USV^T.
```

## 9.1 Signed critical condition

Use a signed normalized determinant:

```math
D
=
\frac{\det A}
{\|A_{:,0}\|\,\|A_{:,1}\|+\epsilon}.
```

The critical condition is:

```math
D=0.
```

Also record:

```math
s_{\min}/s_{\max}
```

from the singular values as a conditioning diagnostic.

## 9.2 Compatibility condition

At singularity, the differential equations have a finite solution only if `c` lies in the range of `A`.

Let `u_null` be the left singular vector corresponding to the smallest singular value.

Define:

```math
N
=
\frac{u_{\rm null}^{T}c}
{\|c\|+\epsilon}.
```

The regularity condition is:

```math
N=0.
```

Thus the sonic point satisfies:

```math
D(R_{\rm son})=0,
\qquad
N(R_{\rm son})=0.
```

This is the numerical analogue of the classical slim-disk conditions:

```text
denominator = 0
numerator   = 0
```

The two free quantities selected by these conditions are:

```text
R_son
l0
```

---

# 10. Recommended Numerical Method: Free-Boundary Collocation

Do not use direct shooting as the main solver in the first implementation.

Do not use `scipy.integrate.solve_bvp` across an interior singular point without reformulating the problem.

Use a free-boundary collocation system that never divides by `det(A)`.

## 10.1 Coordinate mapping

Use a fixed computational coordinate:

```math
\xi\in[0,1].
```

Map it to radius:

```math
\ln R(\xi)
=
\ln R_{\rm son}
+
\xi
\left(
\ln R_{\rm out}
-
\ln R_{\rm son}
\right).
```

The sonic point is always node `0`. The outer boundary is node `N-1`.

Unknowns:

```text
ln u_i, i = 0...N-1
ln T_i, i = 0...N-1
ln R_son
lambda0 = l0/(r_g c)
```

Total unknown count:

```text
2N + 2
```

## 10.2 Interval residuals

For each interval `i -> i+1`, use midpoint collocation:

```math
y_m=\frac{y_i+y_{i+1}}{2},
```

```math
x_m=\frac{x_i+x_{i+1}}{2},
```

```math
g_m=
\frac{y_{i+1}-y_i}
{x_{i+1}-x_i}.
```

Evaluate:

```math
F_m=A(x_m,y_m)g_m+c(x_m,y_m).
```

Set:

```math
F_m=0.
```

This gives:

```text
2(N-1) residuals
```

and remains finite even when the first node is critical, because the first interval midpoint is outside the exact sonic point.

## 10.3 Outer boundary conditions

At `R_out`, impose two thin-disk matching conditions using the same stress and opacity closure.

Recommended:

```math
B_{\Omega}
=
\ln\left(\frac{\Omega_{\rm out}}{\Omega_{K,\rm out}}\right)
=0,
```

and:

```math
B_E
=
\frac{
Q_{\rm visc,thin}^{+}
-
Q_{\rm rad}^{-}
}{
Q_{\rm visc,thin}^{+}
+
Q_{\rm rad}^{-}
}
=0.
```

For the pseudo-Newtonian Keplerian shear:

```math
\frac{d\ln\Omega_K}{d\ln R}
=
-\frac{1}{2}
-
\frac{R}{R-R_{\rm PW}}.
```

Therefore:

```math
Q_{\rm visc,thin}^{+}
=
-W\Omega_K
\frac{d\ln\Omega_K}{d\ln R}.
```

Choose `R_out` large enough that the converged boundary has:

```text
H/R < 0.05--0.1
|Q_adv/Q_visc| < 0.01
|Omega/Omega_K - 1| < 0.01
```

For the literature benchmark use:

```text
R_out = 1e3 r_g,2
```

or larger.

For the later physical IMRI minidisk use:

```text
R_out ~= 0.3 R_H ~= 2.2e2 r_g,2
```

but only after the isolated benchmark works.

## 10.4 Sonic boundary conditions

At node `0`, impose:

```math
D=0,
```

```math
N=0.
```

These add two residuals.

Total residual count:

```text
2(N-1) interval residuals
+ 2 outer boundary residuals
+ 2 sonic regularity residuals
= 2N + 2
```

which matches the unknown count exactly.

## 10.5 Nonlinear solver

Add SciPy as an optional solver dependency:

```toml
[project.optional-dependencies]
solver = ["scipy>=1.11"]
```

Use:

```python
scipy.optimize.least_squares
```

with:

```text
log variables
scaled residuals
jac_sparsity
trust-region reflective algorithm
```

Suggested call:

```python
least_squares(
    residual,
    z0,
    jac_sparsity=pattern,
    x_scale="jac",
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
    max_nfev=...
)
```

The Jacobian is block-banded because each interval residual depends only on neighboring nodes, plus the two global parameters.

Do not use hard clipping of:

```text
H/R
Q_adv
xi_eff
Omega
```

A trial state outside physical bounds should receive a smooth penalty residual rather than being silently clipped.

---

# 11. Initial Guess and Continuation

## 11.1 Start at low accretion rate

Begin with:

```text
Mdot/Mdot_Edd = 1e-3
alpha = 0.01
mu_stress = 0
no wind
```

Initial guesses:

```text
R_son ~= R_ISCO
l0 ~= l_K(R_ISCO)
```

Use the repaired reduced solution for the outer profile.

Construct the inner trial profile by smoothly increasing `u` toward a sonic value and extrapolating `T` inward.

A rough trial is sufficient because the collocation solve enforces regularity.

## 11.2 Continue in accretion rate

Use adaptive continuation:

```text
1e-3
3e-3
1e-2
3e-2
0.1
0.3
1
1.5
2
3
5
10
20
30
50
94
```

Internally use:

```text
Delta ln Mdot = 0.03--0.1
```

and halve the step after failure.

Only continue from a converged solution.

Remap the previous solution to the new sonic-radius grid before using it as the next initial guess.

## 11.3 Continuation order

Recommended:

```text
A. alpha = 0.01, Mdot continuation
B. alpha continuation at fixed Mdot
C. stress_factor 1.5 -> 1.0
D. mu_stress scan only after baseline is stable
```

---

# 12. Inner Supersonic Extension: Milestone T2

After the outer free-boundary solution is validated, continue inward.

## 12.1 Do not divide naively by a singular matrix

At the sonic point:

```math
Ag=-c
```

is rank deficient.

The regular sonic derivative can be found by differentiating the differential equations along the solution.

A practical implementation is:

1. solve for the sonic state using `D=0` and `N=0`;
2. solve for `g_son` using:
   ```math
   A g + c = 0
   ```
   plus the differentiated compatibility condition;
3. calculate the required derivatives numerically from local partials;
4. use the physical inward root;
5. start integration at:
   ```math
   R=R_{\rm son}(1-\epsilon),
   \qquad
   \epsilon\sim10^{-5}-10^{-4};
   ```
6. integrate inward with an implicit ODE solver.

Alternative:

```text
add an inner collocation domain sharing the sonic node
```

and solve the whole subsonic+supersonic solution simultaneously. This is more robust but can be deferred until T1 works.

## 12.2 Inner boundary diagnostic

Near the pseudo-horizon require:

```text
u increases inward
l -> l0
finite entropy
finite positive Sigma and T
no second sonic crossing
```

Do not impose a zero torque at the ISCO. The sonic point and inner torque are outputs.

---

# 13. Dimensionless Scaling

The transonic problem will be better conditioned in dimensionless units.

Recommended scales:

```math
r_g=\frac{GM_2}{c^2},
```

```math
u_0=c,
```

```math
\Omega_0=\frac{c}{r_g},
```

```math
l_0^{\rm unit}=r_gc,
```

```math
\Sigma_0
=
\frac{\dot M_{\rm Edd}}
{2\pi r_gc},
```

```math
\Pi_0=\Sigma_0c^2,
```

```math
Q_0
=
\frac{\dot M_{\rm Edd}c^2}
{2\pi r_g^2},
```

```math
T_0
=
\frac{\mu m_pc^2}{k_B}.
```

The code may retain cgs outputs, but the nonlinear solver should work with dimensionless log variables and order-unity residuals.

Always write the Eddington-rate convention into output metadata:

```math
\dot M_{\rm Edd}
=
\frac{L_{\rm Edd}}{0.1c^2}.
```

---

# 14. Suggested Code Structure

Add:

```text
src/imri_qpe/layer3_minidisk_1d/transonic_potential.py
src/imri_qpe/layer3_minidisk_1d/transonic_thermo.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
src/imri_qpe/layer3_minidisk_1d/transonic_continuation.py
scripts/plot_transonic_slim_branch.py
```

## 14.1 `transonic_potential.py`

Suggested interface:

```python
@dataclass(frozen=True)
class PaczynskiWiitaPotential:
    M_g: float

    def r_g(self) -> float: ...
    def r_pw(self) -> float: ...
    def r_isco(self) -> float: ...
    def phi(self, R): ...
    def dphi_dR(self, R): ...
    def omega_k(self, R): ...
    def dln_omega_k_dlnR(self, R): ...
    def l_k(self, R): ...
```

Tests should compare analytic and finite-difference derivatives.

## 14.2 `transonic_thermo.py`

Suggested functions:

```python
def surface_density(Mdot, R, u):
    ...

def vertical_state(Sigma, T, R, potential, params):
    # Return H, rho, Pgas, Prad, Ptot, Pi, e, tau.
    ...

def integrated_stress(state, alpha, mu_stress, stress_factor):
    ...

def radiative_cooling(state, params):
    ...
```

Reuse existing thermodynamic functions where possible.

## 14.3 `transonic_local.py`

Suggested functions:

```python
def algebraic_state(logR, logu, logT, lambda0, params):
    # Return Sigma, H, rho, P, Pi, W, l, Omega, etc.
    ...

def state_partials(logR, y, lambda0, params):
    # Return explicit and state partials of Pi, rho, e, Omega.
    ...

def differential_residual(logR, y, g, lambda0, params):
    # Return [radial_momentum_residual, energy_residual].
    ...

def differential_matrix(logR, y, lambda0, params):
    # Return A, c such that F(g) = A @ g + c.
    ...

def sonic_diagnostics(logR, y, lambda0, params):
    # Return signed D, compatibility N, smin/smax, null vectors.
    ...
```

## 14.4 `transonic_collocation.py`

Suggested classes:

```python
@dataclass(frozen=True)
class TransonicSlimParams:
    M2_g: float
    Mdot_g_s: float
    alpha: float
    mu_stress: float = 0.0
    stress_factor: float = 1.0
    R_out_rg: float = 1000.0
    n_nodes: int = 80
    ...

@dataclass(frozen=True)
class TransonicSlimProfile:
    R: np.ndarray
    u: np.ndarray
    T: np.ndarray
    Sigma: np.ndarray
    H: np.ndarray
    rho: np.ndarray
    Omega: np.ndarray
    l: np.ndarray
    W: np.ndarray
    Q_visc: np.ndarray
    Q_rad: np.ndarray
    Q_adv: np.ndarray
    xi_eff: np.ndarray
    sonic_radius: float
    l0: float
    ...

@dataclass(frozen=True)
class TransonicSolveResult:
    profile: TransonicSlimProfile | None
    converged: bool
    cost: float
    max_residual: float
    message: str
    ...
```

Functions:

```python
def unpack_state(z, params):
    ...

def collocation_residual(z, params):
    ...

def initial_guess_from_reduced_solver(reduced_profile, params):
    ...

def solve_transonic_outer_branch(params, initial_guess):
    ...
```

## 14.5 `transonic_continuation.py`

Functions:

```python
def remap_profile_to_new_sonic_grid(profile, new_params):
    ...

def continue_in_mdot(base_params, mdot_values):
    ...

def adaptive_continue_in_log_mdot(...):
    ...
```

---

# 15. Required Unit and Regression Tests

Add:

```text
tests/test_transonic_potential.py
tests/test_transonic_local.py
tests/test_transonic_collocation.py
tests/test_transonic_continuation.py
```

## Test A: pseudo-Newtonian geometry

Verify:

```text
R_ISCO = 6 GM/c^2
Omega_K analytic derivative matches finite differences
l_K has an extremum at the ISCO
```

## Test B: vertical closure

Verify:

```text
P/rho approximately equals Omega_K^2 H^2
Sigma = 2 rho H
all thermodynamic quantities are positive
```

## Test C: differential linearity

For random physical states and random gradients:

```text
F(g) ~= A @ g + c
```

## Test D: entropy consistency

Retain the existing first-law and log-gradient cross-checks.

## Test E: reduced thin limit

At:

```text
Mdot/Mdot_Edd = 1e-3
```

outside approximately `10 r_g`, compare the transonic result to the repaired reduced solver:

```text
Sigma agreement < 3 percent
T agreement < 3 percent
Omega/Omega_K - 1 < 1 percent
Q_adv/Q_visc << 1
```

## Test F: sonic regularity

Require:

```text
abs(D) < 1e-6
abs(N) < 1e-6
smin/smax < 1e-6
```

after appropriate residual scaling.

## Test G: collocation residual

Require:

```text
max interval residual < 1e-6--1e-5
global energy L1 < 1e-3
```

## Test H: one sonic crossing

The formal critical determinant should cross zero once on the physical branch.

## Test I: grid convergence

Compare:

```text
N = 40, 80, 160
```

Require convergence of:

```text
R_son
l0
integrated Q_adv/Q_visc
max H/R
luminosity
```

to better than a few percent.

## Test J: continuation reproducibility

Continue upward and downward in `Mdot`. The stationary branch should agree where it is single-valued.

---

# 16. Diagnostic Figures and Tables

Create:

```text
outputs/figures/transonic_profiles_mdot_scan.png
outputs/figures/transonic_sonic_regularization.png
outputs/figures/transonic_branch_summary.png
outputs/tables/transonic_branch_summary.md
outputs/tables/transonic_solver_audit.md
```

For each accretion rate plot:

```text
u/c and effective Mach number
Omega/Omega_K
H/R
Sigma
T
Q_rad/Q_visc
Q_adv/Q_visc
xi_eff
D and N near the sonic point
```

Summary table columns:

```text
Mdot/Mdot_Edd
converged
R_son/r_g
l0/(r_g c)
max H/R
min tau_es
integrated Qadv/Qvisc
Lrad/Ledd
max collocation residual
energy L1
number of sonic crossings
```

---

# 17. Acceptance Criteria for Milestone T1

Milestone T1 passes if:

```text
1. A regular sonic eigenvalue solution is found at Mdot/Mdot_Edd = 1e-3.
2. The outer solution agrees with the repaired reduced thin disk.
3. D = N = 0 at a freely determined sonic radius.
4. The result converges with radial resolution.
5. The solution can be continued at least through Mdot/Mdot_Edd = 10.
6. No clipping of H/R, Q_adv, xi_eff, or Omega is used.
7. Global residuals remain below 1e-3--1e-2.
```

Milestone T1 is especially successful if continuation reaches:

```text
Mdot/Mdot_Edd ~= 94
```

with a regular no-wind solution.

---

# 18. Interpreting the QPE-Target Result

At the QPE target, classify the outcome carefully.

## Outcome A: regular and moderately thick

If:

```text
converged
H/R <~ 1
tau_es >> 1
single sonic crossing
energy residual small
```

then proceed to the physical minidisk radius and later add wind.

## Outcome B: regular but H/R > 1

Then:

```text
the mathematical transonic solution exists,
but the vertically integrated slim-disk approximation is outside its validity.
```

This is not a clean rejection of the QPE model. It means wind and/or a more complete vertical treatment is required.

## Outcome C: no regular no-wind solution only at very high Mdot

If the solver is validated at lower rates but fails near `Mdot ~= 94 Mdot_Edd`, proceed next to a wind-regulated transonic solver.

## Outcome D: no regular solution at low or moderate Mdot

Treat this as a numerical or equation-implementation failure, because classical slim-disk solutions should exist in that regime.

---

# 19. What Not to Add in This Sprint

Do not add:

```text
stream source
tidal torque
wind
MAD physics
time-dependent limit cycle
synthetic X-ray emission
```

until the isolated transonic solver passes.

Do not tune `xi`.

`xi_eff` must remain a diagnostic:

```math
\xi_{\rm eff}
=
-\frac{R\rho}{P}
T\frac{ds}{dR}.
```

---

# 20. Literature to Follow

Use these as primary references.

1. Abramowicz et al. 1988, **Slim Accretion Disks**  
   ADS: https://ui.adsabs.harvard.edu/abs/1988ApJ...332..646A/abstract  
   PDF: https://dydaktyka.fizyka.umk.pl/Pliki/Abramowicz.pdf

2. Sadowski 2009, **Slim Disks Around Kerr Black Holes Revisited**  
   arXiv: https://arxiv.org/abs/0906.0355

3. Sadowski 2011, **Slim Accretion Disks Around Black Holes**  
   arXiv: https://arxiv.org/abs/1108.0396

4. Sadowski et al. 2011, **Relativistic Slim Disks with Vertical Structure**  
   arXiv: https://arxiv.org/abs/1006.4309

The key numerical lessons to preserve are:

```text
- the sonic point is a critical point;
- numerator and denominator regularity conditions must vanish together;
- the swallowed angular momentum is an eigenvalue;
- the sonic location is not known in advance;
- relaxation/free-boundary methods are more robust than naive shooting;
- stationary solutions should be continued gradually in Mdot.
```

---

# 21. Compact Codex Prompt

```text
Implement Milestone T1 of an isolated transonic slim-disk solver.

Scope:
- Schwarzschild IMBH
- Paczynski-Wiita potential
- stationary and axisymmetric
- no stream, tide, or wind
- one-temperature gas+radiation EOS
- electron-scattering diffusion cooling
- algebraic alpha stress W = C_alpha alpha 2H Pgas^mu Ptot^(1-mu)
- baseline mu=0

Use u=-v_R>0 and Mdot=2 pi R Sigma u.
Eliminate Sigma with continuity.
Use y=[ln u, ln T], x=ln R.
Obtain Omega algebraically from:
    Mdot (l-l0) = 2 pi R^2 W
where l0 is an eigenvalue.

Implement radial momentum:
    u du/dR = R(Omega^2-OmegaK^2) - (1/Sigma)dPi/dR

Implement energy:
    Qvisc = Qrad + Qadv
    Qvisc = -W dOmega/dlnR
    Qadv = -(Sigma u/R) T ds/dlnR
    T ds/dlnR = de/dlnR - P/rho^2 d rho/dlnR

Construct the local differential system F(g)=A g+c for
g=[dlnu/dlnR,dlnT/dlnR] using numerical thermodynamic partials.

At the sonic point impose:
    D = normalized det(A) = 0
    N = left-null(A) dot c = 0

Use a free-boundary logarithmic grid:
    lnR = lnRson + xi(lnRout-lnRson)

Unknowns:
    ln u_i, ln T_i, ln Rson, l0/(rg c)

Residuals:
    2 midpoint collocation equations per interval
    2 outer thin-disk matching conditions
    D=0 and N=0 at the sonic node

Solve with scipy.optimize.least_squares and jac_sparsity.
Start at Mdot/Mdot_Edd=1e-3 and continue adaptively to 10, then toward 94.
Keep xi_eff diagnostic only.
Do not add stream/tide/wind in this sprint.

Add unit tests for:
    PW potential
    vertical closure
    F(g)=A g+c linearity
    entropy consistency
    low-Mdot thin-disk agreement
    sonic regularity
    grid convergence
    one sonic crossing
    global energy conservation
```
