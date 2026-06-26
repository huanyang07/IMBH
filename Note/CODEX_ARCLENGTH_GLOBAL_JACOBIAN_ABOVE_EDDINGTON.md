# Codex Next-Step Brief: Robust Pseudo-Arclength and Global-Jacobian Continuation Above Eddington

Repository and reviewed commit:

```text
https://github.com/huanyang07/IMBH
commit: 49e5d16
```

Files reviewed first:

```text
Note/CODEX_TRANSONIC_SOLVER_HARDENING_NEXT_STEP.md
outputs/tables/transonic_n64_arclength_adaptive.md
outputs/figures/transonic_n64_arclength_adaptive.png
scripts/run_transonic_n64_arclength_adaptive.py
src/imri_qpe/layer3_minidisk_1d/transonic_continuation.py
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
```

## Executive assessment

The current continuation run is substantially positive:

```text
- N=64 continuation reaches Mdot/Mdot_Edd ~= 0.996;
- every accepted checkpoint is classified physical/equation/sonic valid;
- H/R remains only ~= 0.16 at the last checkpoint;
- integrated Qadv/Qvisc is ~= 0.094;
- predicted and corrected Mdot values remain close;
- arclength residuals are tiny.
```

There is no strong evidence for a physical fold or disappearance of the branch near
one Eddington. The main symptom is numerical:

```text
- nfev rises to 1100 near Mdot/Mdot_Edd ~= 1;
- the last solve reaches max_nfev;
- the physical residual stalls near 1e-4--2e-4;
- the dominant residual alternates between outer Omega and sonic D;
- the arclength equation is solved orders of magnitude more accurately than
  the physical equations.
```

Therefore the next task is not to add wind or change the astrophysics. It is to
turn the present least-squares pseudo-arclength method into a properly scaled
bordered Newton continuation method with a more accurate global Jacobian.

The most important changes are:

```text
1. Reduce the fixed-Mdot collocation system to a square system by using only
   two independent sonic equations.
2. Compute the continuation tangent from the augmented Jacobian, not only from
   a secant between profiles.
3. Replace the current uniform state scaling by a blockwise continuation metric.
4. Audit and improve the nested finite-difference Jacobian.
5. Use a sparse bordered Newton corrector with line search.
6. Adapt step size using predictor error and branch curvature, not nfev alone.
7. Polish every branch checkpoint before it is used as a tangent anchor.
```

Do not add stream feeding, tides, wind, or time dependence in this sprint.

---

# 1. What the current output says

The branch reaches:

```text
Mdot/Mdot_Edd = 0.9963
max H/R       = 0.160
int Qadv/Qvisc= 0.094
max residual  = 2.05e-4
nfev          = 1100
```

The branch is still moderately thin and only mildly advective. A physical
breakdown of the no-wind slim solution near this point would be unexpected.
The predictor is also working: the predicted and corrected accretion rates
remain close.

The bottleneck is the corrector:

```text
- physical residuals stop improving efficiently;
- sonic D becomes dominant again near the final point;
- outer Omega had been the dominant block over much of the preceding branch;
- the optimizer spends hundreds of evaluations for changes of only a few
  percent in Mdot.
```

This is characteristic of:

```text
- a poorly conditioned/redundant residual system;
- an inaccurate Jacobian;
- unsuitable variable/arclength scaling;
- or an over-constrained least-squares corrector.
```

---

# 2. Make the fixed-\(\dot M\) system square

## 2.1 Current equation count

For `N` nodes, the state vector has:

```text
2N nodal variables
+ log R_son
+ lambda0
= 2N + 2 unknowns.
```

The current collocation residual contains:

```text
2(N-1) interval equations
+ 2 outer-boundary equations
+ 3 sonic equations [D, C1, C2]
= 2N + 3 residuals.
```

Thus the fixed-\(\dot M\) problem is overdetermined by one equation.

The third sonic equation is mathematically redundant at an exact rank-one
critical point, but it is not exactly redundant away from the root. Keeping all
three in least squares creates a residual floor and worsens conditioning.

## 2.2 Use two independent sonic equations in the solver

Use:

```text
D = 0
C = 0
```

where `C` is one independent compatibility equation.

For the scaled matrix

```math
A=
\begin{pmatrix}
a&b\\
c&d
\end{pmatrix},
\qquad
q=
\begin{pmatrix}
e\\f
\end{pmatrix},
```

the two algebraic compatibility candidates are:

```math
C_1=de-bf,
```

```math
C_2=af-ce.
```

At `det(A)=0`, only one is independent.

### Recommended robust pivot

At each accepted branch point choose the compatibility equation using the
better-conditioned left-null construction:

```text
if a^2 + c^2 >= b^2 + d^2:
    use C = C2
else:
    use C = C1
```

Freeze this pivot for the entire Newton corrector step. Do not switch the pivot
during one corrector iteration.

Alternative:

```text
use a smooth weighted combination of C1 and C2
```

but a frozen pivot is simpler and easier to audit.

Keep the unused compatibility equation as a diagnostic. A valid solution must
also make it small.

## 2.3 New equation count

The fixed-\(\dot M\) system becomes:

```text
2(N-1) interval
+ 2 outer
+ 2 sonic
= 2N + 2 equations
```

for `2N+2` unknowns.

The pseudo-arclength system then has:

```text
unknowns: [z, mu]                 -> 2N+3
equations: F(z,mu)=0 + arc = 0   -> 2N+3
```

This is the standard square bordered continuation problem.

---

# 3. Replace the secant-only tangent by a Jacobian tangent

The current code constructs the tangent from two remapped branch points:

```math
t_{\rm sec}
\propto
w_k-w_{k-1}.
```

This is acceptable for the first step, but it becomes noisy when:

```text
- accepted solutions have residuals near 1e-4;
- the mesh moves with R_son;
- branch curvature increases;
- successive Mdot spacings vary.
```

## 3.1 Tangent equation

Let:

```math
F(z,\mu)=0,
\qquad
\mu=\ln(\dot M/\dot M_{\rm Edd}).
```

At an accepted point, compute the tangent from:

```math
F_z t_z + F_\mu t_\mu = 0.
```

Use the previous tangent to fix normalization and orientation:

```math
\begin{bmatrix}
F_z & F_\mu\\
t_{\rm old}^T M_z & t_{\mu,\rm old}M_\mu
\end{bmatrix}
\begin{pmatrix}
t_z\\t_\mu
\end{pmatrix}
=
\begin{pmatrix}
0\\1
\end{pmatrix}.
```

Then normalize:

```math
t^T M t=1
```

and enforce:

```math
t\cdot_M t_{\rm old}>0.
```

Here `M` is the continuation metric defined below.

Use the secant tangent only:

```text
- to initialize the first Jacobian tangent;
- or as a fallback if the bordered tangent solve fails.
```

## 3.2 Why this matters

A Jacobian tangent:

```text
- remains accurate near branch curvature;
- recognizes an actual fold;
- reduces predictor error;
- is less contaminated by residual noise;
- removes the need for MIN_TANGENT_RATIO_GAP and nonadjacent checkpoint pairs.
```

Delete the heuristic that searches backward for a minimum Mdot-ratio gap once
the Jacobian tangent is working. Always use the immediately preceding accepted
point plus the stored tangent.

---

# 4. Use a blockwise continuation metric

The current scaling is approximately:

```python
state_scale = sqrt(number_of_state_variables)
scales = [state_scale for every state variable] + [1 for mu]
```

This suppresses all profile and eigenvalue changes by the same large factor,
while leaving `mu` unscaled. The tangent can therefore be dominated by the
accretion-rate coordinate and the method behaves close to ordinary
fixed-\(\dot M\) continuation.

## 4.1 Define a physically balanced metric

Use:

```math
\|\delta w\|_M^2
=
\frac{1}{N}\sum_i
\left(\frac{\delta\ln u_i}{s_u}\right)^2
+
\frac{1}{N}\sum_i
\left(\frac{\delta\ln T_i}{s_T}\right)^2
+
\left(\frac{\delta\ln R_{\rm son}}{s_R}\right)^2
+
\left(\frac{\delta\lambda_0}{s_\lambda}\right)^2
+
\left(\frac{\delta\mu}{s_\mu}\right)^2.
```

Initial floors:

```text
s_u      = max(RMS recent Delta logu, 0.02)
s_T      = max(RMS recent Delta logT, 0.01)
s_R      = max(abs recent Delta logRson, 0.01)
s_lambda = max(abs recent Delta lambda0, 0.01)
s_mu     = max(abs recent Delta mu, 0.02)
```

Smooth these scales over several accepted steps so they do not jump.

The equivalent scale vector is:

```text
sqrt(N)*s_u      for every logu node
sqrt(N)*s_T      for every logT node
s_R              for logRson
s_lambda         for lambda0
s_mu             for mu
```

## 4.2 Audit tangent composition

For every accepted point report:

```text
fraction of tangent norm in logu block
fraction in logT block
fraction in logRson
fraction in lambda0
fraction in mu
dmu/ds
```

If nearly all tangent norm is in `mu`, the continuation metric is not doing
useful pseudo-arclength continuation.

---

# 5. Use the normal-plane corrector

After predicting:

```math
w_p=w_k+\Delta s\,t_k,
```

use the arclength residual:

```math
g(w)=t_k^T M(w-w_p)=0.
```

This is equivalent to the current formula only when the scaling is exactly
consistent, but it is clearer and less error-prone.

The bordered corrector system is:

```math
G(w)=
\begin{pmatrix}
F(w)\\
g(w)
\end{pmatrix}
=0.
```

Its Jacobian is:

```math
J_G=
\begin{bmatrix}
F_z & F_\mu\\
t_{z,k}^TM_z & t_{\mu,k}M_\mu
\end{bmatrix}.
```

Do not give the arclength row an arbitrary weight of 10.

The current run has:

```text
arc residual ~ 1e-9
physical residual ~ 1e-4
```

so the arclength equation is being enforced much more strongly than the
physics.

Use:

```text
arclength_weight = 1
```

after row/column scaling, or choose it so that the arc Jacobian-row norm is
comparable to the median physical row norm.

---

# 6. Replace the least-squares corrector by a bordered Newton corrector

Once the physical system is square, use a direct sparse Newton step rather than
an overdetermined trust-region least-squares solve.

At each Newton iteration solve:

```math
J_G\,\delta w=-G.
```

For `N=64--256`, use sparse LU:

```python
from scipy.sparse.linalg import splu

delta = splu(J_G.tocsc()).solve(-G)
```

Then apply a line search to:

```math
\Phi(w)=\frac12\|F(w)\|_2^2+\frac12 g(w)^2.
```

Suggested line search:

```text
alpha = 1
while Phi(w + alpha*delta) >
      Phi(w) + c1*alpha*gradPhi_dot_delta:
    alpha *= 0.5
```

with:

```text
c1 = 1e-4
minimum alpha = 1e-6
```

Use a maximum of approximately:

```text
15--30 Newton iterations per continuation step
```

rather than hundreds of least-squares function evaluations.

Keep the existing least-squares path as a fallback and comparison test.

---

# 7. Audit the present global Jacobian

The current Jacobian is assembled from:

```text
- block-local finite differences in z with rel_step ~= 1e-6;
- a full finite-difference Mdot column with step ~= 1e-5;
- local differential matrices that themselves use numerical partials with
  partial_eps ~= 1e-5.
```

This is nested finite differencing.

The outer global step `1e-6` is smaller than the inner thermodynamic
finite-difference scale `1e-5`. The global Jacobian can therefore differentiate
numerical noise rather than the smooth residual.

This is a likely cause of the sharp nfev increase near Eddington.

## 7.1 Add directional-derivative tests

At branch points near:

```text
Mdot/Mdot_Edd = 0.3, 0.7, 0.95, 1.0
```

draw random normalized directions `v` and compare:

```math
Jv
```

against:

```math
\frac{G(w+hv)-G(w-hv)}{2h}.
```

Scan:

```text
h = 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6
```

Report:

```math
\epsilon_J(h)
=
\frac{\|Jv-D_hG\,v\|}
{\|Jv\|+\|D_hG\,v\|+\epsilon}.
```

Target:

```text
median epsilon_J < 1e-4
max epsilon_J < 1e-3
```

for the step used in production.

## 7.2 Compare block-local and dense Jacobians

For small grids:

```text
N = 12, 16
```

compare the assembled sparse Jacobian to a dense finite-difference Jacobian of
the complete augmented residual.

Report:

```text
max relative column error
Frobenius relative error
largest-error column and residual block
```

## 7.3 Increase the outer finite-difference step initially

Until the nested derivatives are removed, test:

```text
global state rel_step = 3e-5, 1e-4
Mdot-column step      = 3e-5, 1e-4
```

rather than `1e-6`.

Choose the step from the directional-derivative error curve, not by assumption.

## 7.4 Use a higher-order Mdot derivative

Replace the two-point central derivative of the \(\mu\) column by a five-point
stencil:

```math
F_\mu
\simeq
\frac{-F(\mu+2h)+8F(\mu+h)-8F(\mu-h)+F(\mu-2h)}
{12h}.
```

This column directly controls the branch tangent and deserves higher accuracy.

---

# 8. Remove nested finite differences

This is the medium-term Jacobian upgrade most likely to matter above
Eddington.

Choose one of the following.

## Option A: automatic differentiation of local blocks

Rewrite the pure numerical kernels for:

```text
interval residual
outer boundary residual
sonic residual
```

as side-effect-free functions and use JAX:

```python
jax.jacfwd
```

or:

```python
jax.jacrev
```

to obtain exact block derivatives.

Then assemble the sparse global Jacobian from those blocks.

For `N=64`, even a dense JAX Jacobian is small enough to use for validation.

## Option B: analytic local thermodynamic partials

Derive analytic derivatives of:

```text
H
rho
P
Pi
e
Omega
Qrad
```

with respect to:

```text
log R
log u
log T
lambda0
mu
```

Then build the interval and sonic Jacobian analytically.

## Option C: complex-step differentiation

This works only after removing:

```text
float(...) casts
nonsmooth max/abs branches
exception-driven residuals
SVD sign choices
```

Use complex step only for smooth local kernels.

Recommended practical order:

```text
1. directional audit of current Jacobian;
2. correct finite-difference step;
3. square bordered Newton;
4. JAX/analytic local blocks if needed.
```

---

# 9. Add Jacobian conditioning diagnostics

At every accepted point compute:

```text
smallest singular value of fixed-Mdot J_z
condition estimate of J_z
condition estimate of bordered J_G
```

For large sparse matrices use:

```python
scipy.sparse.linalg.svds
```

for the smallest singular values, or estimate conditioning from sparse LU.

Interpretation:

```text
- sigma_min(J_z) -> 0 but bordered J_G remains regular:
  genuine fold; pseudo-arclength is working.

- both J_z and bordered J_G become singular:
  tangent/constraint/scaling problem or a higher-codimension point.

- condition number grows while residual is dominated by one block:
  residual/Jacobian scaling problem.
```

Also store:

```text
angle between successive tangents
predictor-corrector displacement
```

---

# 10. Improve step-size control

The current step multiplier is based mostly on `nfev` and is capped at one
previous chord length.

Use a true arclength step `ds`, independent of the previous chord.

## 10.1 Predictor error

After correction define:

```math
e_p
=
\frac{\|w_{k+1}-w_p\|_M}{\Delta s}.
```

## 10.2 Tangent curvature

After computing the new tangent:

```math
\theta
=
\cos^{-1}(t_{k+1}^TMt_k),
```

```math
\kappa_s
\simeq
\frac{\|t_{k+1}-t_k\|_M}{\Delta s}.
```

## 10.3 Step update

Use:

```math
\Delta s_{\rm new}
=
\Delta s\,
{\rm clip}
\left[
\left(\frac{e_{\rm target}}{e_p+\epsilon}\right)^{1/2},
0.5,
1.5
\right].
```

Also enforce:

```math
\Delta s_{\rm new}
\le
\frac{\theta_{\rm target}}{\kappa_s+\epsilon}.
```

Suggested values:

```text
e_target     = 0.05--0.1
theta_target = 5--10 degrees
```

Additional rules:

```text
Newton <= 4 iterations and small predictor error -> increase ds
Newton 5--8 -> keep ds
Newton > 8 or line-search difficulty -> halve ds
corrector failure -> quarter ds and retry
```

A point that reaches `max_nfev` but passes the loose physical tolerance may be
stored, but it must not be used as a tangent anchor until it is polished.

---

# 11. Polish every branch checkpoint

Current accepted branch points have physical residuals around `1e-4`.

This is adequate for a smoke plot but marginal for computing a precise tangent.

After each accepted pseudo-arclength step:

```text
1. hold the corrected Mdot fixed;
2. solve the square fixed-Mdot collocation system to tighter tolerance;
3. require max residual < 1e-6, preferably 1e-7;
4. recompute the Jacobian tangent from the polished state;
5. save the polished state and tangent in the checkpoint.
```

Define two acceptance levels:

```text
usable_point:
    physical and residual < 3e-4

anchor_point:
    optimizer/Newton converged
    residual < 1e-6
    Jacobian audit passed
```

Only `anchor_point` may define the next continuation tangent.

If the fixed-\(\dot M\) polish fails near a true fold, polish using the bordered
arclength system with a smaller step.

---

# 12. Replace the frontier-pair heuristic

The script currently selects a prior checkpoint using a minimum difference in
Mdot ratio. This was introduced to avoid tiny secants, but it can skip
arclength-adjacent states and distort the tangent when curvature grows.

Once Jacobian tangents are stored:

```text
- use the last accepted anchor and its stored tangent;
- orient the new tangent continuously;
- remove MIN_TANGENT_RATIO_GAP;
- keep secant logic only as restart fallback.
```

Checkpoint files should store:

```text
z
mu
tangent
continuation scales
ds
Jacobian condition estimate
residual audit
```

---

# 13. Hybrid fixed-parameter / pseudo-arclength continuation

The current branch remains monotonic through approximately one Eddington. There
is no clear fold in the table.

Use a hybrid strategy:

```text
if abs(t_mu) is comfortably nonzero and J_z is well conditioned:
    use fixed-mu Newton continuation with a Jacobian predictor
else:
    switch to pseudo-arclength
```

Suggested trigger:

```text
pseudo-arclength if:
    abs(t_mu) < 0.2
    or sigma_min(J_z) falls below threshold
    or fixed-mu corrector fails twice
```

This avoids paying the cost of the arclength equation where ordinary
continuation is well conditioned, while retaining fold robustness.

---

# 14. Residual and Jacobian equilibration

Before each Newton solve, equilibrate the bordered Jacobian.

Use row scales from residual blocks and column scales from the continuation
metric or Jacobian norms.

A simple Ruiz iteration is adequate:

```text
repeat 2--5 times:
    scale rows to comparable 2-norms
    scale columns to comparable 2-norms
```

Solve the equilibrated system and transform the step back.

This is preferable to manually setting:

```text
arclength_weight = 10
```

or repeatedly tuning sonic/outer weights.

Continue to report unweighted physical residuals.

---

# 15. Recommended immediate continuation experiment

After implementing the square sonic system, bordered Newton, and tangent
scaling:

```text
1. Load polished anchor points near 0.90 and 0.95.
2. Recompute Jacobian tangents.
3. Set ds to one quarter of the previous chord.
4. Continue to Mdot/Mdot_Edd = 1.2.
5. Then continue to 1.5, 2, 3, 5, and 10 adaptively.
```

Use parameter bounds comfortably outside the target:

```text
Mdot ratio bounds = (0.1, 20)
R_son upper bound = large enough for the target, but flag active bounds
```

Do not jump directly from one to ten.

At each new accretion-rate decade perform:

```text
N=64 vs N=96 comparison
Jacobian directional audit
outer-boundary audit
```

---

# 16. Production diagnostics

Add these columns to the continuation table:

```text
Newton iterations
line-search reductions
predictor error
tangent angle
ds
t_mu
sigma_min(J_z)
condition estimate J_z
condition estimate bordered J
Jacobian directional error
fixed-Mdot polish residual
compatibility diagnostic not used in solver
```

Plot:

```text
Mdot vs arclength s
R_son vs s
lambda0 vs s
H/R max vs s
Qadv/Qvisc vs s
t_mu vs s
sigma_min(J_z) vs s
predictor error vs s
Newton iterations vs s
```

These plots will distinguish a real fold from a numerical stall.

---

# 17. Acceptance criteria above Eddington

A branch point above one Eddington is accepted scientifically only if:

```text
- square physical residual system converges;
- max physical residual < 1e-6 for anchor points;
- unused sonic compatibility residual is also small;
- no active state/eigenvalue/Mdot bounds;
- one physical sonic point;
- bordered Jacobian remains nonsingular;
- predictor-corrector error is controlled;
- N=64 and N=96 profiles agree to a few percent;
- the result is insensitive to modest changes in outer radius;
- no clipping of H/R, Qadv, Omega, or xi_eff.
```

If the branch fails only when:

```text
H/R approaches unity
or optical depth becomes marginal
```

then the next physical extension is wind/vertical-structure physics.

If it fails while:

```text
H/R << 1
and the bordered Jacobian is well conditioned
```

then the failure is still numerical or closure-related.

---

# 18. Concrete code changes

## `transonic_collocation.py`

Add:

```python
def sonic_residual_pair(..., pivot):
    # Return [D, C_selected].
    ...

def unused_sonic_compatibility(..., pivot):
    # Diagnostic only.
    ...

def square_collocation_residual(z, params, pivot):
    ...

def square_collocation_jacobian(z, params, pivot):
    ...

def jacobian_directional_error(...):
    ...
```

Change residual size and sparsity pattern to use two sonic equations.

## `transonic_continuation.py`

Add:

```python
@dataclass(frozen=True)
class ContinuationMetric:
    s_logu: float
    s_logT: float
    s_logRson: float
    s_lambda0: float
    s_mu: float

@dataclass(frozen=True)
class ContinuationAnchor:
    z: np.ndarray
    mu: float
    tangent: np.ndarray
    ds: float
    metric: ContinuationMetric
    ...
```

Add:

```python
def bordered_tangent(...):
    ...

def bordered_newton_corrector(...):
    ...

def polish_anchor(...):
    ...

def adapt_arclength_step(...):
    ...

def equilibrate_bordered_jacobian(...):
    ...
```

Deprecate:

```text
MIN_TANGENT_RATIO_GAP
secant-only tangent as the production default
nfev-only step control
arclength_weight=10
```

## `run_transonic_n64_arclength_adaptive.py`

Change:

```text
TARGET_RATIO = 1
```

only after the new solver passes to:

```text
TARGET_RATIO = 3
```

then later:

```text
TARGET_RATIO = 10
```

Use:

```text
N64 polished checkpoints
N96 spot checks
anchor checkpoint format with tangent and metric
```

---

# 19. Compact Codex prompt

```text
Upgrade the transonic continuation above Mdot/Mdot_Edd~1.

The current N=64 branch reaches 0.996 with H/R~0.16 and Qadv/Qvisc~0.094, so
there is no clear physical fold. The bottleneck is the corrector/Jacobian:
nfev reaches 1100, residuals stall around 1e-4, and the arc equation is solved
much more accurately than the physical equations.

Implement the following:

1. Make the fixed-Mdot collocation system square. Use only two independent
   sonic equations: D=0 and one pivoted compatibility equation C=0. Keep the
   other compatibility equation as a diagnostic.

2. Form the standard square pseudo-arclength system:
       F(z,mu)=0
       t^T M (w-w_pred)=0

3. Compute the tangent from the augmented Jacobian, not only from a secant:
       [F_z F_mu; t_old^T M] t = [0;1]
   normalize in a blockwise continuation metric and preserve orientation.

4. Replace uniform sqrt(nstate) scaling with blockwise RMS scales for logu,
   logT, logRson, lambda0, and mu.

5. Replace least_squares as the main corrector with a sparse bordered Newton
   solve plus line search. Use sparse LU for N<=256.

6. Set arclength weight to unity after scaling/equilibration. The current
   weight=10 over-enforces the arc equation.

7. Audit the global Jacobian with directional derivatives. The current global
   FD step 1e-6 is smaller than the nested local partial_eps~1e-5. Scan FD
   steps and initially use ~3e-5--1e-4. Use a 5-point stencil for F_mu.

8. Medium term: remove nested finite differences with JAX or analytic local
   block derivatives.

9. Polish every accepted point to max residual <1e-6 before it becomes a
   tangent anchor. Save tangent, metric, ds, and Jacobian conditioning in the
   checkpoint.

10. Adapt ds using predictor error and tangent angle, not nfev alone. Remove
    MIN_TANGENT_RATIO_GAP once Jacobian tangents are available.

11. Track sigma_min(J_z) and conditioning of the bordered Jacobian. Use
    pseudo-arclength only when fixed-mu continuation becomes ill conditioned.

12. Restart from polished anchors near 0.9 and 0.95, use a quarter-size first
    step, continue to 1.2, then 1.5, 2, 3, 5, and 10. Perform N=96 spot checks.
```

---

# 20. Interpretation rule

The current run should be summarized as:

```text
Pseudo-arclength continuation has successfully approached one Eddington.
The predictor is adequate, but the overdetermined least-squares corrector and
nested finite-difference Jacobian become inefficient near one Eddington.
The next step is a square, scaled, bordered Newton continuation—not new disk
physics.
```
