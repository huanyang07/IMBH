# Codex Next-Step Brief: Replace the Failed Taylor/L'Hopital Sonic Patch with an Offset Sonic Flow-Map Boundary

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
latest main branch
```

Primary files reviewed:

```text
Note/GPT_TAYLOR_LHOPITAL_NEXT_STEP.md
Note/GPT_SONIC_REGULARITY_HANDOFF.md
outputs/tables/transonic_two_domain_sonic_taylor_summary.md
outputs/tables/transonic_sonic_derivative_roots.md
outputs/tables/transonic_sonic_lhopital_audit.md
outputs/tables/transonic_two_domain_sonic_taylor_patch.md
scripts/run_transonic_sonic_derivative_root_scan.py
scripts/run_transonic_two_domain_sonic_taylor_patch.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
tests/test_transonic_local.py
```

## Executive diagnosis

The Taylor/L'Hopital patch failed for a useful reason.

The current L'Hopital derivative scan is being applied at an approximate sonic state taken from the existing two-domain/dynamic-patch solution. At that state, the regularity condition finds two mathematically consistent derivative roots:

```text
g ~ (-183.5,  73.6)
g ~ ( 161.3, -96.1)
```

These are far from both:

```text
source first-interval slope ~ (18.3, 17.8)
dynamic-patch slope         ~ (19.7, -27.9)
```

This does not necessarily mean the L'Hopital mathematics is wrong. It means the current strategy is wrong for the numerical problem being solved.

The dynamic-patch slope is a finite-buffer average over \(\Delta_s \sim 0.02\), not the true derivative at the singular point. If the true regular derivative is of order \(10^2\), then a Taylor patch over \(\Delta_s=0.02\) moves the solution by order unity:

```text
|g| * Delta_s ~ 2--4
```

A Taylor expansion over such a long patch is not valid.

Therefore the next strategy should not be another long finite Taylor patch. It should be:

```text
solve the singular sonic point algebraically,
start the ODE a tiny distance outside the sonic point,
integrate or collocate the regular ODE to the buffer,
and match the ordinary inner domain there.
```

In other words:

```text
replace "Taylor patch over Delta_s"
with
"offset sonic flow-map boundary."
```

---

# 1. What to stop doing

Do not keep trying these as the main production method:

```text
1. Taylor1/Taylor2 over Delta_s = 0.02 using the L'Hopital derivative.
2. Ordinary midpoint micro-domain collocation including the exact sonic endpoint.
3. Soft C1/C2/K compatibility fallback without a genuine regular derivative.
4. Selecting the physical branch by closeness of g_s to the dynamic finite-buffer slope.
```

The first is too long a Taylor step, the second is too singular-sensitive, the third stalls near \(3\times10^{-4}\), and the fourth compares a true derivative to a finite-difference average.

---

# 2. Correct local mathematical object

Let:

```math
x=\ln R,\qquad y=(\ln u,\ln T),
```

and write the local ODE system as:

```math
F(x,y,g;\lambda_0)=A(x,y;\lambda_0)g+c(x,y;\lambda_0)=0,
```

where:

```math
g=\frac{dy}{dx}.
```

Away from the sonic point,

```math
g=f(x,y;\lambda_0)=-A^{-1}c.
```

At the sonic point,

```math
\det A_s=0.
```

A regular solution requires the usual compatibility condition. For numerical robustness, use the smooth algebraic compatibility from the adjugate rather than an SVD sign-sensitive vector whenever possible.

For a scaled \(2\times2\) system,

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

use:

```math
D=ad-bc,
```

and compatibility equations:

```math
C_1=de-bf,
```

```math
C_2=af-ce.
```

At a rank-one regular sonic point:

```math
D=0,
\qquad
C_1=0,
\qquad
C_2=0.
```

In practice use:

```text
D = 0
one pivoted C = 0
```

as solve rows, while auditing the unused compatibility and K.

---

# 3. Offset sonic flow-map strategy

## 3.1 Main idea

Do not put the first collocation interval directly at the singular point.

Instead introduce:

```text
epsilon0  << epsilon_buf
```

with for example:

```text
epsilon0    = 1e-6 to 1e-5 in log radius
epsilon_buf = 0.005, 0.01, 0.015, 0.02
```

The exact sonic point is at:

```math
x_s=\ln R_s.
```

The regular inner collocation domain starts at:

```math
x_0=x_s+\epsilon_{\rm buf}.
```

Between \(x_s\) and \(x_0\), compute a **flow map**:

```math
y_0=\Phi_{\epsilon_{\rm buf}}(x_s,y_s,\lambda_0,{\rm branch}).
```

Then the global BVP does not contain a singular collocation interval.

---

# 4. Constructing the flow map

## 4.1 Sonic derivative roots

At a trial sonic state \((x_s,y_s,\lambda_0)\):

1. Build \(A_s,c_s\).
2. Enforce \(D=0\) and compatibility.
3. Compute a particular solution of:

```math
A_s g=-c_s.
```

4. Compute the right null vector \(r\) of \(A_s\).
5. Parameterize:

```math
g(a)=g_p+a r.
```

6. Find candidate derivative branches by the regularity condition.

For the next implementation, keep the existing \(l^T B(g)=0\) root finder as a branch generator, but do not assume the derivative root closest to the old finite-buffer slope is the physical one.

## 4.2 Start slightly outside the sonic point

For a candidate derivative \(g_s\), start at:

```math
x_{\epsilon}=x_s+\epsilon_0.
```

Use:

```math
y_\epsilon
=
y_s+\epsilon_0 g_s
```

as the first implementation.

If needed add the second-order term later.

Use very small \(\epsilon_0\), because the derivatives can be large:

```text
if |g_s| ~ 200 and epsilon0 = 1e-5,
then |Delta y| ~ 2e-3, safe.
```

Do not Taylor-expand all the way to \(\Delta_s=0.02\).

## 4.3 Integrate the regular ODE

Integrate:

```math
\frac{dy}{dx}=f(x,y;\lambda_0)=-A^{-1}c
```

from:

```text
x_s + epsilon0
```

to:

```text
x_s + epsilon_buf.
```

Use an implicit or stiff-aware method first:

```text
scipy.integrate.solve_ivp(..., method="Radau")
```

Also test:

```text
DOP853
BDF
```

for comparison.

The flow-map output is:

```math
y_{\rm buf}=\Phi_{\epsilon_{\rm buf}}(x_s,y_s,\lambda_0,{\rm branch}).
```

This replaces:

```math
y_{\rm buf}=y_s+\Delta_s g_s
```

and replaces the failed long Taylor patch.

---

# 5. Global BVP with offset boundary

## 5.1 Unknowns

Use the usual two-domain unknowns, but replace the sonic endpoint collocation patch with:

```text
x_s = logR_son
y_s = (logu_s, logT_s)
lambda0
branch label, discrete
regular inner nodes starting at x0 = x_s + epsilon_buf
outer-domain nodes
```

The derivative \(g_s\) can either be:

```text
computed from branch root inside the flow map
```

or included as an auxiliary unknown during debugging.

## 5.2 Residual rows

At the sonic point solve:

```math
D(x_s,y_s,\lambda_0)=0,
```

```math
C_{\rm pivot}(x_s,y_s,\lambda_0)=0.
```

Then match the first regular inner node to the sonic flow map:

```math
y_{\rm inner,0}
-
\Phi_{\epsilon_{\rm buf}}(x_s,y_s,\lambda_0,{\rm branch})
=0.
```

Then apply ordinary nonsingular collocation from:

```math
x_s+\epsilon_{\rm buf}
```

to \(R_{\rm match}\).

Continue using the already successful two-domain outer extension.

## 5.3 Diagnostics

Always audit:

```text
D
C1
C2
K
smin/smax
chosen derivative branch
g_s
epsilon0
epsilon_buf
flow-map integration error
```

---

# 6. Branch selection

The current L'Hopital root scan at the old sonic state found two far roots. That is not enough to select the physical branch.

Use a **flow-map matching criterion** instead.

For each candidate derivative branch:

1. Compute \(y_{\rm buf}=\Phi_{\epsilon_{\rm buf}}\).
2. Compare it with the first node of the dynamic-patch or two-domain root at the same buffer radius.
3. Reject branches that give unphysical behavior before the buffer:
   ```text
   non-finite y
   negative/invalid Sigma
   H/R blow-up
   Qvisc < 0
   wrong sign of u
   discontinuous entropy
   ```
4. Use the branch that minimizes the global BVP residual after a short local polish.

This is much more reliable than comparing \(g_s\) directly to a finite-buffer slope.

---

# 7. Local sonic-fit diagnostic before the full BVP

Before building the full offset BVP, run a local diagnostic.

At the current N65 source:

```text
x_s, y_s, lambda0 fixed initially
```

For each derivative branch:

```text
branch A
branch B
```

integrate the flow map to:

```text
epsilon_buf = 0.005, 0.01, 0.015, 0.02
```

and compare to the dynamic-patch buffer state.

Output:

```text
branch
epsilon_buf
g_u
g_T
flow success
distance to dynamic buffer
distance to source first regular node
H/R max in local integration
Qvisc sign
entropy monotonicity
```

Create:

```text
outputs/tables/transonic_sonic_flowmap_branch_audit.md
```

This diagnostic will answer whether the far derivative branches are actually incompatible with the accepted outer solution or whether they only look incompatible because the previous comparison used the derivative itself.

---

# 8. If neither derivative branch matches

If both regular derivative branches miss the dynamic-patch buffer by a large amount, do not force them.

That means the old \(x_s,y_s,\lambda_0\) is not the true critical point of the desired branch.

Then solve a local **sonic-to-buffer shooting fit**.

Unknowns:

```text
x_s
logu_s
logT_s
lambda0
```

Discrete branch label:

```text
branch A or branch B
```

Residuals:

```text
D = 0
C_pivot = 0
Phi_epsilon_buf(x_s,y_s,lambda0,branch) - y_buffer_target = 0
```

That gives four equations for four continuous unknowns.

Use the dynamic-patch buffer state as `y_buffer_target`.

This local shooting fit should produce a better sonic state for seeding the full BVP.

Create:

```text
scripts/run_transonic_sonic_flowmap_fit.py
outputs/tables/transonic_sonic_flowmap_fit.md
```

---

# 9. Full offset-flow-map BVP validation

After the flow-map diagnostic and local fit pass, solve the full BVP.

Recommended baseline:

```text
Mdot/Mdot_Edd = 0.90277664
R_match = 6500 rg
R_far = 1e5 rg
N_outer = 54
N_regular_inner = 64
epsilon0 = 1e-5
epsilon_buf = 0.01
far closure = pressure-supported
```

Run both derivative branches if both are viable.

Acceptance for Nreg64:

```text
physical residual <= few e-6
D,C1,C2,K all small
Rson close to previous range, unless flow-map fit finds a controlled shift
lambda0 ~ 3.675--3.676
int_adv ~ 0.07--0.08
```

Then refine:

```text
N_regular_inner = 64, 80, 96, 112, 128
```

with the same \(\epsilon_{\rm buf}\).

Then scan:

```text
epsilon_buf = 0.02, 0.015, 0.01, 0.0075, 0.005
```

at Nreg64 and Nreg96.

The science root should converge as \(\epsilon_{\rm buf}\to0\).

---

# 10. Why this is better than Taylor/L'Hopital patching

The failed Taylor patch tried to represent the entire sonic-to-buffer region with a low-order polynomial:

```math
y_b \approx y_s+\Delta_s g_s+\frac12\Delta_s^2 h_s.
```

That is fragile when:

```text
|g_s| is large,
h_s is huge,
or the solution rapidly curves near the critical point.
```

The flow-map method instead uses the same ODE as the disk model on the interval
outside the singular point. The Taylor expansion is only used over a tiny
\(\epsilon_0\), where it is valid.

This directly addresses the observed failure:

```text
L'Hopital roots are far from old finite-buffer slopes,
so long Taylor extrapolation fails.
```

---

# 11. Optional fallback: Richardson-extrapolated dynamic patch

If the flow-map implementation is too difficult in one sprint, use the dynamic
patch as a controlled numerical regulator and extrapolate \(\Delta_s\to0\).

Run:

```text
Delta_s = 0.05, 0.03, 0.02, 0.015, 0.01, 0.0075
```

at fixed:

```text
Nreg64
Nreg80
```

Fit:

```math
Q(\Delta_s)=Q_0+a\Delta_s+b\Delta_s^2
```

for:

```text
Rson
lambda0
int_adv
physical residual
```

This is not as clean as the flow-map method, but it is more robust than the
failed Taylor patch. It can provide a useful reference solution and sanity
check.

Do not use this as the final production method unless the extrapolated values
are stable under changing the fit range.

---

# 12. Implementation details

## 12.1 Add ODE flow functions

In `transonic_local.py` add:

```python
def local_ode_rhs(logR, y, lambda0, params):
    A, c, *_ = scaled_differential_matrix(logR, y, lambda0, params)
    return np.linalg.solve(A, -c)
```

Add a safe version:

```python
def local_ode_rhs_safe(logR, y, lambda0, params):
    A, c, diagnostics = ...
    if diagnostics.smin_over_smax < threshold:
        raise NearSonicMatrixError
    return solve(A, -c)
```

## 12.2 Add sonic derivative branch generator

```python
def sonic_derivative_branches(logR_s, y_s, lambda0, params):
    # compute D,C diagnostics
    # compute g_p and null vector r
    # find roots of regularity condition
    # return list[Branch(g_s, diagnostics)]
```

Keep the existing root scan, but make it callable.

## 12.3 Add flow map

```python
def sonic_flow_map(logR_s, y_s, lambda0, params, branch, epsilon0, epsilon_buf):
    g_s = branch.g_s
    y0 = y_s + epsilon0 * g_s
    sol = solve_ivp(
        lambda x, y: local_ode_rhs(x, y, lambda0, params),
        (logR_s + epsilon0, logR_s + epsilon_buf),
        y0,
        method="Radau",
        rtol=1e-9,
        atol=1e-11,
    )
    return sol.y[:, -1], diagnostics
```

Later add \(0.5\epsilon_0^2h_s\) if needed.

## 12.4 Add offset BVP mode

In `transonic_collocation.py` add:

```text
sonic_patch_mode = "offset_flowmap"
```

Unknown packing:

```text
logu_s, logT_s
regular inner logu/logT nodes beginning at epsilon_buf
outer nodes
logR_son
lambda0
```

Residuals:

```text
D
C_pivot
regular node 0 - sonic_flow_map(...)
inner regular collocation
interface continuity
outer domain
far boundary
```

At first, keep branch choice fixed per run.

---

# 13. Validation gates

## Gate A: local branch audit

Pass if at least one derivative branch integrates from:

```text
epsilon0 = 1e-5
to
epsilon_buf = 0.01 or 0.02
```

without singularities and has plausible physical diagnostics.

## Gate B: local shooting fit

Pass if the four-variable local fit matches the dynamic buffer state with:

```text
distance in logu/logT < 1e-3--1e-2
D,C small
```

## Gate C: full Nreg64 root

Pass if:

```text
physical residual <= few e-6
D,C1,C2,K small
patch/flow residual not dominant
```

## Gate D: Nreg refinement

Pass if:

```text
Nreg64,80,96,112,128
```

remain stable in:

```text
Rson
lambda0
int_adv
profiles
```

## Gate E: epsilon convergence

Pass if values converge as:

```text
epsilon_buf -> 0
```

---

# 14. What not to do next

Do not add:

```text
wind
stream source
tidal torque
high-Mdot continuation
time-dependent limit cycle
synthetic light curve
```

until the fixed-Mdot sonic root is mesh-robust.

Do not keep tuning C1/C2/K weights. The soft-compatibility experiments already
show this does not cure the floor.

Do not compare the true sonic derivative directly to the dynamic-patch finite
slope and reject it based only on that comparison.

---

# 15. Compact Codex prompt

```text
The Taylor/L'Hopital patch failed because the L'Hopital derivative roots are
large, g~(-183,74) or (161,-96), while the old dynamic-patch slope is a finite
average over Delta_s=0.02, not the true sonic derivative. A Taylor expansion
over Delta_s=0.02 with |g|~200 is invalid. Replace the Taylor patch with an
offset sonic flow-map boundary.

Implement:
1. local_ode_rhs(logR,y,lambda0) = -A^{-1}c for nonsingular points.
2. sonic_derivative_branches(logR_s,y_s,lambda0) using the existing L'Hopital
   root scan.
3. sonic_flow_map: start at logR_s+epsilon0 with y=y_s+epsilon0*g_s,
   integrate the regular ODE to logR_s+epsilon_buf using solve_ivp(Radau).
4. offset_flowmap BVP mode:
       unknown sonic state y_s, logR_s, lambda0
       regular inner nodes start at epsilon_buf
       residuals: D=0, C_pivot=0, y_inner0 - Phi=0, regular collocation,
       interface continuity, outer domain, far boundary.
5. Run both derivative branches. Select branch by flow-map/global residual and
   physical diagnostics, not by closeness of g_s to the old finite-buffer slope.
6. If neither branch matches, solve a local four-variable sonic-to-buffer
   shooting fit for logR_s, logu_s, logT_s, lambda0 with D,C and Phi-y_buffer=0.
7. Validate with Nreg64 then Nreg80/96/112/128 and epsilon_buf scans
   0.02,0.015,0.01,0.0075,0.005.
```

---

# 16. Bottom line

The next production method should be:

```text
sonic constraints at the exact critical point
+
regular ODE flow map from epsilon0 to epsilon_buf
+
ordinary collocation only outside the singular neighborhood.
```

This is the most direct way to avoid both failure modes currently seen:

```text
Taylor expansion too long,
micro-domain collocation too singular-sensitive.
```
