# Codex Next-Step Brief: Second-Critical Barrier Requires Desingularized / DAE Continuation

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
latest main branch
```

Start from:

```text
Note/GPT_HANDOFF_BRANCH_BARRIER_NEXT_STEP.md
```

Key outputs:

```text
outputs/tables/transonic_branch_connection_scan.md
outputs/tables/transonic_branch_barrier_audit.md
outputs/tables/transonic_barrier_critical_probe.md
outputs/tables/transonic_second_critical_flowmap_probe.md
outputs/tables/transonic_short_join_free_tail_scan.md
```

## Executive conclusion

This handoff is not just the same information as the previous prompt. It is
more precise and changes the numerical target.

The exact reverse-seeded sonic branch is locally regular near the first sonic
point, but ordinary ODE continuation in \(x=\ln R\) fails at a second singular
structure near:

```text
R ~= 6.238 rg
```

The state there is not physically exploding:

```text
H/R ~= 5e-4
Omega/Omega_K ~= 0.9996
Qadv/Qvisc ~= 1
```

but the differential matrix becomes singular:

```text
smin/smax ~= 8e-11
cond(A) ~= 1e10
max |dy/dlnR| ~= 1e8
```

So the next bottleneck is **not merely tail matching**. It is the mathematical
regularization of a second critical barrier.

The correct next strategy is:

```text
Replace g = -A^{-1}c ODE continuation through the barrier with a
desingularized phase-space / DAE formulation.
```

Do not resume high-Mdot continuation yet.

---

# 1. What “compatible but no L'Hopital derivative branch” means

At the second-critical candidates, the algebraic compatibility probe finds:

```text
D ~= 0
C1 ~= 0
C2 ~= 0
K ~= 0
```

for candidates near:

```text
fixed lambda0: R ~= 6.118 rg
free lambda0:  R ~= 6.219 rg with dlambda0 ~= +0.0012
```

But the second-critical flow-map probe reports:

```text
no L'Hopital derivative branches found
```

even after very wide derivative scans.

This combination means:

```text
The point is algebraically compatible, but the solution is not a regular
crossing in the coordinate x=lnR under the current A g + c = 0 formulation.
```

The likely possibilities are:

1. **A projected fold in \(R\)**
   The solution curve may be smooth in an abstract parameter \(s\), but
   \(dx/ds\to0\). Then \(dy/dx\) diverges and no finite \(g=dy/dx\) exists.

2. **A higher-order or tangent critical point**
   The first L'Hopital condition is not sufficient, or the standard scalar scan
   is looking for the wrong local parameterization.

3. **A DAE-type critical barrier**
   The branch should be treated as an implicit curve in phase space, not as an
   explicit ODE in \(x\).

4. **A physical branch termination**
   This should be concluded only after desingularized continuation fails.

The evidence currently favors cases 1--3, not immediate physical termination.

---

# 2. Why \(x=\ln R\) is the wrong independent variable near the barrier

The local ODE is written as:

```math
A(x,y;\lambda_0)g+c(x,y;\lambda_0)=0,
```

where:

```math
g=\frac{dy}{dx},
\qquad
y=(\ln u,\ln T).
```

Away from singular points:

```math
g=-A^{-1}c.
```

At the barrier:

```text
A becomes singular,
g diverges,
ODE integrator step size collapses.
```

But this does not prove that the curve \((x,y)\) itself is singular. It only
proves that the graph representation \(y(x)\) is singular.

The proper local representation is to introduce an arbitrary curve parameter
\(s\):

```math
x=x(s),\qquad y=y(s).
```

Then:

```math
g=\frac{dy/ds}{dx/ds}.
```

Substitute this into:

```math
A g+c=0
```

and multiply by \(dx/ds\):

```math
A(x,y)\frac{dy}{ds}+c(x,y)\frac{dx}{ds}=0.
```

Define:

```math
p =
\frac{dz}{ds}
=
\begin{pmatrix}
dx/ds\\
dy_1/ds\\
dy_2/ds
\end{pmatrix},
\qquad
z=(x,y_1,y_2).
```

Then the desingularized equation is:

```math
B(z)p=0,
```

where:

```math
B(z)=\begin{pmatrix} c(z) & A(z) \end{pmatrix}.
```

This remains meaningful even when \(A\) is singular.

---

# 3. Immediate experiment: desingularized null-vector flow

Before building a full BVP, implement a local desingularized flow.

## 3.1 Local vector field

At any state \(z=(x,y)\), compute:

```math
B(z)=\begin{pmatrix} c(z) & A(z) \end{pmatrix}.
```

Find its right null vector:

```math
B(z)p=0.
```

Normalize \(p\) in a chosen metric:

```math
||p||_M=1.
```

Orient it continuously so that:

```math
p_i \cdot_M p_{i-1} > 0.
```

Then integrate:

```math
\frac{dz}{ds}=p(z).
```

This is the desingularized local flow.

## 3.2 Why this is the first test

At ordinary points:

```math
p \propto (1,g),
```

so it reproduces the usual ODE.

At a projected fold:

```math
dx/ds \to 0,
```

but the flow can continue.

This directly answers whether the barrier near \(R\simeq6.238r_g\) is:

```text
a coordinate singularity in x
or
a true termination of the solution curve.
```

## 3.3 Diagnostics

Create:

```text
scripts/run_transonic_desingularized_barrier_flow.py
outputs/tables/transonic_desingularized_barrier_flow.md
outputs/figures/transonic_desingularized_barrier_flow.png
```

Output:

```text
s
R/rg
x
logu
logT
p_x
p_logu
p_logT
p_x sign
D
C1
C2
K
smin/smax(A)
smin/smax(B)
cond(A)
cond(B)
H/R
Omega/Omega_K
Qadv/Qvisc
Qvisc sign
```

## 3.4 Interpretation

### Case A: desingularized flow crosses \(R\simeq6.238r_g\)

Then the barrier is mainly a coordinate singularity. Use the desingularized
trajectory to seed a phase-space collocation segment.

### Case B: \(p_x\to0\) and then changes sign

Then the solution curve turns around in \(R\). A monotonic radial disk branch
may not exist beyond that point for this branch. The old tail is unreachable
from this exact branch.

### Case C: \(B\) itself becomes singular

Then the problem is a higher-codimension critical point. Move to a DAE/two-
critical BVP.

### Case D: physical quantities become pathological

Then this exact branch is physically invalid beyond the barrier.

---

# 4. Phase-space collocation segment

If the null-vector flow suggests a smooth curve exists, replace the ODE tail
near the barrier with phase-space collocation.

## 4.1 Unknowns

For each transition node \(i\):

```text
x_i = logR_i
y_i = (logu_i, logT_i)
p_i = (p_x_i, p_u_i, p_T_i)
```

Unlike the older collocation, \(x_i\) should be an unknown in the transition
segment. Do not fix all transition radii if the curve may turn in \(R\).

## 4.2 Node residuals

At each node:

```math
B(z_i)p_i=0.
```

This gives two residuals.

Normalize:

```math
||p_i||_M^2-1=0.
```

This gives one residual.

So each node has three algebraic residuals.

## 4.3 Interval residuals

Use trapezoidal phase-space defects:

```math
z_{i+1}-z_i
-\frac{\Delta s_i}{2}(p_i+p_{i+1})=0.
```

This gives three residuals per interval.

At first, use fixed \(\Delta s_i\). Later, allow adaptive \(\Delta s_i\) as
unknowns if needed.

## 4.4 Boundary conditions

Start the segment at:

```text
R_join = 5.95--6.10 rg
```

where the exact branch and best exact BVP still agree.

The left boundary should match:

```text
z_left from exact sonic flow
p_left from desingularized null-vector flow
```

The right boundary should connect to a free tail segment, not a strongly fixed
old dynamic tail.

## 4.5 Why this is better

The current ODE continuation fails because it computes:

```math
g=-A^{-1}c.
```

The phase-space form never requires \(A^{-1}\).

It can pass through points where \(A\) is singular as long as the full
\([c\ A]\) matrix has a well-defined null direction.

---

# 5. Two-critical-point BVP if the barrier is genuine

If the desingularized flow shows that \(B\) also becomes singular, treat the
barrier as an interior critical point.

## 5.1 Segment structure

Use:

```text
Segment 1: first sonic point -> second critical point
Segment 2: second critical point -> tail/outer disk
Outer domain: existing pressure-supported two-domain outer disk
```

## 5.2 Unknowns

Include:

```text
first critical state
second critical state
lambda0
left and right phase-space tangents at second critical point
transition nodes on both sides
outer nodes
```

## 5.3 Residuals at the second critical point

At minimum:

```math
D_2=0,
```

```math
C_{2,\rm pivot}=0,
```

and phase-space tangent equations on both sides:

```math
B(z_2)p_- = 0,
```

```math
B(z_2)p_+ = 0.
```

Normalize:

```math
||p_-||_M=1,
\qquad
||p_+||_M=1.
```

Then impose an orientation/branch condition. Possible choices:

```text
p_+ = p_-        for smooth crossing
p_+ != p_-       for branch switching
```

Start with smooth crossing:

```math
p_+-p_-=0
```

as a test, but keep branch-switching as a diagnostic option.

## 5.4 Key audit

The critical question is whether the incoming tangent from the exact branch
aligns with any outgoing tangent that can reach the outer solution.

---

# 6. Fixed-lambda versus free-lambda candidates

The handoff reports:

```text
fixed lambda0 candidate: R ~= 6.118 rg
free lambda0 candidate:  R ~= 6.219 rg, dlambda0 ~= +0.0012
```

For a single fixed-\(\dot M\) disk solution, \(\lambda_0\) is constant along
radius. It cannot change locally at the second critical point.

Therefore:

```text
The fixed-lambda candidate is the relevant candidate for continuing the
currently identified branch.
```

The free-lambda candidate is still useful, but it belongs to a neighboring
global solution family. It should be studied via continuation in \(\lambda_0\),
not by allowing lambda to vary inside one radial profile.

Recommended order:

```text
1. Try desingularized continuation through the fixed-lambda candidate.
2. If that fails, continue the whole global solution in lambda0 toward the
   free-lambda candidate.
```

---

# 7. Why short-join free-tail scans still fail

The short-join free-tail scan shows:

```text
R_join=6.0: physical ~= 0.068
R_join=6.1: physical ~= 0.088
R_join=6.2: physical ~= 0.105
dominant = tail_R
```

This is better than joining after the barrier, but still not solved.

Reason:

```text
The short-join scan attaches an exact local branch to a tail segment, but the
transition segment still does not explicitly handle the second critical
barrier/turning structure.
```

So the correct next move is not more tail weighting. It is a phase-space
transition segment.

---

# 8. Proposed implementation order

## Step 1: local desingularized barrier flow

Implement and run:

```text
scripts/run_transonic_desingularized_barrier_flow.py
```

Use initial data from the free reverse-fit branch at:

```text
R ~= 6.0--6.1 rg
```

not from \(R=7\) or beyond.

## Step 2: phase-space transition segment

Implement a standalone segment from:

```text
R_left ~= 6.0 rg
```

to:

```text
R_right ~= 7--8 rg
```

with no outer disk initially.

Check whether it can cross the \(6.238r_g\) barrier.

## Step 3: phase-space segment + free tail

Attach:

```text
phase-space segment -> free ordinary tail -> outer two-domain disk.
```

Use \(R_{\rm join}\le6.1r_g\).

## Step 4: global two-domain BVP

Only after the segment works, reconnect to:

```text
R_match = 6500 rg
R_far = 1e5 rg
```

## Step 5: grid validation

Only after a fixed-\(\dot M\) global solution exists:

```text
N transition = 64, 96, 128
```

and then high-\(\dot M\) continuation.

---

# 9. Concrete code additions

## `transonic_local.py`

Add:

```python
def extended_matrix_B(logR, y, lambda0, params):
    A, c, *_ = scaled_differential_matrix(logR, y, lambda0, params)
    return np.column_stack([c, A])
```

Add:

```python
def phase_space_null_tangent(logR, y, lambda0, params, metric=None, previous=None):
    B = extended_matrix_B(logR, y, lambda0, params)
    # compute right null vector p=(px,pu,pT)
    # normalize in metric
    # orient using previous
    return p, diagnostics
```

Add:

```python
def phase_space_diagnostics(logR, y, p, lambda0, params):
    # report Bp, D,C1,C2,K, smin/smax(A), smin/smax(B), etc.
```

## New module

```text
src/imri_qpe/layer3_minidisk_1d/transonic_phase_space.py
```

Core residual:

```python
def phase_space_segment_residual(z, params):
    # unpack x_i, y_i, p_i
    # node residuals: B_i p_i
    # normalization residuals: ||p_i||_M^2 - 1
    # interval residuals: z_{i+1}-z_i - 0.5*ds_i*(p_i+p_{i+1})
```

## New scripts

```text
scripts/run_transonic_desingularized_barrier_flow.py
scripts/run_transonic_phase_space_barrier_segment.py
scripts/run_transonic_phase_space_global_connection.py
scripts/run_transonic_second_critical_two_point_bvp.py
```

## New outputs

```text
outputs/tables/transonic_desingularized_barrier_flow.md
outputs/tables/transonic_phase_space_barrier_segment.md
outputs/tables/transonic_phase_space_global_connection.md
outputs/tables/transonic_second_critical_two_point_bvp.md
```

---

# 10. Acceptance criteria

## Desingularized barrier flow passes if

```text
it crosses R=6.238 rg without singular B,
physical variables remain finite,
and p_x behavior identifies whether R turns around.
```

## Phase-space segment passes if

```text
node residuals Bp <= few x 1e-6
interval defects <= few x 1e-6
no residual spike at R~6.238 rg
```

## Global connection passes if

```text
tail_R residual drops from ~0.07--0.10 to <= few x 1e-5 first,
then <= few x 1e-6 after polish.
```

## Stop condition

If phase-space flow shows:

```text
x(s) reaches a maximum and turns around
```

and cannot reconnect to increasing \(R\), then the exact branch cannot form a
monotonic radial disk beyond the second critical point under the current
closure.

---

# 11. Compact Codex prompt

```text
The handoff identifies a second critical barrier near R~6.238 rg. The exact
reverse-seeded sonic branch is locally regular, but ordinary ODE integration
fails because A becomes singular: smin/smax~8e-11, cond(A)~1e10, |dy/dlnR|~1e8,
while H/R~5e-4 and Omega/OmegaK~0.9996. Algebraic critical probes find
compatible candidates at R~6.118 fixed lambda and R~6.219 with lambda shifted,
but the usual L'Hopital derivative scan finds no branches. This means x=lnR is
probably the wrong independent variable near the barrier.

Implement desingularized phase-space continuation.

Instead of g=-A^{-1}c, define z=(x,logu,logT), p=dz/ds, and
B=[c A]. The regular curve satisfies B(z)p=0. Compute normalized right-null
tangent p and integrate dz/ds=p through the barrier. Track p_x to see whether
R turns around.

If local null-flow crosses the barrier, build a phase-space collocation segment
with unknowns z_i and p_i:
    node residual: B_i p_i=0
    normalization: ||p_i||=1
    interval residual: z_{i+1}-z_i-0.5 ds_i(p_i+p_{i+1})=0
Use this segment from Rjoin~6.0 to R~7--8 before attaching the tail.

Start with the fixed-lambda second-critical candidate near R~6.118. Treat the
free-lambda candidate as a neighboring global solution, not a local radial
change of lambda.
```

---

# 12. Bottom line

Yes, `GPT_HANDOFF_BRANCH_BARRIER_NEXT_STEP.md` helps. It supersedes the
previous tail-matching-only diagnosis.

The real next problem is:

```text
Can the exact branch pass through a second critical barrier near R~6.24 rg
when formulated as a desingularized phase-space curve instead of y(logR)?
```

That is what Codex should test next.
