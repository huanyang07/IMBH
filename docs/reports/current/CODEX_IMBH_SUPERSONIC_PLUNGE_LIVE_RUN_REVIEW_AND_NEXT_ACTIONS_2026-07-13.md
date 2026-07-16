# Codex Work Order: Supersonic-Plunge Certification and Next Simulation Gates

**Project:** IMBH/QPE stream-fed minidisk
**Date:** 2026-07-13, Asia/Tokyo
**Primary source reviewed:** `CODEX_GLOBAL_SUPERSONIC_PLUNGE_RESULTS_2026-07-13.md`
**Context:** The `N=128` supersonic-plunge certification run is still active.
**Purpose:** Preserve the successful plunge-boundary result, finish the current certification cleanly, and prevent premature transition to tide/wind simulations.

---

# 1. Executive decision

The new causally outgoing supersonic-plunge boundary appears successful enough to replace the old fixed-reference inner absorber.

The live `N=128` run is progressing through controlled timestep reduction rather than exhibiting a physical or numerical instability.

Therefore:

```text
DO NOT interrupt the active N=128 Newton solve.
DO NOT change equations, tolerances, Jacobian backend, or timestep gates mid-run.
DO NOT begin physical tide or wind calculations yet.
```

The immediate goals are:

1. finish the `N=128` trajectory to the declared shared time;
2. certify mesh behavior using common-radius plunge diagnostics;
3. separate mapped-state relaxation from stream-driven loading;
4. verify that the previous `N=64` collapse is genuinely removed;
5. audit the sonic-gradient mismatch;
6. only then permit a zero-amplitude/tiny-step tidal-operator preflight.

---

# 2. Current numerical interpretation

The live run has shown:

```text
causally outgoing plunge
no incoming radial characteristics
positive mapped internal energy
full-step/half-step convergence
very small nonlinear residuals
near-roundoff mass/angular/energy ledger defects
controlled timestep retries
```

The latest timestep rejection is understood as:

```text
the nonlinear solve converged
but the proposed physical state change exceeded the unchanged 2% gate
```

This is adaptive timestep control, not evidence of branch failure.

The current bottleneck is computational expense, especially in the inner plunge, rather than a new closure pathology.

---

# 3. Locked instructions for the active N=128 run

## 3.1 Finish the current certification unchanged

Continue to:

\[
t=10^{-6}t_{\rm load}
\]

with:

```text
the same equations
the same serial finite-difference Jacobian
the same Newton tolerances
the same 2% physical-change gate
the same timestep controller
the same plunge boundary
```

Do not introduce a new derivative backend during an unaccepted step.

## 3.2 Checkpoint requirements

At every accepted checkpoint, record:

```text
Git SHA
configuration hash
current physical time
current timestep
accepted/rejected step counts
Newton iterations
maximum residual by equation block
mass/J/E ledger defects
inner characteristic count
outer Roche/nozzle active state
minimum internal energy
maximum H/R
```

---

# 4. Colored Jacobian policy

A colored finite-difference Jacobian may reduce runtime, but it must not be introduced into the live production trajectory without equivalence certification.

## 4.1 Offline equivalence test

From one accepted checkpoint, branch the calculation and perform one identical step with:

```text
A. current serial finite-difference Jacobian
B. colored finite-difference Jacobian
```

Use exactly the same:

```text
state
timestep
tolerances
line search
physical-change gate
boundary conditions
```

## 4.2 Required equivalence

Require:

```text
same accept/reject decision
same next timestep chosen by the controller
same nonlinear residual gate
same mass/J/E ledger closure
state difference far below full-step/half-step temporal error
repeatable restart result
```

## 4.3 Coloring audit

The coloring dependency graph must include:

```text
neighboring numerical fluxes
inner and outer boundary rows
radiative cooling
radial and temporal column work
mechanical quadrature offsets
stream source terms
Roche/nozzle terms
any global normalization or active-set dependency
```

Any uncertain or global dependency gets its own color.

## 4.4 Production rule

Preferred:

```text
finish the current N=128 certification with the serial Jacobian
```

A colored backend may be activated only from a later accepted checkpoint, with the backend change recorded in provenance.

---

# 5. Shared-time mesh diagnostics

Comparing first-cell Mach numbers directly is invalid because the first-cell radii differ among meshes.

At the shared time, report the following for `N=64,96,128`.

## 5.1 Common-radius plunge diagnostics

Interpolate or reconstruct to the same physical radii, for example:

```text
4.65 rg
4.75 rg
5.00 rg
```

Report:

```text
radial Mach number
Sigma
temperature
Omega/Omega_K
H/R
mass flux
angular-momentum flux
total-energy flux
```

## 5.2 Emergent sonic structure

Report:

```text
Mach=1 crossing radius
number of cells across the supersonic plunge
minimum radial gradient length / cell width
minimum radial gradient length / H
```

## 5.3 Interface-independent conserved quantities

Compare:

\[
\dot M_{\rm inner},
\qquad
J_{\rm inner},
\qquad
F_{E,\rm inner}.
\]

These are more robust convergence quantities than the first-cell Mach number.

---

# 6. Physical-change controller audit

Every physical-change rejection must identify what triggered it.

Record:

```text
controlling variable
cell index
physical radius
old value
proposed value
fractional change
local Mach number
whether the cell is inside the causally disconnected plunge
```

Do not change the 2% controller during the present certification.

## Later optimization gate

If a deeply supersonic cell consistently controls the global timestep while all characteristics leave the domain, a later work package may test:

```text
local plunge substepping
a multirate implicit method
moving the inner edge while retaining a safe supersonic margin
```

These are optimization experiments and require equivalence tests. They are not part of the current certification.

---

# 7. Sonic-gradient mismatch audit

The stationary plunge construction reports a sonic-gradient mismatch of approximately:

\[
0.189813.
\]

The inward ODE solution closes locally, but this is not yet a demonstrated \(C^1\) match to the first resolved outer interval.

This mismatch must be audited after the active run, not used to interrupt it.

## Required tests

Repeat the stationary plunge construction while varying:

```text
starting logarithmic offset from the sonic point
outer transonic resolution
ODE tolerance
all regular sonic derivative roots
```

Report:

```text
sonic-gradient mismatch
state at 4.5 rg
Mdot/J/F_E at 4.5 rg
mapped global residual near the sonic region
subsequent short-time global evolution
```

## Decision

If the mismatch converges downward, certify the initialization.

If it remains finite but the conserved plunge fluxes and global evolution are insensitive, classify it as an initialization-gradient diagnostic.

If it materially changes the early plunge adjustment, improve the sonic-to-global mapping before physical loading studies.

---

# 8. Roche/nozzle boundary margin

Reporting zero Roche flux is not sufficient.

At every shared-time output, report:

```text
Roche/nozzle active-set state
incoming outer characteristic count
Jacobi/Bernoulli margin to opening
sonic/nozzle regularity residual
distance from the opening threshold
mass/J/E overflow flux
```

Interpretation:

```text
large positive closed-channel margin:
    genuinely inactive overflow

margin approaching zero:
    impending physical opening

active channel:
    overflow must close all conservative ledgers
```

Do not infer physical closure solely from a numerically zero flux.

---

# 9. N=64 extension beyond the old failure

The previous collapse occurred near:

\[
3.9166\times10^{-6}t_{\rm load}.
\]

Extending only to \(2\times10^{-6}t_{\rm load}\) cannot test whether that failure is gone.

## Required bounded extension

After the shared-time certification, extend `N=64` to at least:

\[
\boxed{
5\times10^{-6}t_{\rm load}
}
\]

unless one of the following occurs first:

```text
nonlinear residual failure
mass/J/E ledger failure
loss of causal plunge
Roche/nozzle opening or boundary-regime transition
negative internal energy
unresolved physical front
```

The goal is not yet a long physical evolution.

The goal is to pass the old collapse time with a clear margin.

---

# 10. Source-off relaxation control

The current elapsed time is tiny relative to one loading time, while the plunge variables have already changed appreciably.

This may reflect mapped-state relaxation rather than stream-driven loading.

Run one bounded source-off control:

```text
same mapped initial state
same mesh
same plunge boundary
same timestep and physical gates
stream source set to zero
same short duration
```

A practical first control may use `N=64` or `N=96`.

Compare source-on and source-off:

```text
inner mass flux
inner J and F_E
sonic radius
common-radius Mach profile
maximum H/R
disk mass
thermal energy
controller-triggering cell
```

## Interpretation

### Nearly identical early trajectories

Then the current evolution is mainly initialization relaxation.

Required workflow:

```text
relax a source-free global state
turn the stream on smoothly
start the physical loading clock after relaxation
```

### Clearly different trajectories

Then the stream source is already driving the early response.

Proceed with the declared stream turn-on protocol.

---

# 11. Physical timescale reporting

Every evolution report must translate elapsed time into local physical timescales.

At minimum, report:

\[
t/t_{\rm dyn},
\qquad
t/t_{\rm th},
\qquad
t/t_{\rm in},
\qquad
t/t_{\rm visc},
\qquad
t/t_{\rm load}.
\]

Evaluate these near:

```text
the inner plunge
the stream/source annulus
the tidal/Hill region
```

Do not describe a run as “long” or “stable” using \(t/t_{\rm load}\) alone.

A small fraction of a loading time may still contain many inner dynamical times but very few outer thermal or viscous times.

---

# 12. Tide implementation: what is allowed next

The short plunge certification does not authorize a physical tidal-amplitude campaign.

## Allowed after the plunge gates

Codex may implement and test the distributed-tide operator with:

```text
zero-amplitude recovery
manufactured spatial torque profile
tiny-step conservation
paired pattern-speed power
disk-plus-orbit angular-momentum ledger
disk-plus-orbit energy ledger
```

This is an operator preflight only.

## Still blocked

Do not begin a physical tidal-amplitude continuation or hot-phase search until:

```text
the shared-time plunge comparison passes
the N64 old-failure extension passes
the source-off relaxation control is interpreted
the no-tide baseline is tested over declared local timescales
```

Wind remains blocked until the no-wind tidal evolution is understood.

---

# 13. Updated certification sequence

```text
P0  Finish active N128 to 1e-6 t_load unchanged.

P1  Produce the common-time N64/N96/N128 diagnostic snapshot.

P2  Audit controller-triggering cells and variables.

P3  Perform sonic-gradient refinement.

P4  Run the bounded source-off relaxation control.

P5  Extend N64 beyond the old failure to >=5e-6 t_load.

P6  Certify or reject the colored Jacobian offline.

P7  Extend the no-tide baseline over declared local thermal/inflow times.

P8  Implement only zero-amplitude/tiny-step distributed-tide preflights.

P9  Begin physical finite-tide evolution.

P10 Add wind after the no-wind tide/limit-cycle behavior is understood.
```

---

# 14. Acceptance gates

## Active N128 completion

```text
reaches exactly 1e-6 t_load
no incoming inner characteristics
residuals pass unchanged gate
mass/J/E ledgers pass unchanged gates
no hidden clipping
restart reproducibility retained
```

## Shared-time mesh support

```text
inner Mdot/J/F_E convergence
common-radius Mach convergence
sonic radius convergence
maximum H/R convergence
plunge adequately resolved
Roche margin reported
```

Use predeclared quantitative tolerances from the current project status; do not relax them after seeing the result.

## Colored Jacobian

```text
same accepted state within temporal error
same controller decision
same ledger closure
complete dependency coloring
```

## Source-off control

```text
source-on/off difference quantified
initial-relaxation contribution identified
physical loading clock defined
```

## N64 extension

```text
passes former 3.9166e-6 failure time
reaches >=5e-6 t_load or a named physical/numerical stop event
```

---

# 15. Stop conditions

Stop and reassess if:

1. common-radius plunge quantities do not converge with mesh;
2. the emergent sonic radius drifts without convergence;
3. the 0.189813 gradient mismatch controls the global evolution;
4. the Roche/nozzle margin is pinned numerically at zero;
5. source-off and source-on trajectories reveal an uncontrolled initialization transient;
6. the old N64 collapse reappears after the new plunge boundary;
7. colored Jacobian results differ beyond temporal truncation error;
8. inner characteristic causality is lost;
9. the deep plunge controls cost so severely that outer physical evolution cannot be reached.

In case 9, test multirate/subcycled plunge treatment only in a separate equivalence work package.

---

# 16. Immediate Codex instruction

```text
Do not interrupt the active N128 solve.

Finish it to 1e-6 t_load using the current serial Jacobian, equations,
tolerances, and 2% physical-change controller.

After completion:

1. Produce a common-time N64/N96/N128 snapshot.
2. Compare Mach and primitives at common radii, not first-cell centers.
3. Report emergent sonic radius, inner Mdot/J/F_E, plunge resolution,
   controller-triggering variable/cell, and Roche-opening margin.
4. Audit the 0.189813 sonic-gradient mismatch through offset and resolution
   refinement.
5. Run one short source-off control to separate mapped-state relaxation from
   physical stream loading.
6. Extend N64 beyond the former 3.9166e-6 collapse, preferably to at least
   5e-6 t_load.
7. Test the colored Jacobian only offline from the same accepted checkpoint
   and timestep.
8. Permit only zero-amplitude/tiny-step tide-operator tests afterward.
9. Do not begin physical tidal continuation until the no-tide state is
   evolved over declared local thermal and inflow times.
10. Wind remains last.
```

---

# 17. Bottom line

The new supersonic-plunge boundary appears to have removed the old inner-boundary pathology.

The live `N=128` run currently shows controlled adaptive integration, not instability.

The next task is to distinguish:

```text
true stream-driven loading
from
relaxation of the mapped initial state
```

and to show that the plunge solution is converged at common physical radii.

Finish the present certification without changing the solver, then perform the bounded diagnostics above before beginning the physical tide campaign.
