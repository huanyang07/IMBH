# Response to Codex on Commit `35dbd2f`: Revised Diagnostics, Solver Gate, and Physics-Control Plan

**Project:** IMBH/QPE stream-fed minidisk
**Repository:** `huanyang07/IMBH`
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Current milestone referenced by Codex:** commit `35dbd2f`
**Date:** 2026-07-14, Asia/Tokyo
**Purpose:** Accept Codex’s corrections, remove stale instructions, and lock the next sequence without reopening rejected solver paths.

---

# 1. Executive verdict

I agree with Codex’s reply.

The previous work order was written before commit `35dbd2f`, so its instruction to finish the `N=128` trajectory is stale.

The updated status is:

```text
N=64,96,128 persisted milestone near 1.001e-6 t_load    complete
N=128 completion gate                                   passed
comparison at exactly equal physical seconds            still required
fixed-radius plunge diagnostics                          still required
inner J and F_E reporting                               missing
controller-trigger location                             missing
Roche channel                                            safely closed in raw energy margin
colored finite-difference Jacobian                       rejected by repository regression
sparse-forward derivative path                           current certified baseline
source-on/source-off separation                          high scientific priority
former N64 failure-time extension                        still required
physical distributed-tide continuation                   blocked
wind                                                      blocked
```

The new priority order should be:

```text
1. Add missing diagnostics and immutable milestone checkpoints.
2. Perform one bounded solver-efficiency certification.
3. Produce an exact-common-physical-time mesh comparison.
4. Audit sonic initialization and run matched source-on/source-off controls.
5. Extend N64 beyond the former collapse to at least 5e-6 t_load.
6. Certify the no-tide baseline in local physical timescale units.
7. Run only zero-amplitude/tiny-step distributed-tide preflights.
8. Begin physical tide only after all preceding gates pass.
9. Add wind last.
```

The solver-efficiency step must be tightly capped. It must not become another open-ended numerical campaign that postpones the source-off physics control.

---

# 2. Freshness and evidence note

The exact `35dbd2f` GitHub tree could not be independently fetched during this review because of GitHub cache/rate-limit behavior.

This work order therefore treats the detailed Codex response as the current authoritative milestone description.

Codex should attach:

```text
exact full SHA
branch name
test count
milestone checkpoint hashes
current PROJECT_STATUS excerpt
```

to the next report.

Do not rerun a completed task merely because an older review listed it.

---

# 3. Corrections accepted from Codex

## 3.1 `N=128` is complete

The old P0 instruction is closed.

Do not rerun the `N=128` trajectory.

Preserve all three milestone states and their accepted-step histories.

## 3.2 Equal normalized loading fraction is not equal physical time

The existing comparison used equal normalized loading fraction.

The reported physical-time mismatch is small—approximately

\[
2.36\times10^{-4}
\]

between `N=64` and `N=128` at the milestone—but the final mesh certification should use exactly one physical time.

## 3.3 First-cell Mach is not a convergence diagnostic

The first cell center occurs at a different radius on each mesh.

Use common physical radii and an emergent sonic-point diagnostic instead.

## 3.4 Inner angular and energy fluxes are missing

The runner currently reports inner mass flux only.

Every milestone must also include:

\[
F_{J,\rm inner},
\qquad
F_{E,\rm inner}.
\]

## 3.5 Controller location is missing

The controller records only maximum state changes.

It must report the controlling variable, cell, radius, and characteristic regime.

## 3.6 The Roche channel is not pinned numerically at zero

The available-energy margin is strongly negative, approximately

\[
-8.6\times10^{16}
\text{ to }
-8.9\times10^{16}\ {\rm erg\,g^{-1}}.
\]

That is strong evidence that the channel is physically closed at the current milestone.

Still report a normalized margin and active-set/complementarity diagnostic.

## 3.7 Do not revive the colored Jacobian

The repository already contains a regression rejecting the colored derivative under thermal-energy scaling.

The sparse-forward derivative path is the certified baseline.

Future efficiency work should use deterministic gate-aware termination and selected local/block derivative columns, not the rejected coloring architecture.

---

# 4. Immutable milestone checkpoint policy

The current workflow should stop overwriting one checkpoint per mesh.

Create immutable files such as:

```text
global_plunge_N064_tphys_<value>_sha_<short>.npz
global_plunge_N096_tphys_<value>_sha_<short>.npz
global_plunge_N128_tphys_<value>_sha_<short>.npz
```

Each milestone must include or reference:

```text
full Git SHA
configuration
mesh edges and centers
physical time in seconds
t/t_load
accepted and rejected step history
current timestep
Jacobian backend
residual tolerances
physical-change gate
mass/J/E ledgers
mechanical-reference array and hash
Roche active state and margin
inner characteristic count
```

Create a manifest containing SHA-256 for every checkpoint.

A restart must reproduce the same next accepted state under the same backend and timestep.

---

# 5. Diagnostics to implement first

These changes are inexpensive and must precede additional long trajectories.

## 5.1 Exact common-physical-time output

Choose one common physical time

\[
t_{\rm common}
\]

that lies inside the accepted trajectory of all three meshes.

Preferred method:

1. restart from the accepted checkpoint immediately before \(t_{\rm common}\);
2. shorten only the final timestep so every mesh lands exactly at the same physical time;
3. preserve the original trajectories;
4. save separate exact-time certification checkpoints.

A diagnostic interpolation between bracketing checkpoints is permitted for exploration, but the final certification should use exact-hit states.

Do not compare meshes solely at equal `t/t_load` if the physical loading time differs slightly with mesh.

## 5.2 Inner conserved flux triplet

Report:

\[
F_M,\qquad
F_J,\qquad
F_E
\]

at the inner face for every accepted milestone and rejected candidate.

## 5.3 Fixed-radius plunge diagnostics

Evaluate at shared radii, initially:

```text
4.65 rg
4.75 rg
5.00 rg
```

Report:

```text
radial Mach
Sigma
temperature
Omega/Omega_K
H/R
mass flux
angular-momentum flux
total-energy flux
radial characteristic speeds
```

## 5.4 Sonic location and resolution

Report:

```text
Mach=1 crossing radius
number of cells between the sonic crossing and inner face
minimum radial gradient length / cell width
minimum radial gradient length / H
```

## 5.5 Controller trigger

For every timestep rejection, record:

```text
controlling variable
cell index
radius
old value
candidate value
fractional change
local Mach
all radial characteristic signs
whether the cell is causally disconnected from the outer disk
```

## 5.6 Roche closure

Define a dimensionless closure margin using a physical barrier scale, for example:

\[
\widehat{\Delta B}_{\rm Roche}
=
\frac{
B_{\rm available}-B_{\rm saddle}
}{
\left|\Phi_{\rm eff}(L_{1/2})
-\Phi_{\rm eff}(R_{\rm edge})\right|
}
\]

or another explicitly derived Hill/Roche barrier normalization.

Report:

```text
raw margin in erg/g
dimensionless margin
active/inactive channel state
complementarity residual
incoming outer characteristic count
overflow M/J/E flux
```

If the channel is inactive, do not report a fabricated sonic/nozzle residual as though a nozzle solution were being solved.

Use:

```text
nozzle residual = not active / not evaluated
```

and report the active-set residual instead.

---

# 6. Bounded solver-efficiency certification

Codex is right that solver cost should be addressed before the source-off and `5e-6` campaigns.

But this work package must be bounded.

The certified reference remains:

```text
sparse-forward derivative assembly
```

Test only two candidate improvements.

## Candidate A — deterministic gate-aware termination

The nonlinear solve may stop early only when all of the following hold:

```text
every physical residual block passes its unchanged gate
mass/J/E ledger defects pass
the Newton update is below a declared state tolerance
the accepted physical-change controller is evaluated
the result is deterministic under restart
```

Do not weaken any residual gate.

Do not terminate merely because the global norm has crossed one aggregate threshold.

Record the exact stop reason.

## Candidate B — local/block derivative columns

Replace only selected expensive, demonstrably local derivative columns, for example:

```text
thermodynamic primitive recovery
radiative cooling
radial and temporal column work
local stress
local source moments
```

Keep uncertain, boundary, active-set, and nonlocal columns on the certified sparse-forward path.

Every block derivative must be checked against directional finite differences.

Do not use the rejected coloring implementation.

## Benchmark states

Use one identical accepted checkpoint and timestep at:

```text
N=64
N=128
```

Run:

```text
reference sparse-forward
candidate A
candidate B
candidate A+B only if each passes separately
```

## Acceptance gates

Require:

```text
same accept/reject outcome
same physical-change controlling variable and cell
same next timestep selected
same residual and ledger gates
state difference <= 0.1 of the measured full-step/half-step temporal error
inner M/J/E differences below the same bound
deterministic restart result
no extra nonlinear retries
```

Also measure:

```text
wall-clock time
residual evaluations
Jacobian evaluations
linear factorizations
inner-boundary time
Roche/source/cooling derivative time
```

Adopt an optimization only if it gives a material speed improvement, preferably at least about `1.5x`, without reducing robustness.

If neither candidate passes, proceed with sparse-forward at `N=64` for the physics controls.

Do not begin a third optimization architecture.

---

# 7. Sonic-gradient audit

The stationary plunge reported a sonic-gradient mismatch near

\[
0.189813.
\]

Run a bounded audit over:

```text
sonic starting offset
outer transonic resolution
ODE tolerance
regular sonic derivative root
```

Compare:

```text
gradient mismatch
state at 4.5 rg
inner M/J/E
common-radius Mach profile
early exact-time global evolution
controller-triggering location
```

Interpretation:

- convergence downward: certify the mapping;
- finite mismatch but invariant flux/evolution: retain as a mapping-gradient diagnostic;
- material trajectory change: improve the sonic-to-global initialization before interpreting source loading.

Preserve every audit seed as an immutable checkpoint.

---

# 8. Matched source-on/source-off control

This control is essential, but its interpretation needs care.

Turning off the source changes the physical equations and may move a state that was balanced with the source.

Therefore use both an instantaneous tendency decomposition and a matched trajectory.

## 8.1 Instantaneous tendency decomposition

At the same initial state, evaluate:

\[
R_{\rm on},
\qquad
R_{\rm off},
\qquad
R_{\rm source}=R_{\rm on}-R_{\rm off}.
\]

Report the source-on, source-off, and source-only contributions to:

```text
cell mass tendency
angular-momentum tendency
energy tendency
inner flux response
controller norm
```

This is the cleanest first separation of mapping/operator relaxation from explicit source forcing.

## 8.2 Matched short trajectories

Use:

```text
same immutable initial checkpoint
same N=64 mesh first
same backend
same timestep sequence where possible
same physical final time
source on versus source off
```

Compare:

```text
inner M/J/E
sonic radius
fixed-radius Mach
maximum H/R
disk mass
thermal energy
controller-triggering variable and cell
```

## 8.3 Interpretation

### On/off trajectories nearly identical

The early evolution is dominated by mapping/operator relaxation.

Then:

1. build or relax a source-free reference;
2. ramp the stream smoothly;
3. define the start of the physical loading clock after relaxation.

### On/off difference comparable to the full evolution

The source is already driving the early response.

Retain the source-on initial condition, but document the initial transient.

### Important qualification

The source-off trajectory is a counterfactual control, not automatically a physical equilibrium.

Do not label every source-off change “mapping relaxation” unless supported by the instantaneous tendency decomposition or a separately relaxed source-free state.

---

# 9. Extend N=64 beyond the former failure

After the diagnostics and bounded solver certification, extend `N=64` to at least:

\[
\boxed{
5\times10^{-6}t_{\rm load}
}
\]

unless a named stop event occurs first.

The former collapse occurred near:

\[
3.9166\times10^{-6}t_{\rm load}.
\]

Allowed stop events:

```text
nonlinear residual failure
mass/J/E ledger failure
loss of causal inner outflow
Roche channel regime transition
negative internal energy
unresolved physical front
one-zone validity failure
```

Report elapsed time in local physical units.

This remains a boundary/stability certification run, not yet the final QPE experiment.

---

# 10. Physical timescale reporting

At every major milestone, report:

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

Evaluate near:

```text
inner plunge
stream/source annulus
Hill/tidal region
```

Do not claim stability of a region before covering several of its relevant thermal and inflow times.

---

# 11. Tide policy

After all previous gates pass, Codex may implement only the distributed-tide operator preflight:

```text
zero-amplitude recovery
manufactured torque profile
paired pattern-speed power
tiny-step disk-plus-orbit angular ledger
tiny-step disk-plus-orbit energy ledger
```

Physical tidal-amplitude continuation remains blocked until:

```text
exact common-time mesh comparison passes
source-on/source-off control is interpreted
N64 passes the former failure time
no-tide evolution covers declared local timescales
```

Wind remains last.

---

# 12. Revised locked sequence

```text
WP0  Preserve completed N64/N96/N128 milestone checkpoints.

WP1  Add exact-time/fixed-radius/full-flux/controller/Roche diagnostics.

WP2  Run one bounded solver-efficiency certification:
     deterministic gate-aware termination;
     selected local/block derivative columns;
     no colored Jacobian.

WP3  Produce the exact-common-physical-time N64/N96/N128 snapshot.

WP4  Audit the sonic-gradient mismatch.

WP5  Run instantaneous source-tendency decomposition and matched
     source-on/source-off N64 trajectories.

WP6  Extend N64 to >=5e-6 t_load or a named stop event.

WP7  Certify the no-tide baseline in local physical timescale units.

WP8  Implement zero-amplitude/tiny-step distributed-tide preflights.

WP9  Begin physical finite-tide evolution.

WP10 Add wind after the no-wind finite-tide behavior is understood.
```

---

# 13. Prohibited actions

Do not:

1. rerun the completed N128 milestone;
2. compare first-cell Mach as a mesh-convergence metric;
3. revive the rejected colored derivative path;
4. launch an open-ended solver-optimization campaign;
5. overwrite milestone checkpoints;
6. interpret zero Roche flux without reporting its margin;
7. interpret the source-off trajectory as pure relaxation without tendency decomposition;
8. begin physical tide before the no-tide gates;
9. add wind before the tide/stability result;
10. loosen residual, ledger, or physical-change gates to save runtime.

---

# 14. Immediate Codex instruction

```text
Treat the N64/N96/N128 1.001e-6 t_load milestone as complete.

First add:
    exact physical time;
    inner M/J/E;
    fixed-radius plunge diagnostics;
    sonic position and resolution;
    controller variable/cell/radius;
    normalized Roche closure margin;
    immutable checkpoint manifests.

Then perform one bounded solver-efficiency certification:
    sparse-forward is the reference;
    test deterministic gate-aware termination;
    test selected local/block derivative columns;
    do not use the rejected colored path.

After that:
    generate exact-common-time N64/N96/N128 checkpoints;
    audit the 0.189813 sonic-gradient mismatch;
    run instantaneous source-on/off tendency decomposition;
    run matched short N64 source-on/source-off trajectories;
    extend N64 to at least 5e-6 t_load or a named stop event.

Only after those gates and local-timescale certification should the
distributed-tide operator preflight begin.

Physical tidal continuation remains blocked.
Wind remains last.
```

---

# 15. Bottom line

Codex’s revised assessment is correct.

The inner plunge milestone is complete.

The next bottleneck is no longer boundary survival at \(10^{-6}t_{\rm load}\).

It is obtaining:

```text
unambiguous diagnostics
a certified efficient nonlinear path
and a clean separation between initialization relaxation and stream forcing.
```

Complete those bounded tasks before spending substantial compute on the physical tide and limit-cycle search.
