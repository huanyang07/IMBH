# Global Solver-Efficiency WP2 Results

**Date:** 2026-07-14
**Scope:** One bounded optimization audit of the accepted global
supersonic-plunge/Roche-edge backward-Euler step.  No equations, physical
boundaries, residual gates, ledger gates, or adaptive physical-change gates
were changed.

## Decision

WP2 is complete as a bounded negative result.

The production path remains the serial, independently perturbed
`sparse_forward` Jacobian.  Neither tested candidate met the adoption gate, so
their experimental solver controls were removed.  The new work counters and
timers remain because they are observational and are now included in adaptive
attempt records.

## Immutable benchmark

The benchmark loaded, but never rewrote,
`outputs/checkpoints/global_supersonic_roche_N64.npz`:

```text
elapsed physical time       2.1747227583983317 s
requested next dt           0.001519729662593771 s
requested dt/t_load         1.0e-9
maximum nonlinear calls     600
```

The checkpoint was reloaded after every benchmark and its four conserved
arrays were bitwise unchanged.

## Reference cost

The accepted serial sparse-forward step gave:

```text
wall time                         197.99 s
solver nfev                       600
solver njev                       596
actual Jacobian assemblies        596
residual evaluations              153773
residual wall time                188.77 s
Jacobian-inclusive wall time      191.81 s
maximum scaled residual           4.6247e-9
maximum storage-ledger defect     2.3821e-16
```

This identifies finite-difference Jacobian residuals as the dominant cost.
SciPy does not expose a reliable count of internal dense factorizations, so
`njev` and actual Jacobian assemblies are reported rather than inventing a
factorization count.

## Candidate A: gate-aware termination

The experimental stop required all unchanged physical residual and ledger
gates plus a declared packed-primitive iterate-update tolerance.  At an update
tolerance of `5e-6`, it accepted at:

```text
wall time                         167.66 s
nfev                              512
Jacobian assemblies               507
maximum scaled residual           9.9848e-9
maximum storage-ledger defect     2.7851e-16
last packed iterate update        2.2022e-6
speedup                           1.18x
```

An update tolerance of `1e-8` never activated and reproduced the 600-call
reference.  The `5e-6` result also stopped at an earlier thermal iterate: its
maximum `d ln T` over the physical step was `0.0032640`, compared with
`0.0034579` for the reference.  Because it missed the preferred `1.5x` speed
gate and did not first pass the stricter temporal-error equivalence gate, it
was not promoted.

## Candidate B: independent blocked columns

The second candidate retained every serial sparse-forward finite-difference
column and evaluated independent columns concurrently.  It did not use the
rejected colored perturbation architecture.  A small-mesh certification found
the blocked and serial Jacobians and accepted primitive states equal to
`1e-12`.

On the production N64 checkpoint with four workers, however, it took
`177.94 s` to reach the same gate-terminated state as Candidate A.  Thread
overhead and contention made it slower than serial Candidate A and only
`1.11x` faster than the fully converged reference.  It was removed.

## Bounded stop

The candidates failed the coarse-mesh material-speed gate.  WP2 therefore did
not spend the much larger cost required to repeat rejected candidates at N128
or to run full-step/two-half-step temporal certification.  This is the declared
bounded stop, not an inference that faster solvers are impossible.

No third optimization architecture, colored Jacobian, tolerance relaxation,
or residual-weight scan is authorized in the current campaign.

## Retained implementation

Every backward-Euler result now records:

```text
Jacobian mode and termination reason
total residual evaluations
actual Jacobian assemblies
SciPy-reported nfev and njev
residual wall time
Jacobian-inclusive wall time when measurable
total nonlinear wall time
last accepted Jacobian-iterate update
```

Adaptive attempt histories preserve the same audit.  These measurements do
not alter the nonlinear trajectory.

## Next work package

Proceed to WP3: generate exact-common-physical-time N64/N96/N128 milestones
and the fixed-radius/full-flux/sonic/Roche comparison.  Continue subsequent
physics controls with the certified serial sparse-forward backend.
