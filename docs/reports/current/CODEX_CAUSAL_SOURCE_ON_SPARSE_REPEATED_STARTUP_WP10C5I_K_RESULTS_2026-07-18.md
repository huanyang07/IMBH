# Causal Source-On Sparse Repeated Startup WP10c5i-k Results

Date: 2026-07-18

## Verdict

The complete five-field causal DAE now passes the locked exact-stream
source-on startup, practical sparse-Jacobian parity, adaptive repeated-step,
restart, conservation, and N16/N32 equal-physical-time gates.

```text
WP10c5i exact-stream N16/N32 startup        PASSED
WP10c5i full/two-half temporal comparison  PASSED
WP10c5j local sparse Jacobian parity        PASSED
WP10c5k N16 repeated adaptive startup       PASSED
WP10c5k N32 equal-time startup              PASSED
WP10c5k baseline-subtracted mesh response   PASSED
long evolution                              NOT AUTHORIZED
tide, wind, stability, hot state, cycle     NOT AUTHORIZED
```

This is a numerical startup milestone. It is not a relaxed disk, a physical
stream-impact calculation, a stability result, a hot advective state, or a
limit cycle.

## Locked Physical Context

The source-on audit adds the exact compact-C2 cell moments already provided by
the Kerr-Schild migration layer:

```text
source center                  240 rg
logarithmic half-width         0.08
rest-mass supply               5 Mdot_Edd
source normalization defect    0
active cells                   2 at N16 and N32
outer Roche channel            closed
tide                           off
wind                           off
```

The injected primitive is an exact circularized regression stream with
surface density `1e5 g cm^-2` and temperature `1e6 K`. It is deliberately
labelled as a regression source, not a ballistic Layer-1 calibration. Mass,
radial momentum, angular momentum, and Killing energy all come from the same
immutable injected four-state.

## WP10c5i: Source-On Startup

The source is inserted without changing the `15N+5` DAE count, physical
equations, boundary contract, tolerances, or active set.

| Quantity | N16 | N32 |
|---|---:|---:|
| Unknowns/residuals | `245` | `485` |
| Equilibrated rank | `245/245` | `485/485` |
| Timestep (s) | `1.56892e-8` | `2.27952e-8` |
| Maximum scaled residual | `8.79746e-9` | `3.64410e-9` |
| Maximum algebraic residual | `4.50e-15` | `2.53e-15` |
| Minimum scattering depth | `1.70e4` | `1.70e4` |
| Full/two-half relative error | `5.62945e-6` | `1.20969e-6` |

Both full steps and all four independently solved half steps preserve the
closed Roche active set and pass the unchanged nonlinear, algebraic,
optical-depth, and conservation gates.

## WP10c5j: Practical Jacobian Backend

The complete residual has an exact block-local sparsity pattern with
nearest-neighbor coupling. A deterministic graph coloring requires 18 colors
at N4, N16, N32, and N64. Central finite differences therefore require 36
residual evaluations per Jacobian, independent of mesh size.

| Quantity | N16 | N32 |
|---|---:|---:|
| Dense central evaluations | `490` | `970` |
| Colored central evaluations | `36` | `36` |
| Evaluation reduction | `13.61x` | `26.94x` |
| Pattern nonzeros | `2755` | `5555` |
| Structural rank | `245/245` | `485/485` |
| Omitted derivative maximum | `0` | `0` |
| Colored/dense matrix defect | `0` | `0` |
| Directional relative defect | `7.09e-16` | `5.89e-16` |
| Sparse/dense root defect | `9.20e-18` | `1.41e-19` |

The sparse Newton path uses max-norm row/column equilibration followed by
SuperLU with COLAMD ordering. It reproduces the dense accepted roots and
accept/reject decisions without modifying the equations.

## WP10c5k: Adaptive Repeated Startup

The reusable evolution layer provides:

- increment-primary backward Euler;
- colored sparse Jacobian assembly;
- equilibrated sparse direct solves;
- deterministic reject, halve, and grow control;
- exact nonlinear, algebraic, optical-depth, conservation, and active-set
  gates;
- checksummed restart files containing the complete state, previous physical
  increment, timestep history, counters, and provenance.

N16 runs eight accepted steps. Its midpoint is saved and reloaded, and the
next accepted step is reproduced bit for bit from the in-memory and restored
states. The final checkpoint also round-trips bit for bit.

N32 is then evolved to the exact N16 elapsed physical time rather than to the
same step count.

| Quantity | N16 | N32 |
|---|---:|---:|
| Accepted steps | `8` | `7` |
| Rejected attempts | `0` | `0` |
| Elapsed time (s) | `3.392784696e-7` | `3.392784696e-7` |
| Loading time (s) | `1.59054e6` | `1.65693e6` |
| Elapsed/loading | `2.13311e-13` | `2.04763e-13` |
| Mass-ledger defect | `6.32e-13` | `8.14e-12` |
| Restart final round trip | bitwise | bitwise |
| Midpoint resumed step | bitwise | not repeated |

Every accepted step stays below:

```text
maximum scaled residual             1e-8
maximum scaled algebraic residual   1e-10
maximum scaled primitive change     5e-4
maximum scaled total change         1e-3
conservation defect                 1e-10
```

The scattering depth remains near `1.70e4`, and the Roche channel remains
closed.

## Cancellation-Safe Mass Audit

The total disk mass is about `1e29 g`, while the short-run change is about
`2.55e21 g`. Direct subtraction of endpoint totals loses meaningful low bits:

```text
N16 endpoint-subtraction defect   2.32e-10
N32 endpoint-subtraction defect   9.91e-9
```

That subtraction is not used for acceptance. The DAE's primary conserved
increments are integrated cell by cell and summed across accepted steps:

\[
\Delta M_{\rm disk}
=
\sum_{n,i}{\cal V}_i\,\Delta U^D_{i,n}.
\]

This cancellation-safe quantity agrees with the time-integrated source and
boundary fluxes to `6.32e-13` at N16 and `8.14e-12` at N32. Endpoint
subtraction remains in the report only as a diagnostic.

## Mesh Diagnostic

The first attempted mesh decision compared the largest cell-centered `H/R`.
That is not a common physical observable for this preflight seed:

- the seed anchors its thermodynamic end states at the first and last cell
  centers so that the physical Hill/Roche boundary solve remains admissible;
- those centers move from `284.52 rg` at N16 to `308.73 rg` at N32;
- the raw final maxima consequently differ by `6.83%` before the source has
  time to change the disk appreciably.

Moving the seed anchor to the grid face was tested and rejected because it
changes the outer reservoir state and prevents the gas-radiation nozzle root
from converging. No boundary physics was changed to force a comparison.

The accepted mesh diagnostic instead reconstructs each mesh on the same
129-point log-radius grid and compares the baseline-subtracted response

\[
\Delta\ln(H/R)
=
\ln(H/R)_{\rm final}-\ln(H/R)_{\rm initial}.
\]

| Mesh quantity | Difference | Gate |
|---|---:|---:|
| Mass response per injected mass | `1.990e-2` | `5e-2` |
| Inner mass flux / supply | `2.616e-2` | `5e-2` |
| Outer mass flux / supply | `0` | `5e-2` |
| Maximum common-radius `Delta ln(H/R)` difference | `2.051e-3` | `5e-3` |
| RMS common-radius `Delta ln(H/R)` difference | `6.620e-4` | diagnostic |
| Raw cell-centered maximum `H/R` difference | `6.828e-2` | diagnostic |

The mesh gate therefore certifies the short-time response of the integrator,
not the arbitrary preflight seed as a continuum physical state.

## Dominant Remaining Limitation

The initial inner mass flux is approximately

\[
\frac{\dot M_{\rm inner}}{\dot M_{\rm stream}}
\simeq -9.19\times10^4.
\]

The sign follows the declared face-flux convention. Its magnitude shows that
the first `3.39e-7 s` are dominated by relaxation of the deliberately simple
preflight datum, not by physical stream loading. The injected mass is only
`2.78e16 g`, compared with a net disk-mass change of about `-2.55e21 g`.

This is why the passing startup gate does not authorize a longer physical
trajectory from the same datum.

## Classification

WP10c5i-k certifies:

- exact source moments in the complete causal DAE;
- N16/N32 source-on consistent initialization and bounded tiny steps;
- temporal full/two-half consistency;
- an exact local sparsity pattern and 18-color sparse Jacobian;
- sparse/dense matrix, directional, root, and accept/reject parity;
- repeated adaptive stepping at N16 and equal-time N32 stepping;
- cancellation-safe aggregate conservation;
- bitwise restart and resumed-step reproducibility;
- a baseline-subtracted early-response mesh gate.

It does not certify:

- a source-compatible or relaxed initial disk;
- ballistic stream impact or Layer-1 injection calibration;
- long-time no-tide loading;
- stability, a hot advective branch, or a limit cycle;
- a physical tide or wind.

## Locked Next Work

The next package must improve the physical initial-value problem before
extending duration.

1. Add a matched source-on/source-off causal control at N16 and N32 using the
   same accepted timestep history. Verify that the difference recovers the
   exact stream mass, angular momentum, and Killing energy moments.
2. Construct one constraint-consistent causal datum whose inner throughput is
   order unity relative to the `5 Mdot_Edd` supply, while retaining zero inner
   incoming characteristics, an optically thick column, a closed Roche edge,
   full rank, and exact primitive/face maps.
3. Reject the datum if those conditions cannot be met without floors,
   clipping, a boundary-mode change, or a relaxed residual gate.
4. Repeat WP10c5k from the selected source-compatible datum at N16/N32.
5. Only after that response and mesh gate may duration or resolution increase.
6. Keep distributed tide, wind, hot-state, stability, and cycle searches
   blocked until a practical no-tide loading trajectory exists.

## Reproduction

```text
PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-source-on-audit

PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-sparse-backend-audit

PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-repeated-source-on-audit
```

Machine-readable outputs and restart files are generated under ignored
`outputs/` in accordance with the artifact policy.

## Repository Verification

The completed implementation passed:

```text
focused causal DAE/evolution tests: 25 passed
full test suite:                     484 passed, 4 subtests passed
Python compilation:                  passed
git diff --check:                    passed
repository hygiene:                  passed
```

The twelve untracked review notes under `docs/reports/gpt/` were not modified
or included in this work package.
