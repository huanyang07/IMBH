# Fixed-Q Continuation Implementation Preflight WP10c9d6c7c3b5c4f24e14b

## Classification

`fixed_Q_continuation_implementation_preflight_certified_primary_pilot_manifest_authorized`

The bounded fixed-Q continuation infrastructure is implemented and certified
for its preflight scope. No physical state was advanced and no nonlinear root
was solved by this work package.

This result authorizes only a definitions-only primary bounded-continuation
execution manifest. It does not authorize the continuation run itself, a
held-out continuation, a fixed-Q micro-solver, a physical microburst, fast
averaging, or reduced slow evolution.

## Implemented contracts

The fixed-Q method now exposes a continuation payload valid after any fully
accepted BDF1 or BDF2 step. The payload preserves:

- current and previous primitive states;
- the complete mapped-storage and responsive-height interval history;
- the fixed Q3 target and constraint scales;
- the next-step reaction basis and transform;
- a raw-reaction-coordinate multiplier predictor;
- elapsed time, accepted-step count, and BDF order;
- an optional carried bordered Broyden matrix with anchor metadata;
- deterministic provenance.

Rejected steps fail closed and cannot define continuation history.

The final bordered secant matrix is serialized in raw reaction-channel
coordinates. At the next frozen-normalized step, both its multiplier columns
and the multiplier predictor are rebased through the new state-local reaction
transform. Exact anchor, target, constraint-scale, and state/rate-scale
compatibility is required before reuse.

Warm equal-step BDF2 roots may use the carried matrix without forcing an exact
assembly at iteration zero. The existing line-search-failure refresh remains
available and unchanged. Cold solve behavior and production defaults are
preserved.

## Fixture-level certification

The focused implementation test constructs accepted BDF1 and BDF2 fixture
roots, serializes the arbitrary-BDF2 continuation state, reloads it bitwise,
rebases the solver state, and executes a warm BDF2 fixture correction without
an initial exact Jacobian assembly.

The test also proves that:

- a rejected result cannot enter continuation history;
- a one-bit change to the solver anchor invalidates the compatibility hashes;
- the carried multiplier action closes after reaction-coordinate rebasing;
- the warm fixture uses zero exact assemblies and advances the matrix age;
- checkpoint read/write and root-cost timing counters are populated.

The pinned focused suite reports:

```text
21 passed, 1 skipped in 199.57 s
```

The skip is the prospective committed-artifact closure test; it becomes active
when this canonical package is committed.

## Canonical seed-history reconstruction

The primary 20 ms, `h=1e-7 s` coarse BDF1-to-BDF2 evidence from
WP10c9d6c7c3b5c4f24e11 is reused by hash. The preflight does not call the
fixed-Q nonlinear solver.

Instead, it re-evaluates the declared straight primitive paths on the exact
canonical BDF1 and BDF2 endpoints and reconstructs the missing complete
mapped-storage and responsive-height histories. The results are:

```text
BDF1 primitive history bitwise                 true
BDF2 primitive history bitwise                 true
BDF1 augmented residual bitwise                true
BDF2 augmented residual bitwise                true
maximum BDF1 residual reconstruction defect    0
maximum BDF2 residual reconstruction defect    0
minimum path reconstruction factor             1
complete mapped history finite                 true
complete responsive-height history finite      true
continuation checkpoint roundtrip bitwise       true
```

The canonical continuation checkpoint is 25,050 bytes. It deliberately has no
carried nonlinear matrix because the historical seed artifact did not preserve
that matrix. The prospective execution manifest must therefore begin with the
declared cold BDF2 root; it may not invent or infer a historical solver matrix.

## Profiling infrastructure

Each fixed-Q root can now report separate wall-time counters for:

- complete residual evaluations;
- monolithic residual evaluation;
- reaction construction;
- descriptor assembly;
- descriptor sparse LU;
- exact complete Jacobian assembly;
- bordered linear solves;
- line-search residual evaluations;
- physical acceptance checks.

Checkpoint read and write timings are recorded separately by the continuation
serializer. Existing counts for residual evaluations, exact assemblies,
Broyden updates, and linear solves remain available.

## Provenance

The implementation preflight ran from clean tracked commit
`041662f61f84d3653cecd383400f2682fb09d641` with BLAS and OpenMP thread counts
pinned to one, matching the canonical seed environment. The canonical package
contains source, parent-artifact, and seed-artifact hashes plus closing SHA256
checksums.

## Limitations

This certificate proves infrastructure and exact seed-history recovery only.
It does not prove:

- that a physical BDF2 continuation root will pass;
- that a carried matrix will remain an effective nonlinear model;
- that warm continuation is cheaper than a same-history cold solve;
- arbitrary multi-step physical stability;
- a useful operational timestep;
- fixed-Q attraction, mixing, averaging, or a slow closure.

The fixture tolerance is intentionally loose and is not a replacement for the
unchanged `1e-10` physical root gate.

## Next authorized work

Freeze a definitions-only primary bounded-continuation execution manifest that
binds the already declared e14a experiment:

1. one cold BDF2 continuation root from the canonical seed;
2. three warm equal-step BDF2 roots with carried-matrix reuse;
3. arbitrary-BDF2 checkpointing and bitwise two-step suffix replay;
4. one same-history cold shadow;
5. the conditional full-step versus two-half-step audit;
6. all inherited physical, storage, reaction, ledger, and excision gates;
7. separate scientific-validity and cost classifications.

Do not execute that pilot until its prospective manifest is reviewed and
committed.

## Canonical evidence

```text
results/canonical/
causal_inner_face36_fixed_q_continuation_implementation_preflight_
wp10c9d6c7c3b5c4f24e14b/
```

The package contains the reconstructed continuation checkpoint,
implementation contract, seed metrics, focused-test output, provenance,
summary, and closing checksums.
