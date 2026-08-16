# Fixed-Q Bounded Continuation and Cost Manifest WP10c9d6c7c3b5c4f24e14a

## Classification

`fixed_Q_bounded_continuation_cost_manifest_frozen_implementation_preflight_authorized`

This package is definitions-only. It advances no state and preserves the
certified adaptive-refresh BDF1-to-BDF2 ladder unchanged. It authorizes only
an implementation preflight for restartable BDF2 continuation state,
cross-step nonlinear-solver state, reaction-coordinate rebasing, and cost
instrumentation.

The bounded continuation pilot, held-out continuation, operational-timestep
study, fixed-Q micro-solver, physical microburst, fast averaging, and reduced
slow evolution remain unauthorized.

## Parent authorization semantics

The parent canonical package contains two differently scoped records:

- `contract.json` is the input contract that existed before the refined
  ladder and therefore has `one_Q_execution_manifest_authorized=false`;
- `summary.json` is the post-ladder decision and has
  `one_Q_execution_manifest_authorized=true`.

These records are not scientifically contradictory. This manifest assigns
their roles explicitly and freezes both paths and hashes in
`parent_authorization.json`. No parent artifact is rewritten.

## Seed sufficiency finding

The hash-validated primary coarse artifact contains the accepted BDF1 and
BDF2 primitive roots, increments, rates, multipliers, and physical reaction
action. It does not contain the complete mapped-storage and
responsive-height path histories as first-class canonical arrays.

Those histories are required for authentic continuation because the
responsive-height temporal contribution is a path-dependent one-form. Before
execution, the implementation package must do one of the following:

1. recompute the complete storage histories by evaluating the declared
   straight primitive path on the exact canonical BDF1/BDF2 states, freeze
   the result, and certify its residual/history consistency; or
2. rerun one authentic coarse BDF1-to-BDF2 startup and canonicalize a complete
   continuation checkpoint.

Synthetic, projected, endpoint-inferred without the declared path, or
manually corrected histories are forbidden.

## Continuation-state contract

The implementation preflight must introduce a continuation payload valid
after any fully accepted BDF1 or BDF2 step. It must persist:

- current and previous primitive states;
- complete mapped-storage and responsive-height interval histories;
- current and previous timesteps;
- fixed Q3 target and constraint scales;
- multiplier predictor;
- reaction basis and transform;
- elapsed time and accepted-step count;
- final bordered Broyden matrix and its anchor metadata;
- source, configuration, and parent-artifact hashes.

Rejected steps may never define continuation history. Save/load equality must
be bitwise.

## Reaction-coordinate contract

The current solver uses a state-local frozen-normalized reaction basis.
Therefore a carried bordered matrix cannot silently retain the preceding
step's multiplier coordinates.

The serialized nonlinear state must use raw reaction-channel coordinates.
For

```text
mu = T lambda,
```

the next step must rebase the multiplier predictor and multiplier columns as

```text
lambda_new = inverse(T_new) T_old lambda_old,

J_lambda_new = J_lambda_old inverse(T_old) T_new.
```

Physical reaction action invariance is binding. Direct equality of multiplier
coordinates is diagnostic only.

## Prospective primary pilot

After a separate implementation certificate, the intended physical pilot is
frozen as follows:

- middle layout and primary 20 ms state;
- fixed Q3 and `h=1e-7 s`;
- one cold BDF2 continuation root;
- three warm BDF2 roots using the carried final matrix;
- checkpoint after every accepted endpoint;
- restart before the second warm root and bitwise replay of the final
  two-step suffix;
- one non-propagating cold shadow at the second warm root, using identical
  input state and history;
- one conditional matched-endpoint full-step versus two-half-step audit.

The four main roots span only `4e-7 s`. This is an infrastructure and cost
pilot, not a fast-relaxation, attraction, orbit, or averaging experiment.

The cold root requires an initial exact complete matrix and may use one
additional line-failure refresh. Warm roots do not force an exact matrix at
iteration zero and may use at most one exact refresh after exhausting the
complete line search.

## Inherited scientific gates

Every accepted main, replay, cold-shadow, and half-step root retains:

```text
maximum scaled residual                    <= 1e-10
maximum Q3 relative defect                 <= 1e-12
maximum ledger relative defect             <= 1e-12
maximum storage parity defect              <= 1e-9
minimum reconstruction factor              >= 1 - 1e-12
maximum reconstruction factor              <= 1 + 1e-12
raw Schur rank                              = 3
raw Schur condition number                  <= 1e8
maximum H/R                                 <= 0.12
minimum scattering optical depth           >= 1
maximum scaled primitive change             <= 5e-3
incoming excision characteristics           = 0
```

The four-step cumulative absolute ledger budget is `4e-12`.

## Cost and equivalence gates

Cost is assessed against the same-history cold shadow, not against a root at a
different physical state. The prospective reuse gates are:

- warm/cold scaled-state difference at most `1e-8`;
- warm/cold physical-reaction-action relative difference at most `1e-8`;
- at least two of three warm roots require no exact refresh;
- same-history warm/cold wall-time ratio at most `0.75`;
- half-step state disagreement at most `0.1` of the full-step change;
- half-step physical-reaction-action relative disagreement at most `0.1`.

Multiplier-coordinate agreement is nonbinding. Wall time is an engineering
gate and cannot change the scientific classification of a valid root.

## Profiling contract

The implementation must record wall and process time for residual evaluation,
reaction construction, descriptor assembly and sparse LU, complete Jacobian
assembly, bordered solve, line-search residuals, physical acceptance, and
checkpoint I/O. It must also record residual evaluations, exact assemblies,
Broyden updates, failed line-search trials, matrix age, residual margin, and
checkpoint bytes.

## Decision branches

`bounded_continuation_and_reuse_passed`

: All scientific, history, restart, replay, equivalence, half-step, and cost
  gates pass.

`bounded_continuation_valid_cost_failed`

: The physical and numerical continuation passes, but matrix reuse or cost
  gates fail. This blocks a physical microburst pending solver optimization;
  it is not a fixed-Q equation failure.

`bounded_continuation_failed`

: Any residual, history, Q3, storage, physical, ledger, restart, replay, or
  warm/cold equivalence gate fails.

Only a future reviewed execution package may assign one of these
classifications.

## Canonical evidence

The prospective machine-readable package is stored under

```text
results/canonical/
causal_inner_face36_fixed_q_bounded_continuation_cost_manifest_
wp10c9d6c7c3b5c4f24e14a/
```

It includes the execution manifest, explicit parent-authorization roles,
seed inventory, provenance, summary, and closing SHA256 checksums.

## Next authorized work

Implement and test only the continuation/restart/solver-state infrastructure.
Do not advance a committed physical state during that preflight. A passing
implementation certificate may authorize the bounded primary pilot defined
above; it may not directly authorize held-out propagation, a microburst,
averaging, or reduced slow evolution.
