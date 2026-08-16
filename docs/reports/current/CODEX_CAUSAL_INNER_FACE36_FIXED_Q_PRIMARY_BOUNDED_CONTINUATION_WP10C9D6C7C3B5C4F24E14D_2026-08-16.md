# Primary Fixed-Q Bounded-Continuation Result WP10c9d6c7c3b5c4f24e14d

## Classification

`bounded_continuation_failed`

The frozen primary continuation pilot stopped at the first warm root. The
initial cold BDF2 continuation root passed every declared numerical, history,
physical, reaction, and checkpoint gate. The next root, initialized from the
carried raw-coordinate Broyden matrix with no forced exact assembly, reached a
scaled residual of `5.708109263036221e-9` after the frozen eight-iteration
budget. This exceeds the unchanged `1e-10` root gate.

The warm candidate was rejected and never entered continuation history. No
later warm root, suffix replay, same-history cold shadow, or matched-endpoint
half-step control was executed. The accepted trajectory therefore advances by
one new BDF2 step, or `1e-7 s`; two roots were attempted.

This is a binding rejection of the frozen cross-step matrix-reuse policy. It
is not evidence of a physical failure of the fixed-Q equations and does not
invalidate the certified local BDF1-to-BDF2 history ladder.

## Frozen execution identity

```text
execution commit          2dc3b89f2f1dea968c2f77e3bcc9a39841aac069
execution tree            2493ee21faafa6292fe22b61415cc4b3ca9bccfa
manifest contract SHA256  2f45b04fef03123f3d58108a2c693616f4c7284f8ced5a0a785f3b5b4673d62d
continuation seed SHA256  929f844ecd1dba520bcdffdeab4e8876c5842d536032ca8cb2d77bfe609cd653
```

BLAS and OpenMP thread counts were pinned to one. The runner and focused test
sources were committed before execution and are recorded by hash in the
canonical package.

## Cold continuation root

`cold_1` used the prospectively declared cold policy: one initial exact
complete bordered matrix and at most one exact refresh after complete
line-search failure.

It passed with:

```text
maximum scaled residual                 4.737492683089679e-13
maximum Q3 relative defect              1.430243688090699e-16
maximum storage-parity defect           1.941506911043733e-14
maximum reaction-ledger defect          3.6127335177262564e-22
maximum constraint-action defect        1.3322929791812172e-16
raw Schur condition number              3.418196118024292e4
raw Schur rank                          3
minimum reconstruction factor           1.0
maximum reconstruction factor           1.0
maximum H/R                             0.09783748666878898
minimum scattering optical depth        19.254315793914518
maximum scaled primitive change         0.004409792597730185
incoming excision characteristics       0
```

The root required 19 residual evaluations, seven Newton iterations, six
Broyden updates, and two exact matrix assemblies. Its stale matrix stalled at
`2.0075607842784393e-10`; all 12 frozen line-search trials then failed, so the
authorized `line_search_failure` refresh was taken. The refreshed full step
reduced the residual to `4.737492683089679e-13`.

The arbitrary-BDF2 checkpoint round-tripped bitwise. Its serialized size is
`1,314,743` bytes and its SHA256 is
`e33ac8f1b71a5c7a807dbc37eded55874dd0a8d6d4105a835e46b94ccde34810`.

## Warm-root failure

`warm_1` used the accepted `cold_1` history and carried solver matrix. It
correctly performed no exact assembly at iteration zero. Its residual history
contracted as follows:

```text
2.4158857789359516
7.972685459471510e-2
4.654474565745483e-3
1.3080992014727066e-3
4.789703177979221e-5
9.797465829575192e-6
3.203294013376379e-8
9.278950857671830e-9
5.708109263036221e-9
```

The final two iterations accepted half and quarter steps. Because each line
search found a merit-decreasing step, the frozen refresh trigger—complete
line-search failure—never fired. The solver then exhausted its eight-iteration
budget with zero exact assemblies and a residual margin of `57.0811` relative
to the binding tolerance.

Only `nonlinear_root` and `complete_residual` failed. The rejected endpoint
still passed every non-root audit:

```text
maximum Q3 relative defect              5.588616363475395e-16
maximum storage-parity defect           2.270974296952425e-14
maximum reaction-ledger defect          1.827554837285793e-22
maximum constraint-action defect        1.3322943285040861e-16
raw Schur condition number              3.435144785740073e4
raw Schur rank                          3
minimum/maximum reconstruction factor   1.0 / 1.0
maximum H/R                             0.09783748204272863
minimum scattering optical depth        19.25431742390132
maximum scaled primitive change         0.004279816508184186
incoming excision characteristics       0
```

This pattern localizes the binding failure to nonlinear convergence under the
declared carried-matrix iteration/refresh contract. It does not select a
constraint, storage, reaction, admissibility, or excision failure.

## Cost evidence

The two attempted roots consumed about `4399.62 s` (`73.33 min`) of root wall
time:

| Root | Wall time | Residual evaluations | Exact assemblies | Bordered solves |
|---|---:|---:|---:|---:|
| `cold_1` | `2727.12 s` | 19 | 2 | 7 |
| `warm_1` | `1672.49 s` | 12 | 0 | 8 |

The profiling clocks are nested, but they make the dominant cost clear.
`cold_1` records `2187.68 s` in line-search residual evaluation and
`1936.10 s` in monolithic residual work, compared with `212.47 s` in exact
Jacobian work. `warm_1` records `1344.86 s` in line-search residual evaluation
and `1284.25 s` in monolithic residual work, while all bordered linear solves
together cost only `0.027 s`.

The pilot did not reach the same-history cold shadow, so the frozen warm/cold
cost ratio is unevaluable. No cost or reuse pass can be claimed.

## Scientific interpretation

The result preserves three distinct conclusions:

1. The accepted cold root extends the certified fixed-Q solution by one
   authentic BDF2 step with complete physical and history closure.
2. The carried Broyden matrix remains useful enough to reduce the next root by
   more than eight orders of magnitude, but it does not meet the unchanged
   root gate within the frozen warm budget.
3. A refresh policy triggered only by complete line-search failure is
   insufficient here: late damped steps continue to reduce the merit and
   therefore suppress the only permitted refresh trigger.

The existing fixed-Q residual, reaction construction, local timestep
certificate, and physical admissibility remain supported. The failed item is
the proposed operational policy for repeated warm continuation.

## Next plan

The next artifact should be definitions-only and should preserve this binding
rejection. It should authorize no additional trajectory until reviewed.

1. Analyze the committed `cold_1` matrix, `warm_1` event trace, and rejected
   endpoint without advancing state.
2. Prospectively define one nonpropagating exact-Jacobian diagnostic at the
   saved `warm_1` endpoint. Its purpose is to distinguish stale carried-matrix
   information from an eight-iteration budget limitation; it may not convert
   this pilot into a pass.
3. Replace the line-failure-only refresh trigger in a future candidate with a
   prospectively frozen stagnation or iteration-reserve trigger, or derive and
   audit a cross-step temporal-block update. Do not relax the `1e-10` residual
   gate retrospectively.
4. Optimize the measured bottleneck before another four-root pilot. Prioritize
   repeated monolithic residual, reaction, and descriptor work; dense bordered
   solves are negligible at this size.
5. Only after a new solver-policy certificate should the complete primary
   four-root continuation be rerun. Held-out continuation, operational
   timestep search, physical microbursts, fast averaging, and reduced slow
   evolution remain blocked.

## Canonical evidence

```text
results/canonical/
causal_inner_face36_fixed_q_primary_bounded_continuation_
wp10c9d6c7c3b5c4f24e14d/
```

The package contains the cold accepted checkpoint, both attempted-root arrays,
complete solver traces and profiling, execution identity, provenance,
machine-readable classification, catalog records, and closing SHA256 checksums.
