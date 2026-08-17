# Fixed-Q Operational-Timestep Predictor Manifest WP10c9d6c7c3b5c4f24e14w

## Classification

`operational_timestep_predictor_repair_manifest_frozen_execution_authorized`

This definitions-only package supersedes e14u before any nonlinear root. It
preserves the same primary state, `h=2e-7 s` variable-step BDF2 rung, certified
two-step `h=1e-7 s` reference, bitwise replay, and every binding scientific
gate.

The only execution-policy change is explicit predictor selection:

```text
initial scaled increment              previous accepted scaled increment
maximum predictor component           4.548844155366097e-3
unchanged primitive-change bound       5.000000000000000e-3
last-rate extrapolation                 forbidden for this rung
```

The solver rate argument is constructed algebraically so that the variable-
step BDF initialization exactly reproduces that accepted previous increment.
The root remains free to move under Newton corrections, subject to the
unchanged primitive-change and line-search bounds.

A pass authorizes only a definitions-only `h=4e-7 s` manifest. A root,
matched-endpoint, replay, or physical-gate failure stops the study.

The focused post-freeze suite passes `6/6` with one prospective result skip;
all canonical checksums close.
