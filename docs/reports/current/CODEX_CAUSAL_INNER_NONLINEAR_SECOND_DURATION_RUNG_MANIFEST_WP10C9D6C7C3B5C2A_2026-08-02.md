# Second nonlinear duration-rung manifest WP10c9d6c7c3b5c2a

## Classification

`second_nonlinear_duration_rung_manifest_frozen_one_e_minus_three_second_propagation_authorized`

This definitions-only package freezes continuation from the certified first-rung BDF2 history. It changes no operator or production default and propagates no state.

## Frozen experiment

- committed `p3_buffer45__generic_five_field` base/response history through `2e-4 s`
- continuation to `1e-3 s`; common outputs every `1e-4 s`
- no new BDF1 startup; committed `1.8e-4/2e-4 s` history
- maximum main step `1e-4 s`, reached only through ratio `<=2`
- restart/replay from `6e-4 s`
- strict `dt<=5e-5 s` shadow over `8e-4-1e-3 s`
- main local/summed error budgets remain `2.5e-4` / `5e-3`
- correct active coupling face: `48`

## Scope

The horizon is about `0.1804` of one N128 cell-crossing time. A pass authorizes only the definitions-only `5e-3 s` third-rung manifest. Fixed-Q and reduced evolution remain blocked.
