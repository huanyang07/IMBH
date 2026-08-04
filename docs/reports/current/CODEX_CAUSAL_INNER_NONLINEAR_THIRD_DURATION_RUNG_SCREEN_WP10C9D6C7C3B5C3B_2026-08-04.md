# Third nonlinear duration-rung screen WP10c9d6c7c3b5c3b

## Classification

`third_rung_screen_certified_five_e_minus_three_completion_manifest_authorized`

The frozen `1e-3 -> 2e-3 s` fail-fast screen passes for the base and generic
five-field perturbed trajectories. This authorizes only a fresh definitions-
only manifest for completion of the third rung through `5e-3 s`.

The historical c2d failure remains unchanged. The corrected single-source
target construction is used here, and both new same-target replay branches
are bitwise reproducible.

## Frozen experiment executed

- The initial base and perturbed states and complete BDF2 histories are the
  committed canonical `1e-3 s` arrays from WP10c9d6c7c3b5c2e1.
- One integer-microsecond source defines every target used by the main,
  replay, and strict branches.
- Main branches advance from `1e-3` to `2e-3 s` at `dt <= 2e-4 s`.
- Serialized replay branches restart at `1.6e-3 s` and land on
  `1.6/1.8/2.0e-3 s`.
- Strict branches restart at `1.8e-3 s` and advance with
  `dt <= 1e-4 s` through `2e-3 s`.
- Every controller decision uses one full implicit BDF2 step and two
  independent half steps.

## Binding results

Both main branches accept all five comparisons on the first attempt. Both
replay and both strict branches accept their two comparisons on the first
attempt. No trajectory takes a rejected controller step.

| Quantity | Maximum measured value | Gate |
|---|---:|---:|
| Main local-error estimate | `1.2953795064e-8` | `<= 2.5e-4` |
| Main summed local error | `5.7176837942e-8` | `<= 5e-3` |
| Strict local-error estimate | `3.6079215099e-9` | `<= 3.125e-5` |
| Scaled nonlinear residual | `9.2260400250e-11` | `<= 1e-10` |
| Discrete-ledger defect | `0` | `= 0` |
| Tier-I export-ledger defect | `2.3017494887e-12` | inherited |
| Mapped endpoint/path closure | `3.5554446041e-12` | inherited |
| Incoming excision characteristics | `0` | `= 0` |

For both base and perturbed branches, serialized replay is bitwise identical
to the main branch in:

- target labels;
- primitive states;
- all 13 Tier-I exports;
- the serialized restart round trip.

The main-versus-strict response comparison passes by a wide margin:

| Response diagnostic | Measured | Gate |
|---|---:|---:|
| Maximum scaled state difference | `9.7210239858e-11` | `<= 5e-3` |
| Maximum scaled Tier-I difference | `2.3363998181e-11` | `<= 5e-3` |
| State-history cosine | `1.0` | `>= 0.90` |
| Tier-I history cosine | `0.999999999986` | `>= 0.90` |

Final readiness also passes. The maximum `H/R` is `0.0986422`, the minimum
scattering optical depth is `18.9605`, and the minimum reconstruction factor
is exactly `1`.

## Interpretation

The corrected canonical target contract remains stable over the first
additional millisecond of the third duration rung. There is no evidence in
this screen for restart drift, target-label drift, temporal-controller
failure, nonlinear residual loss, ledger loss, reconstruction clipping, or
an incoming excision mode.

This is still only a bounded screen. It does not certify the full `5e-3 s`
third rung, a cell-crossing-time truth trajectory, fixed-Q fast dynamics, or
a reduced slow model.

## Authorized next package

`WP10c9d6c7c3b5c3c_third_duration_rung_completion_manifest`

The next commit must be definitions only. It may freeze direct BDF2
continuation from the committed `2e-3 s` base and perturbed histories through
`5e-3 s`, with fresh replay/strict checkpoints, unchanged method/readiness
gates, and a fail-fast execution order. It must not propagate the completion
trajectory in the manifest commit.

Fixed-Q experiments and reduced slow evolution remain blocked.
