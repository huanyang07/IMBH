# Third nonlinear duration-rung spatial-confirmation manifest WP10c9d6c7c3b5c3g

## Classification

`third_duration_rung_spatial_confirmation_manifest_frozen_middle_fine_generic_propagation_authorized`

This definitions-only package freezes the middle/fine generic-five-field spatial confirmation at `5e-3 s`. It propagates no state and changes no operator or production default.

The coarse c3d base/perturbed result is reused by hash. New middle and fine base/perturbed trajectories must use their layout-native committed short-horizon histories, the one canonical target source, correct active coupling faces `96/192`, serialized replay, and strict final-interval shadows.

## Frozen experiment

The binding profile is the prospectively declared full-amplitude
`p3_buffer45__generic_five_field` perturbation. The layouts are:

| Layout | Cells | Inner refinement | Active coupling face |
|---|---:|---:|---:|
| Coarse | 64 | 1 | 48 |
| Middle | 112 | 2 | 96 |
| Fine | 208 | 4 | 192 |

The committed coarse c3d base and perturbed main/replay/strict histories are
immutable reference evidence. The new middle and fine trajectories start
from the committed layout-native base and generic states at `30/40 us`, with
the previous timestep `1e-5 s`; mapped and responsive-height histories are
reconstructed deterministically from those primitive states. There is no new
BDF1 startup.

One integer-microsecond source defines all output labels:

```text
main    2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.0 ms
replay  4.4, 4.8, 5.0 ms
strict  4.8, 4.9, 5.0 ms
```

The main controller begins at `5e-6 s`, grows by no more than two, and is
capped at `4e-4 s`. The strict final interval is capped at `1e-4 s`.
Independent target-array construction is forbidden.

The fail-fast order is middle base, middle perturbed, fine base, fine
perturbed, state analysis, instantaneous Tier-I analysis, then
windowed-cumulative Tier-I analysis. Every base/perturbed trajectory must
complete main, bitwise replay, and strict-shadow gates before a durable cache
is accepted. The middle layout must finish before the fine layout begins.

State responses are conservatively restricted to the common 64-cell parent. Instantaneous and windowed-cumulative 13-component Tier-I responses use the inherited `0.75/0.05/0.90` spatial gates. Strict temporal uncertainty must be no more than ten percent of an observable middle/fine spatial difference.

## Frozen measurements and gates

The state observable is the perturbed-minus-independently-evolved-base
response after conservative restriction of every layout to the common
64-cell parent.

The 13 Tier-I observables retain fixed physical scales. The interface M/J/E
flux must be evaluated at the layout's actual active face, never parent face
48 on the middle or fine layout. Both instantaneous histories and a
trapezoidal cumulative response over the common `2-5 ms` window are binding;
the latter is explicitly not described as a zero-to-horizon cumulative
export.

For state, instantaneous exports, and windowed cumulative exports, the
prospective spatial gates are:

```text
RMS order                         >= 0.75
maximum order                     >= 0.75
significant-component order       >= 0.75
fine normalized difference        <= 0.05
history cosine                    >= 0.90
refinement-error cosine           >= 0.90
relative activity                 >= 1e-8
```

Temporal uncertainty is the conservative sum of the base and perturbed
main-versus-strict response differences. For an observable middle/fine
spatial difference, that envelope must be no more than ten percent of the
difference. A refinement error below five times the frozen uncertainty is
reported only as an upper bound; its nominal order and direction are not
accepted as evidence, and the upper bound must remain below the fine-
difference gate.

The unchanged method gates include residual `<=1e-10`, discrete-ledger
defect `<=1e-12`, local/summed controller errors `<=2.5e-4/5e-3`, exact
reconstruction, optical depth `>=1`, `H/R<=0.12`, no incoming excision mode,
and bitwise replay.

## Cost and staging

The committed coarse base-plus-perturbed completion took about `6.17 h`.
Allowing for the longer `40 us -> 5 ms` middle/fine continuation and their
larger cell counts gives optimistic planning estimates near `21.6 h` for the
middle pair and `40.1 h` for the fine pair. These are scheduling estimates,
not scientific gates. Durable trajectory caches and middle-before-fine
execution prevent a late failure from discarding completed evidence.

Authorized next: `WP10c9d6c7c3b5c3h_third_duration_rung_spatial_confirmation`.

The `2e-2 s` fourth rung, fixed-Q experiments, reduced slow evolution, tide, wind, production promotion, and N1024 remain blocked.

## Decision logic

- If the middle and fine trajectories, replay, strict uncertainty, state,
  instantaneous Tier-I, and windowed-cumulative Tier-I gates all pass, only a
  definitions-only `2e-2 s` fourth-rung manifest is authorized.
- If a trajectory or method gate fails, stop at that layout and localize it.
- If state or exports fail spatially, separate the state, correct-face export,
  and temporal-uncertainty contributions before considering any numerical
  change.
- No operator redesign is authorized without a stable, observable,
  noncontracting mechanism.

The historical c2d and b4b3 failures remain unchanged. This manifest defines
a new longer-horizon generic experiment; it does not relabel either result.

## Reproducibility

Canonical configuration, frozen manifest, provenance, summary, and hashes
are stored under
`results/canonical/causal_inner_nonlinear_third_duration_rung_spatial_confirmation_manifest_wp10c9d6c7c3b5c3g/`.
The focused evidence suite reports `4 passed`.
