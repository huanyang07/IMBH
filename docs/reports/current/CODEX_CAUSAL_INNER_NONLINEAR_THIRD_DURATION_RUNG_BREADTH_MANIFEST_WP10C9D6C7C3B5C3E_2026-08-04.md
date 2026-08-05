# Third nonlinear duration-rung breadth manifest WP10c9d6c7c3b5c3e

## Classification

`third_duration_rung_breadth_manifest_frozen_coarse_heldout_duration_screen_authorized`

This definitions-only package freezes four coarse held-out duration trajectories through `5e-3 s` and the later generic middle/fine spatial-confirmation scope. It propagates no state.

Held-outs: `p4__inward_acoustic, p4__outward_acoustic, p3_buffer45__material, p4__inward_shear_acoustic_mix`.

## Coarse held-out stage

The four profiles retain their original, prospectively frozen definitions,
full-positive amplitude, and fail-fast order.  Each perturbed trajectory
starts from the committed WP10c9d6c7c3b4b2 `dt=2.5e-6 s` states at
`37.5/40 us`.  The mapped and responsive-height BDF histories are rebuilt
deterministically from those committed primitive states; no new BDF1 startup
or linear-basis recombination is permitted.

The certified c3d base main/replay/strict trajectory is reused by hash.  Each
new profile uses one tangent and one process, saves a durable cache only after
its complete main/replay/strict stage passes, and stops the campaign on the
first failure.

The one canonical output source is inherited from c3c:

```text
main    2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.0 ms
replay  4.4, 4.8, 5.0 ms
strict  4.8, 4.9, 5.0 ms
```

The main controller begins at `5e-6 s`, grows by at most a factor of two,
and is capped at `4e-4 s`.  The strict shadow is capped at `1e-4 s`.
Nonlinear residual, error-control, storage/path, ledger, readiness, outgoing
excision, bitwise replay, and main-versus-strict response gates remain
unchanged.

## Later spatial scope

The generic five-field spatial confirmation is frozen in outline but not
authorized for propagation.  It must compare the conservatively restricted
state response and all 13 Tier-I exports across the three embedded layouts,
using the correct active coupling faces `48/96/192`.  Middle/fine replay and
strict shadows are required, and a fresh definitions-only spatial manifest
must be committed after all coarse held-outs pass.

The inherited spatial gates remain `p >= 0.75`, fine normalized difference
`<= 0.05`, and history/refinement-error cosine `>= 0.90` for observable
channels.  Strict temporal uncertainty must be no more than ten percent of
the observable medium/fine spatial error.

Authorized next: `WP10c9d6c7c3b5c3f_coarse_heldout_third_duration_rung_screen`.

Middle/fine propagation, the `2e-2 s` rung, fixed-Q experiments, and reduced slow evolution remain blocked.

## Decision logic

- If all four coarse held-outs pass, authorize only a definitions-only
  middle/fine generic spatial-confirmation manifest.
- If any held-out fails, preserve the completed earlier profiles and
  localize the first failed profile before considering a numerical change.
- The fourth duration rung is authorized only after both coarse breadth and
  generic spatial confirmation pass.
